import argparse
from PIL import Image
import numpy as np
import time

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from intersections import find_closest_intersection
from lighting import calculate_lighting
from utils import EPSILON, normalize, reflect


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]) - 1)
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]) - 1)
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]) - 1)
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def setup_scene(objects):
    """Split objects into materials, lights, and geometry."""
    materials = [o for o in objects if isinstance(o, Material)]
    lights = [o for o in objects if isinstance(o, Light)]
    geometry = [o for o in objects if not isinstance(o, (Material, Light))]
    return materials, lights, geometry


def construct_ray_grid(camera, width, height):
    """
    Generate rays for the entire image grid using vectorized operations.
    Returns:
        ray_origins: (H, W, 3)
        ray_directions: (H, W, 3)
    """
    # Camera basis
    look_at = np.array(camera.look_at, dtype=np.float32)
    position = np.array(camera.position, dtype=np.float32)
    up_vector = np.array(camera.up_vector, dtype=np.float32)
    
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)
    
    up = np.cross(right, forward)
    # up already normalized if right and forward are orthogonal and normalized
    
    # Screen dimensions
    aspect_ratio = width / height
    screen_width = float(camera.screen_width)
    screen_height = screen_width / aspect_ratio
    screen_dist = float(camera.screen_distance)
    
    screen_center = position + forward * screen_dist
    
    # Generate Grid
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y) # xv, yv are (H, W)
    
    # Normalized coords [-1, 1]
    ndc_x = -((xv + 0.5) / width - 0.5) * 2.0
    ndc_y = -(((yv + 0.5) / height - 0.5) * 2.0)
    
    half_width = screen_width / 2.0
    half_height = screen_height / 2.0
    
    offset_x = right * (ndc_x[..., np.newaxis] * half_width)
    offset_y = up * (ndc_y[..., np.newaxis] * half_height)
    
    points_on_screen = screen_center + offset_x + offset_y
    
    # Directions
    directions = points_on_screen - position
    # Normalize directions
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    
    # Origins are all the same camera position
    origins = np.tile(position, (height, width, 1))
    
    return origins, directions


def refract(incident, normal, ior):
    """
    Vectorized Snell's Law.
    incident: (N, 3) normalized
    normal: (N, 3) normalized
    ior: scalar or (N, 1)
    
    Returns:
        refracted_ray (N, 3) or zeros if specific ray is TIR
        tir_mask (N,) boolean, True if Total Internal Reflection occurred
    """
    # Determine entering or exiting
    # We assume 'ior' is the refractive index of the material.
    # Air is approx 1.0.
    
    cos_i = np.sum(incident * normal, axis=-1, keepdims=True)
    
    entering = cos_i < 0
    
    # If entering: n1 = 1.0, n2 = ior. eta = 1/ior.
    # If exiting: n1 = ior, n2 = 1.0. eta = ior/1.
    
    # Note: epsilon check might be needed for cos_i ~= 0, but generally fine.
    
    eta = np.where(entering, 1.0 / ior, ior)
    n_eff = np.where(entering, normal, -normal)
    
    # cos_i must be positive for the formula relative to the surface normal facing the ray
    # n_eff faces AGAINST incident (mostly) if entering?
    # No, n_eff should point towards the denser medium usually?
    # Standard formula:
    # vector I, vector N (same side as I? No, usually N points out).
    # If N points out, and I points in. dot(I, N) < 0.
    # We want valid refraction.
    
    # Let's use:
    # cos_theta1 = -dot(I, N). (If N points against I, this is positive).
    # If dot > 0 (exiting), we flip N so dot becomes negative.
    
    c1 = -np.sum(incident * n_eff, axis=-1, keepdims=True) # Positive
    
    k = 1.0 - eta * eta * (1.0 - c1 * c1)
    
    tir = (k < 0).flatten()
    
    # T = eta * I + (eta * c1 - sqrt(k)) * N
    # This formula assumes N points OUT (towards I).
    # n_eff points towards I.
    
    sqrt_k = np.sqrt(np.maximum(0, k))
    
    t_vec = eta * incident + (eta * c1 - sqrt_k) * n_eff
    
    return t_vec, tir


def render_scene(camera, scene_settings, objects, image_width, image_height):
    """
    Vectorized render loop with support for separate reflection and transparency rays.
    """
    materials, lights, geometry = setup_scene(objects)
    
    # Helper to build material props
    num_materials = len(materials)
    mat_props = {
        'diffuse': np.zeros((num_materials, 3), dtype=np.float32),
        'specular': np.zeros((num_materials, 3), dtype=np.float32),
        'shininess': np.zeros((num_materials,), dtype=np.float32),
        'reflection': np.zeros((num_materials, 3), dtype=np.float32),
        'transparency': np.zeros((num_materials,), dtype=np.float32),
    }
    
    for i, m in enumerate(materials):
        mat_props['diffuse'][i] = m.diffuse_color
        mat_props['specular'][i] = m.specular_color
        mat_props['shininess'][i] = m.shininess
        mat_props['reflection'][i] = m.reflection_color
        mat_props['transparency'][i] = m.transparency
        
    # Geo index -> Mat index
    geo_mat_indices = np.array([obj.material_index for obj in geometry], dtype=np.int32)
    
    # Initial Rays
    print(f"Generating rays for {image_width}x{image_height}...")
    ray_origins, ray_directions = construct_ray_grid(camera, image_width, image_height)
    
    # Flatten everything to manage a dynamic list of active rays
    active_origins = ray_origins.reshape(-1, 3)
    active_directions = ray_directions.reshape(-1, 3)
    
    # Track which pixel each ray contributes to
    pixel_indices = np.arange(image_width * image_height, dtype=np.int32)
    
    # Current accumulated intensity (Color Multiplier) for the ray path
    active_intensities = np.ones((len(pixel_indices), 3), dtype=np.float32)
    
    # Final Image Buffer (Flat, then reshape at end)
    final_image_flat = np.zeros((len(pixel_indices), 3), dtype=np.float32)
    
    bg_color = np.array(scene_settings.background_color, dtype=np.float32)
    max_depth = int(scene_settings.max_recursions)
    
    # Default IOR for glass if not specified
    # Changed from 1.5 to 1.0 based on user requirement to avoid inverted image
    DEFAULT_IOR = 1.0
    
    print("Starting render loop...")
    for bounce in range(max_depth + 1): 
        num_rays = len(active_origins)
        if num_rays == 0:
            break
            
        print(f"Bounce {bounce}, Active Rays: {num_rays}")
        
        # Intersect
        t, hit_mask, normals, obj_indices = find_closest_intersection(active_origins, active_directions, geometry)
        
        # Handle Misses: Add Background
        missed_mask = ~hit_mask
        if np.any(missed_mask):
            missed_pixels = pixel_indices[missed_mask]
            missed_intensities = active_intensities[missed_mask]
            
            contribution = bg_color * missed_intensities
            # Accumulate using advanced indexing. Note: duplicates in pixel_indices are possible (if we had splitting), 
            # here we have unique rays per pixel effectively? 
            # Actually, splitting creates multiple rays for same pixel. 
            # np.add.at is correct for buffering into shared array with potential duplicates.
            np.add.at(final_image_flat, missed_pixels, contribution)

        # Handle Hits
        hit_idxs = np.where(hit_mask)[0]
        if len(hit_idxs) == 0:
            break
            
        # Get active subset for hits
        hit_origins = active_origins[hit_indices := hit_idxs]
        hit_directions = active_directions[hit_indices]
        hit_indices_geo = obj_indices[hit_indices]
        hit_normals = normals[hit_indices]
        hit_pixel_inds = pixel_indices[hit_indices]
        hit_intensities = active_intensities[hit_indices]
        hit_t = t[hit_indices]
        
        # Map to material
        hit_mat_indices = geo_mat_indices[hit_indices_geo]
        
        # Intersection points
        hit_points = hit_origins + hit_directions * hit_t[..., np.newaxis]
        
        # Calculate Local Lighting (Diffuse + Specular)
        # We need a function that works on flat arrays. lighting.py has been patched to use shape[:-1], so it handles (N, 3).
        
        local_colors = calculate_lighting(
            hit_points, hit_normals, hit_directions, hit_mat_indices, 
            mat_props, geometry, lights, scene_settings
        )
        
        # Add contribution
        np.add.at(final_image_flat, hit_pixel_inds, local_colors * hit_intensities)
        
        if bounce >= max_depth:
            break
            
        # Prepare Next Rays
        # We will separate hits into Reflective and Transparent
        # Note: A surface can be BOTH. 
        # We will check material properties.
        
        ref_colors = mat_props['reflection'][hit_mat_indices] # (N, 3)
        transparency = mat_props['transparency'][hit_mat_indices] # (N,)
        
        next_origins_list = []
        next_directions_list = []
        next_intensities_list = []
        next_pixels_list = []
        
        # 1. Reflection
        # Identify rays that need reflection
        ref_strength = np.max(ref_colors, axis=-1)
        do_reflect = ref_strength > EPSILON
        
        if np.any(do_reflect):
            refl_idxs = np.where(do_reflect)[0]
            
            # Compute reflection vectors
            r_normals = hit_normals[refl_idxs]
            r_dirs = hit_directions[refl_idxs]
            r_points = hit_points[refl_idxs]
            r_intensities = hit_intensities[refl_idxs]
            r_ref_colors = ref_colors[refl_idxs]
            r_pixels = hit_pixel_inds[refl_idxs]
            
            new_dirs = reflect(r_dirs, r_normals)
            new_origins = r_points + r_normals * EPSILON
            new_intensities = r_intensities * r_ref_colors
            
            next_origins_list.append(new_origins)
            next_directions_list.append(new_dirs)
            next_intensities_list.append(new_intensities)
            next_pixels_list.append(r_pixels)
            
        # 2. Transparency / Refraction
        do_trans = transparency > EPSILON
        
        if np.any(do_trans):
            trans_idxs = np.where(do_trans)[0]
            
            t_normals = hit_normals[trans_idxs]
            t_dirs = hit_directions[trans_idxs]
            t_points = hit_points[trans_idxs]
            t_intensities = hit_intensities[trans_idxs]
            t_transparency = transparency[trans_idxs]
            t_pixels = hit_pixel_inds[trans_idxs]
            
            # Refract
            # We use DEFAULT_IOR = 1.0 to prevent "upside down" inversion (User Preference)
            # Ideally could vary per material but stuck with default
            refracted_dirs, tir_mask = refract(t_dirs, t_normals, DEFAULT_IOR)
            
            # For TIR, typically we add to reflection, but we might have already added reflection above.
            # If TIR happens, energy should technically go to reflection.
            # But duplicate handling is complex.
            # Simple approach: If TIR, ignore trace (energy lost or assumed absorbed/handled by reflection pass).
            # If NOT TIR, trace refracted ray.
            
            valid_refract = ~tir_mask
            
            if np.any(valid_refract):
                v_dirs = refracted_dirs[valid_refract]
                # Offset origin? 
                # Points are ON surface. Refracted ray goes IN.
                # We need to nudge INWARDS (along refracted dir)
                v_points = t_points[valid_refract]
                v_pixel_inds = t_pixels[valid_refract]
                
                # Nudge
                v_origins = v_points + v_dirs * EPSILON
                
                # Scale intensity
                # t_transparency is scalar.
                v_int_in = t_intensities[valid_refract]
                v_trans = t_transparency[valid_refract][:, np.newaxis]
                v_new_intensities = v_int_in * v_trans
                
                next_origins_list.append(v_origins)
                next_directions_list.append(v_dirs)
                next_intensities_list.append(v_new_intensities)
                next_pixels_list.append(v_pixel_inds)
        
        # Combine lists
        if not next_origins_list:
            break
            
        active_origins = np.concatenate(next_origins_list, axis=0)
        active_directions = np.concatenate(next_directions_list, axis=0)
        active_intensities = np.concatenate(next_intensities_list, axis=0)
        pixel_indices = np.concatenate(next_pixels_list, axis=0)
        
    return final_image_flat.reshape(image_height, image_width, 3)


def save_image(image_array, output_path):
    """Save rendered image to file."""
    # Clip and convert
    image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    start_time = time.time()
    image_array = render_scene(camera, scene_settings, objects, args.width, args.height)
    end_time = time.time()
    print(f"Render completed in {end_time - start_time:.2f} seconds.")

    save_image(image_array * 255.0, args.output_image)


if __name__ == '__main__':
    main()
