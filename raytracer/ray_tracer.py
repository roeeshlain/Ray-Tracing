import argparse
from PIL import Image
import numpy as np

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


def construct_camera_basis(camera):
    """Build orthonormal basis for camera."""
    forward = normalize(np.array(camera.look_at) - np.array(camera.position))
    right = normalize(np.cross(forward, camera.up_vector))
    up = np.cross(right, forward)  # Already normalized
    return right, up, forward


def compute_ndc(x, y, width, height):
    """Convert pixel coords to normalized device coords [-1, 1]."""
    ndc_x = -((x + 0.5) / width - 0.5) * 2.0
    ndc_y = -(((y + 0.5) / height - 0.5) * 2.0)
    return ndc_x, ndc_y


def get_ray_through_pixel(camera, x, y, image_width, image_height, right, up, forward, cam_pos, screen_center, half_width, half_height):
    """Generate ray through pixel using precomputed values."""
    ndc_x, ndc_y = compute_ndc(x, y, image_width, image_height)
    point_on_screen = screen_center + right * (ndc_x * half_width) + up * (ndc_y * half_height)
    direction = normalize(point_on_screen - cam_pos)
    return cam_pos, direction


def compute_reflection(material, ray_dir, normal, hit_point, geometry, lights, materials, scene_settings, depth, bg_color):
    """Calculate reflection contribution for shiny surfaces."""
    reflect_color = material.reflection_color
    if not reflect_color or max(reflect_color) < EPSILON:
        return None, 0.0
    
    # Pool balls need strong, consistent reflections - no Fresnel falloff
    strength = max(reflect_color)
    reflected_dir = reflect(ray_dir, normal)
    reflected_origin = hit_point + normal * EPSILON
    
    reflected_color = trace_ray(
        reflected_origin,
        reflected_dir,
        geometry,
        lights,
        materials,
        scene_settings,
        depth - 1,
        bg_color,
    )
    
    return reflected_color * np.array(reflect_color), strength


def trace_ray(ray_origin, ray_direction, geometry, lights, materials, scene_settings, depth, bg_color):
    """Trace single ray through scene."""
    if depth <= 0:
        return bg_color

    hit = find_closest_intersection(ray_origin, ray_direction, geometry)
    if not hit:
        return bg_color

    color = calculate_lighting(hit, ray_direction, geometry, lights, materials, scene_settings)
    
    # Handle material index out of range gracefully
    mat_idx = hit["object"].material_index
    if mat_idx >= len(materials):
        mat_idx = len(materials) - 1
    material = materials[mat_idx]

    if depth > 0:
        reflected, strength = compute_reflection(
            material, ray_direction, hit["normal"], hit["point"],
            geometry, lights, materials, scene_settings, depth, bg_color
        )
        if reflected is not None:
            # Simple blend: more reflection, less surface color
            color = color * (1.0 - strength) + reflected * strength

    return np.clip(color, 0.0, 1.0)


def setup_scene(objects):
    """Split objects into materials, lights, and geometry."""
    materials = [o for o in objects if isinstance(o, Material)]
    lights = [o for o in objects if isinstance(o, Light)]
    geometry = [o for o in objects if not isinstance(o, (Material, Light))]
    return materials, lights, geometry


def precompute_camera_data(camera, image_width, image_height):
    """Precompute camera values used for every pixel."""
    right, up, forward = construct_camera_basis(camera)
    cam_pos = np.array(camera.position, dtype=float)
    screen_center = cam_pos + forward * camera.screen_distance
    
    aspect_ratio = image_width / image_height
    screen_height = camera.screen_width / aspect_ratio
    half_width = camera.screen_width / 2.0
    half_height = screen_height / 2.0
    
    return right, up, forward, cam_pos, screen_center, half_width, half_height


def render_scene(camera, scene_settings, objects, image_width, image_height):
    """Render the scene to an image array."""
    materials, lights, geometry = setup_scene(objects)
    image_array = np.zeros((image_height, image_width, 3), dtype=np.float32)
    
    # Precompute camera values once
    right, up, forward, cam_pos, screen_center, half_width, half_height = precompute_camera_data(camera, image_width, image_height)
    bg_color = np.array(scene_settings.background_color, dtype=float)
    max_depth = int(scene_settings.max_recursions)

    print(f"Rendering {image_width}x{image_height}...")
    for y in range(image_height):
        if y % 50 == 0:
            print(f"Row {y}/{image_height}")
        for x in range(image_width):
            origin, direction = get_ray_through_pixel(
                camera, x, y, image_width, image_height,
                right, up, forward, cam_pos, screen_center, half_width, half_height
            )
            color = trace_ray(origin, direction, geometry, lights, materials, scene_settings, max_depth, bg_color)
            image_array[y, x] = color * 255.0

    return image_array


def save_image(image_array, output_path):
    """Save rendered image to file."""
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

    image_array = render_scene(camera, scene_settings, objects, args.width, args.height)

    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
