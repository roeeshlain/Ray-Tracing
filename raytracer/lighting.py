import numpy as np
from intersections import find_closest_intersection
from utils import EPSILON, normalize, reflect

def compute_shadow(points, light, geometry, scene_settings, geo_transparency):
    """
    Vectorized soft shadow calculation.
    points: (H, W, 3)
    Returns: shadow_intensity (H, W) - 0.0 means fully shadowed, 1.0 means fully lit
    """
    light_pos = np.array(light.position, dtype=np.float32)
    radius = float(light.radius)
    rays_root = int(scene_settings.root_number_shadow_rays)
    total_samples = rays_root * rays_root
    
    # Vector to light center
    to_light_center = light_pos - points
    dist_to_light_center = np.linalg.norm(to_light_center, axis=-1)
    
    # Early out if no samples
    if total_samples <= 0:
        return np.ones(points.shape[:-1], dtype=np.float32)
        
    # Hard shadows optimization (radius ~= 0)
    if radius < EPSILON:
        # Single ray to center
        direction = to_light_center / (dist_to_light_center[..., np.newaxis] + EPSILON)
        # Shift origin slightly
        shadow_origins = points + direction * EPSILON
        
        t, mask, _, hit_obj_indices = find_closest_intersection(shadow_origins, direction, geometry)
        
        # Visible if no hit closer than light distance
        # BUT if hit object is transparent, we treat it as NOT visually blocking (approximated)
        # We need Object Index from intersection to know material.
        
        blocked_by_geo = mask & (t < dist_to_light_center - EPSILON)
        
        # Check transparency of blocker
        # hit_obj_indices is -1 where no hit, but mask handles that.
        # where blocked_by_geo is True, hit_obj_indices is valid.
        
        # Safe indexing
        safe_indices = np.maximum(0, hit_obj_indices)
        blocker_transparency = geo_transparency[safe_indices] # (H, W)
        
        # "Is Blocked" means: Hit Geometry AND Dist < LightDist AND Transparency < Threshold
        # If transparency is high (e.g. > 0), we treat it as unblocked (light passes).
        # We could modulate light by (1-transp), but "unblocked" is cleaner for "is shadow?"
        # Let's say if transparency > 0.01, it doesn't cast a shadow.
        
        is_transparent_blocker = blocker_transparency > EPSILON
        
        real_block = blocked_by_geo & (~is_transparent_blocker)
        
        return np.where(real_block, 0.0, 1.0)
    
    # Soft shadows
    # We need to accumulate visibility
    visible_count = np.zeros(points.shape[:-1], dtype=np.float32)
    
    # Basis for light disk ... (REMOVED COMMENT BLOCK FOR BREVITY) ...
    
    # Pre-generate stratified samples on unit disk (2D)
    # shape: (total_samples, 2)
    samples_2d = []
    for i in range(total_samples):
        # Stratified sampling
        r_idx = i // rays_root
        c_idx = i % rays_root
        
        # Jitter or center? Prompt used specific formula with i/total
        angle = (i / total_samples) * 2.0 * np.pi
        r_scale = np.sqrt((i + 0.5) / total_samples)
        samples_2d.append((r_scale * np.cos(angle), r_scale * np.sin(angle)))
    
    # Loop over samples
    for rx, ry in samples_2d:
        # Let's compute basis vectors once
        # normalized to_light
        light_dir = to_light_center / (dist_to_light_center[..., np.newaxis] + EPSILON)
        
        # Safe up vector
        global_up = np.tile(np.array([0,1,0], dtype=np.float32), light_dir.shape[:-1] + (1,))
        use_x = np.abs(light_dir[..., 1]) >= 0.9
        global_up[use_x] = np.array([1,0,0], dtype=np.float32)
        
        right = np.cross(light_dir, global_up)
        right = right / (np.linalg.norm(right, axis=-1, keepdims=True) + EPSILON)
        
        up = np.cross(right, light_dir)
        up = up / (np.linalg.norm(up, axis=-1, keepdims=True) + EPSILON)
        
        offset = radius * (right * rx + up * ry)
        target = light_pos + offset
        to_sample = target - points
        dist_to_sample = np.linalg.norm(to_sample, axis=-1)
        sample_dir = to_sample / (dist_to_sample[..., np.newaxis] + EPSILON)
        
        # Cast ray
        shadow_origins = points + sample_dir * EPSILON
        t, mask, _, hit_obj_indices = find_closest_intersection(shadow_origins, sample_dir, geometry)
        
        # Check blockage
        blocked_by_geo = mask & (t < dist_to_sample - EPSILON)
        
        # Check transparency
        safe_indices = np.maximum(0, hit_obj_indices)
        blocker_transparency = geo_transparency[safe_indices]
        
        is_transparent_blocker = blocker_transparency > EPSILON
        
        real_block = blocked_by_geo & (~is_transparent_blocker)
        
        visible_count += np.where(real_block, 0.0, 1.0)
        
    return visible_count / total_samples


def calculate_lighting(points, normals, ray_directions, object_indices, material_props, geometry, lights, scene_settings):
    """
    Vectorized Phong lighting.
    
    points: (H, W, 3)
    normals: (H, W, 3)
    ray_directions: (H, W, 3) (Ray FROM camera/bounce)
    object_indices: (H, W)
    material_props: dict containing arrays:
        - 'diffuse': (N, 3)
        - 'specular': (N, 3)
        - 'shininess': (N,)
        - 'reflection': (N, 3)
        - 'transparency': (N,)
    """
    
    # Gather material properties for all pixels
    safe_indices = np.maximum(0, object_indices)
    
    diffuse_colors = material_props['diffuse'][safe_indices] # (H, W, 3)
    specular_colors = material_props['specular'][safe_indices]
    shininess = material_props['shininess'][safe_indices][..., np.newaxis] # (H, W, 1)
    
    # Pre-compute Geometry Transparency Map for Shadows
    # Mapping: Geometry Index -> Material Transparency
    # geometry is list of objects. Each obj has 'material_index'.
    # material_props['transparency'] is list of transp values by Mat Id.
    
    # Build list of transparencies aligned with geometry list indices
    geo_mat_indices = [obj.material_index for obj in geometry]
    geo_transparency_list = [material_props['transparency'][midx] for midx in geo_mat_indices]
    
    # Convert to array for fast lookup by find_closest_intersection return indices
    geo_transparency_arr = np.array(geo_transparency_list, dtype=np.float32)
    
    
    # Ambient
    color = diffuse_colors * 0.2
    
    # View direction (V) = -RayDir
    view_dir = -ray_directions
    # Normalize view_dir to prevent drift
    view_dir = view_dir / (np.linalg.norm(view_dir, axis=-1, keepdims=True) + EPSILON)
    
    for light in lights:
        light_pos = np.array(light.position, dtype=np.float32)
        light_color = np.array(light.color, dtype=np.float32)
        
        # Vector to light (L)
        to_light = light_pos - points
        dist_sq = np.sum(to_light * to_light, axis=-1, keepdims=True)
        dist = np.sqrt(dist_sq)
        light_dir = to_light / (dist + EPSILON)
        
        # N dot L
        ndotl = np.sum(normals * light_dir, axis=-1, keepdims=True)
        # Mask for facing light
        facing_light = ndotl > 0
        
        # Calculate shadow only where facing light
        # Optimization: We could skip shadow calc for back-facing pixels
        # But for vectorization simplicity, we might run it everywhere or mask inputs?
        # running everywhere is easier code, maybe slower.
        # Let's compute shadow everywhere for now, straightforward.
        
        shadow_visibility = compute_shadow(points, light, geometry, scene_settings, geo_transparency_arr)
        # shadow_visibility is (H, W) -> make (H, W, 1)
        visibility = shadow_visibility[..., np.newaxis]
        
        # Light intensity factor (shadow + user intensity)
        # shadow_factor = 1.0 - (1.0 - visibility) * light.shadow_intensity
        shadow_factor = 1.0 - (1.0 - visibility) * float(light.shadow_intensity)
        
        # Combined mask: facing light AND shadow factor > 0
        effective_light = (ndotl * shadow_factor) # (H, W, 1)
        
        # Diffuse term
        # diffuse = diff_color * light_color * ndotl * shadow_factor
        diffuse_term = diffuse_colors * light_color * effective_light
        
        # Specular term (Phong)
        # R = reflect(-L, N)
        # reflect takes input, normal. reflect(-L) is wrong. reflect(I, N).
        # We want reflection of Light Vector? No, standard Phong uses Reflection of Light Vector around Normal?
        # Or Reflection of View Vector?
        # "reflect_dir = reflect(-light_dir, normal)" from original code.
        # Original: reflect_dir = reflect(-light_dir, normal) -> Incoming vector is -L.
        # Correct.
        r_vector = reflect(-light_dir, normals)
        
        # R dot V
        rdotv = np.sum(r_vector * view_dir, axis=-1, keepdims=True)
        # Clamp to [0, 1] to avoid explosion
        rdotv = np.clip(rdotv, 0.0, 1.0)
        
        specular_term = (rdotv ** shininess) * specular_colors * light_color
        specular_term *= float(light.specular_intensity) * shadow_factor
        
        # Add to accumulator where facing light
        # Using np.where to avoiding adding garbage from back faces
        # (Though ndotl < 0 clamp handles diffuse, specular check rdotv > 0 handles specular)
        
        final_contribution = np.where(facing_light, diffuse_term + specular_term, 0.0)
        
        color += final_contribution
        
    return np.clip(color, 0.0, 1.0)
