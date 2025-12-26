import numpy as np

from intersections import find_closest_intersection
from utils import EPSILON, clamp_color, normalize, reflect


def compute_shadow(point, light, geometry):
    """Calculate soft shadow with stratified sampling for smooth penumbra."""
    light_pos = np.array(light.position, dtype=float)
    to_light = light_pos - point
    distance = np.linalg.norm(to_light)
    direction = to_light / distance
    
    # If light has no radius, do hard shadow only
    if light.radius < EPSILON:
        origin = point + direction * EPSILON
        hit = find_closest_intersection(origin, direction, geometry)
        if hit and hit["distance"] < distance - EPSILON:
            return 0.0
        return 1.0
    
    # Stratified sampling in a disk for smooth shadows
    # Increase samples for softer shadows while keeping cost reasonable
    visible = 0
    total = 12  # smoother penumbra with moderate cost
    
    # Create orthonormal basis for light
    up = np.array([0, 1, 0]) if abs(direction[1]) < 0.9 else np.array([1, 0, 0])
    right = normalize(np.cross(direction, up))
    up = normalize(np.cross(right, direction))
    
    for i in range(total):
        # Stratified disk sampling
        angle = (i / total) * 2.0 * np.pi
        radius = light.radius * np.sqrt((i + 0.5) / total)  # Uniform area distribution
        
        offset = radius * (right * np.cos(angle) + up * np.sin(angle))
        sample_pos = light_pos + offset
        
        to_sample = sample_pos - point
        sample_dist = np.linalg.norm(to_sample)
        sample_dir = to_sample / sample_dist
        
        shadow_origin = point + sample_dir * EPSILON
        hit = find_closest_intersection(shadow_origin, sample_dir, geometry)
        if not hit or hit["distance"] >= sample_dist - EPSILON:
            visible += 1
    
    return visible / total


def calculate_lighting(intersection, ray_direction, geometry, lights, materials, scene_settings):
    """Calculate Phong lighting at intersection point."""
    point = intersection["point"]
    normal = intersection["normal"]
    
    # Handle material index out of range gracefully
    mat_idx = intersection["object"].material_index
    if mat_idx >= len(materials):
        mat_idx = len(materials) - 1
    material = materials[mat_idx]
    
    # Ambient component
    diff_color = np.array(material.diffuse_color, dtype=float)
    spec_color = np.array(material.specular_color, dtype=float)
    color = diff_color * 0.2  # Ambient light
    
    view_dir = normalize(-ray_direction)
    
    for light in lights:
        light_pos = np.array(light.position, dtype=float)
        light_color = np.array(light.color, dtype=float)
        
        # Diffuse
        to_light = light_pos - point
        light_dir = normalize(to_light)
        ndotl = max(0.0, np.dot(normal, light_dir))
        
        if ndotl < EPSILON:
            continue
        
        # Soft shadow calculation
        visibility = compute_shadow(point, light, geometry)
        shadow_factor = 1.0 - (1.0 - visibility) * light.shadow_intensity
        
        if shadow_factor < EPSILON:
            continue
        
        # Add diffuse
        diffuse = ndotl * diff_color * light_color * shadow_factor
        color += diffuse
        
        # Specular (Phong)
        reflect_dir = reflect(-light_dir, normal)
        spec_dot = max(0.0, np.dot(reflect_dir, view_dir))
        if spec_dot > 0:
            specular = (spec_dot ** material.shininess) * spec_color * light_color
            specular *= light.specular_intensity * shadow_factor
            color += specular
    
    return np.clip(color, 0.0, 1.0)
