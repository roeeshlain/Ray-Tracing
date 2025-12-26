import numpy as np

from intersections import find_closest_intersection
from utils import EPSILON, clamp_color, normalize, reflect


def _sample_point_in_sphere(radius, rng):
    """Sample a random point inside a sphere of given radius."""
    # Use Marsaglia method: random direction * random radius^(1/3)
    while True:
        v = rng.normal(size=3)
        norm = np.linalg.norm(v)
        if norm > 0:
            dir_unit = v / norm
            break
    r = radius * (rng.random() ** (1.0 / 3.0))
    return dir_unit * r


def _shadow_visibility(point, light, geometry, samples):
    """Return fraction of unoccluded shadow rays (soft shadows)."""
    if samples <= 1 or light.radius <= EPSILON:
        direction = normalize(np.array(light.position) - point)
        distance_to_light = np.linalg.norm(np.array(light.position) - point)
        shadow_origin = point + direction * EPSILON
        hit = find_closest_intersection(shadow_origin, direction, geometry)
        return 0.0 if (hit and hit["distance"] < distance_to_light) else 1.0

    # Stratified sampling across a disk approximation of the light sphere for lower noise
    rng = np.random.default_rng()
    side = int(np.sqrt(samples))
    if side < 1:
        side = 1
    visible = 0
    total = side * side
    base_pos = np.array(light.position)
    for i in range(side):
        for j in range(side):
            # jitter within each cell
            u = (i + rng.random()) / side
            v = (j + rng.random()) / side
            # map square to disk, then to sphere surface (simple disk for speed)
            r = light.radius * np.sqrt(u)
            theta = 2 * np.pi * v
            offset = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            # random rotation around up-axis for more variation
            rot_theta = rng.uniform(0, 2 * np.pi)
            rot = np.array([
                [np.cos(rot_theta), 0, -np.sin(rot_theta)],
                [0, 1, 0],
                [np.sin(rot_theta), 0, np.cos(rot_theta)],
            ])
            jitter = rot @ offset
            target = base_pos + jitter
            direction = normalize(target - point)
            distance_to_light = np.linalg.norm(target - point)
            shadow_origin = point + direction * EPSILON
            hit = find_closest_intersection(shadow_origin, direction, geometry)
            if not hit or hit["distance"] >= distance_to_light:
                visible += 1
    return visible / total


def calculate_lighting(intersection, ray_direction, geometry, lights, materials, scene_settings):
    point = intersection["point"]
    normal = intersection["normal"]
    obj = intersection["object"]
    material = materials[obj.material_index]

    # Minimal ambient - let lights do the work
    color = np.array([0.0, 0.0, 0.0])

    view_dir = normalize(-ray_direction)
    # Use squared sample count with a reasonable floor/ceiling for smooth penumbra
    base_samples = int(scene_settings.root_number_shadow_rays)
    shadow_samples = max(25, base_samples * base_samples)
    shadow_samples = min(shadow_samples, 49)

    for light in lights:
        light_dir = normalize(np.array(light.position) - point)
        ndotl = max(np.dot(normal, light_dir), 0.0)
        if ndotl <= 0:
            continue

        visibility = _shadow_visibility(point, light, geometry, shadow_samples)
        # Apply shadow intensity: visibility 0 = full shadow, visibility 1 = full light
        shadow_factor = 1.0 - (1.0 - visibility) * light.shadow_intensity
        if shadow_factor <= EPSILON:
            continue

        diffuse = ndotl * shadow_factor * np.array(material.diffuse_color) * np.array(light.color)

        reflect_dir = normalize(reflect(-light_dir, normal))
        spec_angle = max(np.dot(reflect_dir, view_dir), 0.0)
        specular = (spec_angle ** material.shininess) * shadow_factor
        specular *= light.specular_intensity
        specular_color = specular * np.array(material.specular_color) * np.array(light.color)

        color += diffuse + specular_color

    return clamp_color(color)
