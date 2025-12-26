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
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def construct_camera_basis(camera):
    """Build an orthonormal basis (right, up, forward) from camera vectors."""
    forward = normalize(np.array(camera.look_at) - np.array(camera.position))
    right = normalize(np.cross(forward, np.array(camera.up_vector)))
    up = normalize(np.cross(right, forward))
    return right, up, forward


def get_ray_through_pixel(camera, x, y, image_width, image_height):
    right, up, forward = construct_camera_basis(camera)
    aspect_ratio = image_width / image_height
    screen_height = camera.screen_width / aspect_ratio

    ndc_x = -((x + 0.5) / image_width - 0.5) * 2.0
    ndc_y = -(((y + 0.5) / image_height - 0.5) * 2.0)

    screen_center = np.array(camera.position) + forward * camera.screen_distance
    point_on_screen = (
        screen_center
        + right * ndc_x * camera.screen_width / 2.0
        + up * ndc_y * screen_height / 2.0
    )

    direction = normalize(point_on_screen - np.array(camera.position))
    origin = np.array(camera.position, dtype=float)
    return origin, direction


def trace_ray(ray_origin, ray_direction, geometry, lights, materials, scene_settings, depth):
    if depth <= 0:
        return np.array(scene_settings.background_color, dtype=float)

    hit = find_closest_intersection(ray_origin, ray_direction, geometry)
    if not hit:
        return np.array(scene_settings.background_color, dtype=float)

    color = calculate_lighting(hit, ray_direction, geometry, lights, materials, scene_settings)

    material = materials[hit["object"].material_index]
    if material.reflection_color:
        reflection_strength = float(max(material.reflection_color))
    else:
        reflection_strength = 0.0
    reflection_strength = max(0.0, min(1.0, reflection_strength))

    if reflection_strength > EPSILON and depth > 0:
        reflected_dir = normalize(reflect(ray_direction, hit["normal"]))
        reflected_origin = hit["point"] + hit["normal"] * EPSILON

        # Schlick Fresnel: stronger at glancing angles
        cos_i = max(0.0, -np.dot(ray_direction, hit["normal"]))
        fresnel = reflection_strength + (1.0 - reflection_strength) * ((1.0 - cos_i) ** 5)

        reflected_color = trace_ray(
            reflected_origin,
            reflected_dir,
            geometry,
            lights,
            materials,
            scene_settings,
            depth - 1,
        )

        tint = np.array(material.reflection_color, dtype=float)
        color = color * (1.0 - fresnel) + reflected_color * fresnel * tint

    return np.clip(color, 0.0, 1.0)


def render_scene(camera, scene_settings, objects, image_width, image_height):
    materials = [o for o in objects if isinstance(o, Material)]
    lights = [o for o in objects if isinstance(o, Light)]
    geometry = [o for o in objects if not isinstance(o, (Material, Light))]

    image_array = np.zeros((image_height, image_width, 3), dtype=np.float64)

    print(f"Rendering {image_width}x{image_height}...")
    for y in range(image_height):
        if y % 50 == 0:
            print(f"Row {y}/{image_height}")
        for x in range(image_width):
            origin, direction = get_ray_through_pixel(camera, x, y, image_width, image_height)
            color = trace_ray(
                origin,
                direction,
                geometry,
                lights,
                materials,
                scene_settings,
                int(scene_settings.max_recursions),
            )
            image_array[y, x] = color * 255.0

    return image_array


def save_image(image_array, output_path):
    # Direct save without gamma - colors are already in linear space
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
