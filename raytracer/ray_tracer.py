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
from vector import Vector
from ray import Ray
from scene_settings import SceneSettings
from light import Light



EPSILON = 1e-6 #Small constant used for acne (self-intersection) prevention


def parse_scene_file(file_path):
    """
    Parse a scene file, construct all the scene objects and return the camera, scene settings, and objects.
    Remained unchanged from the original skeleton.

    Args:
        file_path (str): The path to the scene file.
    
    Returns:
        tuple: A tuple containing the camera, scene settings, and objects.
    """
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


def save_image(image_array, output_image):
    """
    Save the image to a file - recieves image array(np array) and output image path

    Args:
        image_array (np.array): The image array to save.
        output_image (str): The path to save the image to.
    """
    image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))

    # Save the image to a file
    image.save(output_image)

def find_closest_intersection(ray, intersectable):
    """
    Find the closest intersection point between a ray and a list of intersectable objects.

    Iterates over all objects in the scene and checks for intersection with the ray.
    Returns the closest intersection point, the normal at that point, and the object that was intersected.

    Args:
        ray (Ray): The ray to check for intersection.
        intersectable (list): A list of objects that can be intersected by the ray

    Returns:
        tuple: A tuple containing the closest intersection point(Vector or None), the object that was intersected(Object or None), the normal at that point(Vector or None). 
    """
    closest_intersection = None
    closest_normal = None
    closest_object = None
    min_dist = float('inf')

    for obj in intersectable:
        result = obj.ray_interception(ray)
        if result:
            intersection, normal = result
            dist = (intersection - ray.origin).magnitude()
            if dist < min_dist:
                min_dist = dist
                closest_intersection = intersection
                closest_normal = normal
                closest_object = obj
    
    return closest_intersection, closest_object, closest_normal

def trace_ray(ray, intersectable, lights, materials, scene_settings, remaining_recursions=None):
    """
    Trace a ray through the scene and return the color of the pixel at the intersection point.
    Recursively traces the ray through the scene, following reflections, transparency and shadows.

    Args:
        ray (Ray): The ray to trace.
        intersectable (list): A list of objects that can be intersected by the ray
        lights (list): A list of light sources in the scene
        materials (list): A list of materials in the scene
        scene_settings (SceneSettings): The scene settings
        remaining_recursions (int, optional): The number of remaining recursions. Defaults to None.

    Returns:
        Vector: The color of the pixel at the intersection point.
    """
    if remaining_recursions is None:
        remaining_recursions = int(scene_settings.max_recursions)
    if remaining_recursions < 0:
        return Vector(*scene_settings.background_color)

    intersection, hit_object, normal = find_closest_intersection(ray, intersectable)

    if not hit_object:
        return Vector(*scene_settings.background_color) # Background color is returned if no intersection

    # Flip normal if ray is coming from inside the object(to enable tracing with transparent objects)
    if ray.direction.dot(normal) > 0:
        normal = -normal

    # Material properties
    material = materials[hit_object.material_index - 1]
    
    # Calculate offset points to avoid self-intersection (Acne)
    hit_point_out = intersection + normal * EPSILON 
    hit_point_in = intersection - normal * EPSILON

    current_color = Vector(0, 0, 0) 
    total_reflection = Vector(0, 0, 0)

    # Local Lighting (Shadows + Diffuse + Specular)
    for light in lights:
        light_dir = (light.position - intersection).normalize() # Direction from intersection to light
        light_distance = (light.position - intersection).magnitude() # Distance from intersection to light
        
        # Shadow / visibility check
        N = int(scene_settings.root_number_shadow_rays)
        shadow_intensity = 0.0  # 0.0 = fully blocked, 1.0 = fully visible

        if N <= 1 or light.radius < EPSILON:
            # Hard shadow (single ray toward light center)
            shadow_ray = Ray(hit_point_out, light_dir)
            shadow_hit, _, _ = find_closest_intersection(shadow_ray, intersectable)
            if shadow_hit and (shadow_hit - hit_point_out).magnitude() < light_distance:
                shadow_intensity = 0.0
            else:
                shadow_intensity = 1.0
        else:
            # Soft shadows:  N×N sampling over an area light
            total_samples = N * N
            unblocked_count = 0
            L_vec = (intersection - light.position).normalize() # Plane perpendicular to direction from light to hit point

            arbitrary_up = Vector(0, 1, 0) # An arbitrary vector perpendicular to L_vec
            if abs(L_vec.dot(arbitrary_up)) > 0.99: # If L_vec is parallel to arbitrary_up, choose a different arbitrary_up
                arbitrary_up = Vector(1, 0, 0)

            # Create a coordinate system for the light plane
            light_u = L_vec.cross(arbitrary_up).normalize()
            light_v = L_vec.cross(light_u).normalize()

            cell_size = light.radius / N
            for i in range(N): # Choose random points in a grid of N x N cells inside the light radius
                for j in range(N):
                    rand_u = (i + np.random.rand()) * cell_size - (light.radius / 2) 
                    rand_v = (j + np.random.rand()) * cell_size - (light.radius / 2)

                    light_sample = light.position + light_u * rand_u + light_v * rand_v
                    sample_dir = (light_sample - intersection).normalize()
                    sample_dist = (light_sample - intersection).magnitude()

                    shadow_ray = Ray(hit_point_out, sample_dir)
                    shadow_hit, _, _ = find_closest_intersection(shadow_ray, intersectable)
                    if not shadow_hit or (shadow_hit - hit_point_out).magnitude() > sample_dist:
                        unblocked_count += 1

            light_hit_ratio = unblocked_count / total_samples

        light_intensity = (1.0 - light.shadow_intensity) + light.shadow_intensity * light_hit_ratio

        if light_intensity > 0:
            # Diffuse
            diffuse_factor = max(0, normal.dot(light_dir))
            diffuse_contribution = material.diffuse_color * light.color * diffuse_factor
            
            # Specular
            specular_contribution = Vector(0,0,0)
            if material.shininess > 0:
                reflect_vector = -(light_dir.reflect(normal))
                view_dir = (ray.origin - intersection).normalize()
                
                specular_factor = max(0, view_dir.dot(reflect_vector)) ** material.shininess
                specular_contribution = light.color * material.specular_color * specular_factor * light.specular_intensity 
            
            # Combine Diffuse + Specular
            current_color += (diffuse_contribution + specular_contribution) * light_intensity


    # Recursion: Reflection
    if material.reflection_color.magnitude() > 0:
        reflect_dir = (ray.direction.reflect(normal))
        reflected_ray = Ray(hit_point_out, reflect_dir)
        if remaining_recursions <= 0:
            reflected_color = Vector(*scene_settings.background_color)
        else:
            reflected_color = trace_ray(
                reflected_ray,
                intersectable,
                lights,
                materials,
                scene_settings,
                remaining_recursions=remaining_recursions - 1,
            )
        total_reflection = reflected_color * material.reflection_color

    # Recursion: Transparency
    transparency_color = Vector(0, 0, 0)
    if material.transparency > 0:
        transparency_ray = Ray(hit_point_in, ray.direction)
        if remaining_recursions <= 0:
            transparency_color = Vector(*scene_settings.background_color)
        else:
            transparency_color = trace_ray(
                transparency_ray,
                intersectable,
                lights,
                materials,
                scene_settings,
                remaining_recursions=remaining_recursions - 1,
            )

    # Final Color Mixing Formula
    # Output = (Background * Trans) + (Diffuse + Specular) * (1 - Trans) + Reflection
    # 'current_color' currently contains (Diffuse + Specular) from all lights
    # Reflection is added independently, not affected by transparency
    
    final_color = (transparency_color * material.transparency) + \
                  (current_color * (1 - material.transparency)) + \
                  total_reflection
    
    return final_color


def main():

    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    # Create intersectable list
    intersectable = [obj for obj in objects if isinstance(obj, (Sphere, Cube, InfinitePlane))]
    # Create a lights list
    lights = [obj for obj in objects if isinstance(obj, Light)]
    # Create materials list
    materials = [obj for obj in objects if isinstance(obj, Material)]

    # Result array
    image_array = np.zeros((args.height, args.width, 3), dtype=np.float32)
    
    # Pre-processing : calculate camera vectors
    forward_vector = (camera.look_at - camera.position).normalize()
    right_vector = forward_vector.cross(camera.up_vector).normalize()
    up_vector = right_vector.cross(forward_vector).normalize()
    
    # Main rendering loop
    for y in range(args.height):
        for x in range(args.width):
            # args.width - 1 - x, args.height - 1 - y Flips the image axis to match the camera's coordinate system
            ray = camera.get_ray(args.width - 1 - x, args.height - 1 - y, args.width, args.height, forward_vector, right_vector, up_vector)
            
            final_color = trace_ray(
                ray,
                intersectable,
                lights,
                materials,
                scene_settings,
                remaining_recursions=int(scene_settings.max_recursions),
            )
            
            # Store in image
            image_array[y, x] = final_color.to_rgb() 

    # Save the output image
    save_image(image_array, args.output_image)

if __name__ == '__main__':
    main()
