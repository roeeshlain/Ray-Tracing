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



EPSILON = 1e-6


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


def save_image(image_array, output_image):
    image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))

    # Save the image to a file
    image.save(output_image)

def find_closest_intersection(ray, intersectable):
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

def trace_ray(ray, intersectable, lights, materials, scene_settings, depth=0):
    if depth > scene_settings.max_recursions: 
        return Vector(0, 0, 0)

    intersection, hit_object, normal = find_closest_intersection(ray, intersectable)

    if not hit_object:
        return Vector(*scene_settings.background_color) # Background reflection

    # Adjust normal for direction (if hitting from inside)
    if ray.direction.dot(normal) > 0:
        normal = -normal

    # Material properties
    material = materials[hit_object.material_index - 1]
    
    # Calculate offset points to avoid self-intersection (Acne)
    hit_point_out = intersection + normal * EPSILON #this point is epsilon far in the direction of normal, to avoid self-intersection
    hit_point_in = intersection - normal * EPSILON

    current_color = Vector(0, 0, 0)
    total_reflection = Vector(0, 0, 0)

    # Local Lighting (Shadows + Diffuse + Specular)
    for light in lights:
        light_dir = (light.position - intersection).normalize()
        light_distance = (light.position - intersection).magnitude()
        
        # Shadow Check
        shadow_intensity = 0.0 # 0 = fully in shadow, 1 = fully lit
        
        if light.radius < EPSILON: # Hard Shadow
            # Direction TO light
            shadow_ray = Ray(hit_point_out, light_dir)
            shadow_hit, shadowed_obj, _ = find_closest_intersection(shadow_ray, intersectable)
            # Check if hit object is closer than light
            if shadow_hit and (shadow_hit - hit_point_out).magnitude() < light_distance:
                shadow_intensity = 0.0
            else:
                shadow_intensity = 1.0
        else: # Soft Shadow
            total_samples = scene_settings.root_number_shadow_rays ** 2
            unblocked_count = 0
            
            # 1. Coordinate System for the Area Light
            # We need a plane perpendicular to the direction from Light to Hit Point.
            # Vector from Light to Hit Point:
            L_vec = (intersection - light.position).normalize()
            
            # Find an arbitrary vector distinct from L_vec to compute cross product
            arbitrary_up = Vector(0, 1, 0)
            if abs(L_vec.dot(arbitrary_up)) > 0.99: # If parallel to vertical, pick horizontal
                 arbitrary_up = Vector(1, 0, 0)
            
            # Basis vectors for the light plane
            light_u = L_vec.cross(arbitrary_up).normalize()
            light_v = L_vec.cross(light_u).normalize()
            
            # Grid width is light.radius (as per user request: "wide as the defined light radius")
            # Usually radius implies width=2*radius, but user said "as wide as radius".
            # Assuming radius is the *extent* of the square, or the side length = radius.
            cell_size = light.radius / scene_settings.root_number_shadow_rays

            for i in range(int(scene_settings.root_number_shadow_rays)):
                for j in range(int(scene_settings.root_number_shadow_rays)):
                    # Random jitter within the cell
                    rand_u = (i + np.random.rand()) * cell_size - (light.radius / 2)
                    rand_v = (j + np.random.rand()) * cell_size - (light.radius / 2)
                    
                    # Calculate sample position on the light plane relative to light center
                    light_sample = light.position + light_u * rand_u + light_v * rand_v
                    
                    sample_dir = (light_sample - intersection).normalize()
                    sample_dist = (light_sample - intersection).magnitude()
                    
                    shadow_ray = Ray(hit_point_out, sample_dir)
                    shadow_hit, shadowed_obj, _ = find_closest_intersection(shadow_ray, intersectable)
                    
                    if not shadow_hit or (shadow_hit - hit_point_out).magnitude() > sample_dist:
                         unblocked_count += 1
            
            shadow_intensity = unblocked_count / total_samples

        if shadow_intensity > 0:
            # Diffuse
            diffuse_factor = max(0, normal.dot(light_dir))
            diffuse_contribution = material.diffuse_color * light.color * diffuse_factor
            
            # Specular
            specular_contribution = Vector(0,0,0)
            if material.shininess > 0:
                reflect_vector = -(light_dir.reflect(normal))
                view_dir = (ray.origin - intersection).normalize()
                
                specular_factor = max(0, view_dir.dot(reflect_vector)) ** material.shininess
                specular_contribution = light.color * material.specular_color * specular_factor 
            
            # Combine Diffuse + Specular
            current_color += (diffuse_contribution + specular_contribution) * shadow_intensity


    # Recursion: Reflection
    if material.reflection_color.magnitude() > 0:
        reflect_dir = (ray.direction.reflect(normal)) 
        reflected_ray = Ray(hit_point_out, reflect_dir)
        reflected_color = trace_ray(reflected_ray, intersectable, lights, materials, scene_settings, depth + 1)
        total_reflection = reflected_color * material.reflection_color

    # Recursion: Transparency
    transparency_color = Vector(0,0,0)
    if material.transparency > 0:
        transparency_ray = Ray(hit_point_in, ray.direction) 
        transparency_color = trace_ray(transparency_ray, intersectable, lights, materials, scene_settings, depth + 1)

    # 2. Final Color Mixing Formula
    # IN THE INSTRUCTIONS PDF: Output = (Background * Trans) + (Diffuse + Specular) * (1 - Trans) + Reflection
    # 'current_color' currently contains (Diffuse + Specular) from all lights
    
    return (transparency_color * material.transparency) + \
           (current_color * (1 - material.transparency)) + \
           total_reflection


def main():

    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    #create intersectable list
    intersectable = [obj for obj in objects if isinstance(obj, (Sphere, Cube, InfinitePlane))]
    #create a lights list
    lights = [obj for obj in objects if isinstance(obj, Light)]
    #create materials list
    materials = [obj for obj in objects if isinstance(obj, Material)]

    # Result array
    image_array = np.zeros((args.height, args.width, 3), dtype=np.float32)
    
    # Pre-processing
    forward_vector = (camera.look_at - camera.position).normalize()
    right_vector = forward_vector.cross(camera.up_vector).normalize()
    up_vector = right_vector.cross(forward_vector).normalize()
    
    # Main rendering loop
    # Optimization idea: Vectorize this loop later if asked, but for now simple loop
    for y in range(args.height):
        for x in range(args.width):
            # args.width - 1 - x Flips the Left/Right axis to fix mirroring
            ray = camera.get_ray(args.width - 1 - x, args.height - 1 - y, args.width, args.height, forward_vector, right_vector, up_vector)
            
            final_color = trace_ray(ray, intersectable, lights, materials, scene_settings)
            
            # Store in image
            image_array[y, x] = final_color.to_rgb()

    # Save the output image
    save_image(image_array, args.output_image)

if __name__ == '__main__':
    main()
