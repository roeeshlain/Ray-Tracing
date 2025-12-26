import numpy as np

from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from utils import EPSILON, normalize


def intersect_sphere(ray_origin, ray_direction, sphere):
    """Ray-sphere intersection using quadratic formula."""
    sphere_pos = np.array(sphere.position, dtype=float)
    oc = ray_origin - sphere_pos
    
    # Simplified: a=1 for normalized direction
    b = np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - c
    
    if discriminant < 0:
        return None
    
    sqrt_disc = np.sqrt(discriminant)
    t = -b - sqrt_disc  # Nearest intersection
    
    if t < EPSILON:
        t = -b + sqrt_disc  # Try far intersection
        if t < EPSILON:
            return None
    
    point = ray_origin + t * ray_direction
    normal = (point - sphere_pos) / sphere.radius  # Already normalized
    return {"distance": t, "point": point, "normal": normal, "object": sphere}


def intersect_plane(ray_origin, ray_direction, plane):
    """Ray-plane intersection."""
    normal = np.array(plane.normal, dtype=float)
    denom = np.dot(normal, ray_direction)
    
    if abs(denom) < EPSILON:
        return None
    
    t = (plane.offset - np.dot(normal, ray_origin)) / denom
    if t < EPSILON:
        return None
    
    point = ray_origin + t * ray_direction
    # Face normal toward ray
    face_normal = -normal if denom > 0 else normal
    
    return {"distance": t, "point": point, "normal": face_normal, "object": plane}


def intersect_cube(ray_origin, ray_direction, cube):
    """Ray-AABB intersection using slab method."""
    half = cube.scale / 2.0
    cube_pos = np.array(cube.position, dtype=float)
    cube_min = cube_pos - half
    cube_max = cube_pos + half
    
    # Safe division
    inv_dir = np.where(np.abs(ray_direction) < EPSILON, 1.0 / EPSILON, 1.0 / ray_direction)
    t_min = (cube_min - ray_origin) * inv_dir
    t_max = (cube_max - ray_origin) * inv_dir
    
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    t_near = np.max(t1)
    t_far = np.min(t2)
    
    if t_near > t_far or t_far < EPSILON:
        return None
    
    t = t_near if t_near > EPSILON else t_far
    point = ray_origin + t * ray_direction
    
    # Find which face was hit
    dist = np.abs(point - cube_min)
    dist2 = np.abs(point - cube_max)
    combined = np.minimum(dist, dist2)
    axis = np.argmin(combined)
    
    normal = np.zeros(3)
    normal[axis] = 1.0 if point[axis] > cube_pos[axis] else -1.0
    
    return {"distance": t, "point": point, "normal": normal, "object": cube}


def find_closest_intersection(ray_origin, ray_direction, geometry):
    """Find nearest intersection along ray."""
    closest = None
    min_dist = float("inf")
    
    for obj in geometry:
        if isinstance(obj, Sphere):
            hit = intersect_sphere(ray_origin, ray_direction, obj)
        elif isinstance(obj, InfinitePlane):
            hit = intersect_plane(ray_origin, ray_direction, obj)
        elif isinstance(obj, Cube):
            hit = intersect_cube(ray_origin, ray_direction, obj)
        else:
            continue
        
        if hit and hit["distance"] < min_dist:
            min_dist = hit["distance"]
            closest = hit
    
    return closest
