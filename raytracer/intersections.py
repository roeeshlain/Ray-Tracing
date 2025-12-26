import numpy as np

from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from utils import EPSILON, normalize


def intersect_sphere(ray_origin, ray_direction, sphere):
    oc = np.array(ray_origin) - np.array(sphere.position)
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere.radius ** 2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    t = None
    if t1 > EPSILON:
        t = t1
    elif t2 > EPSILON:
        t = t2
    if t is None:
        return None

    point = ray_origin + t * ray_direction
    normal = normalize(point - np.array(sphere.position))
    return {"distance": t, "point": point, "normal": normal, "object": sphere}


def intersect_plane(ray_origin, ray_direction, plane):
    normal = np.array(plane.normal, dtype=float)
    denom = np.dot(normal, ray_direction)
    if abs(denom) < EPSILON:
        return None

    # Plane definition expected by the scene files: n·p = offset
    t = (plane.offset - np.dot(normal, ray_origin)) / denom
    if t < EPSILON:
        return None

    point = ray_origin + t * ray_direction
    # Flip normal to oppose incoming ray for consistent lighting
    if np.dot(normal, ray_direction) > 0:
        normal = -normal

    return {"distance": t, "point": point, "normal": normalize(normal), "object": plane}


def intersect_cube(ray_origin, ray_direction, cube):
    half = cube.scale / 2.0
    cube_min = np.array(cube.position) - half
    cube_max = np.array(cube.position) + half

    dir_safe = np.where(np.abs(ray_direction) < EPSILON, EPSILON, ray_direction)
    t_min = (cube_min - ray_origin) / dir_safe
    t_max = (cube_max - ray_origin) / dir_safe

    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    t_near = np.max(t1)
    t_far = np.min(t2)

    if t_near > t_far or t_far < EPSILON:
        return None

    t = t_near if t_near > EPSILON else t_far
    if t < EPSILON:
        return None

    point = ray_origin + t * ray_direction

    normal = np.zeros(3)
    if abs(point[0] - cube_min[0]) < EPSILON:
        normal = np.array([-1, 0, 0])
    elif abs(point[0] - cube_max[0]) < EPSILON:
        normal = np.array([1, 0, 0])
    elif abs(point[1] - cube_min[1]) < EPSILON:
        normal = np.array([0, -1, 0])
    elif abs(point[1] - cube_max[1]) < EPSILON:
        normal = np.array([0, 1, 0])
    elif abs(point[2] - cube_min[2]) < EPSILON:
        normal = np.array([0, 0, -1])
    elif abs(point[2] - cube_max[2]) < EPSILON:
        normal = np.array([0, 0, 1])

    return {"distance": t, "point": point, "normal": normalize(normal), "object": cube}


def find_closest_intersection(ray_origin, ray_direction, geometry):
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
            hit = None

        if hit and hit["distance"] < min_dist:
            min_dist = hit["distance"]
            closest = hit

    return closest
