from vector import Vector
import numpy as np

class Cube:
    """
    Cube class - represents a cube in 3D space.
    Includes a specific method for ray intersection.

    Attributes:
        position (Vector): The position of the center of the cube.
        scale (float): The edge length(scale) of the cube.
        material_index (int): The index of the material of the cube.
    """
    def __init__(self, position, scale, material_index):
        self.position = Vector(*position)
        self.scale = scale
        self.material_index = material_index
    
    def ray_interception(self, ray): # Using slab method
        epsilon = 1e-6

        # X Axis
        min_x = self.position.x - self.scale/2
        max_x = self.position.x + self.scale/2
        
        if abs(ray.direction.x) < epsilon:
            if ray.origin.x < min_x or ray.origin.x > max_x:
                return False
            t_min = -float('inf')
            t_max = float('inf')
        else:
            t_min = (min_x - ray.origin.x) / ray.direction.x
            t_max = (max_x - ray.origin.x) / ray.direction.x
            if t_min > t_max:
                t_min, t_max = t_max, t_min

        # Y Axis
        min_y = self.position.y - self.scale/2
        max_y = self.position.y + self.scale/2
        
        if abs(ray.direction.y) < epsilon:
            if ray.origin.y < min_y or ray.origin.y > max_y:
                return False
            ty_min = -float('inf')
            ty_max = float('inf')
        else:
            ty_min = (min_y - ray.origin.y) / ray.direction.y
            ty_max = (max_y - ray.origin.y) / ray.direction.y
            if ty_min > ty_max:
                ty_min, ty_max = ty_max, ty_min

        if (t_min > ty_max) or (ty_min > t_max):
            return False

        if ty_min > t_min:
            t_min = ty_min
        if ty_max < t_max:
            t_max = ty_max

        # Z Axis
        min_z = self.position.z - self.scale/2
        max_z = self.position.z + self.scale/2
        
        if abs(ray.direction.z) < epsilon:
            if ray.origin.z < min_z or ray.origin.z > max_z:
                return False
            tz_min = -float('inf')
            tz_max = float('inf')
        else:
            tz_min = (min_z - ray.origin.z) / ray.direction.z
            tz_max = (max_z - ray.origin.z) / ray.direction.z
            if tz_min > tz_max:
                tz_min, tz_max = tz_max, tz_min

        if (t_min > tz_max) or (tz_min > t_max):
            return False

        if tz_min > t_min:
            t_min = tz_min
        if tz_max < t_max:
            t_max = tz_max

        # If ray starts inside the cube, t_min will be negative.
        # We must choose the smallest positive t to find the intersection ahead of the ray.
        t = t_min
        if t < 0:
            t = t_max
        if t < 0:
            return False

        intersection_point = ray.origin + ray.direction * t
        
        normal = Vector(0, 0, 0)  # Placeholder for normal
        # Determine the normal based on the closest face
        if abs(intersection_point.x - self.position.x) > abs(intersection_point.y - self.position.y) and abs(intersection_point.x - self.position.x) > abs(intersection_point.z - self.position.z):
            normal = Vector(np.sign(intersection_point.x - self.position.x), 0, 0)
        elif abs(intersection_point.y - self.position.y) > abs(intersection_point.z - self.position.z):
            normal = Vector(0, np.sign(intersection_point.y - self.position.y), 0)
        else:
            normal = Vector(0, 0, np.sign(intersection_point.z - self.position.z))
        return intersection_point, normal  # Return normal
