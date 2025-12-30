from vector import Vector
import numpy as np

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = Vector(*position)
        self.scale = scale
        self.material_index = material_index
    #to implement ray_interception(self,Ray)
    
    def ray_interception(self, ray): #using slab method
        t_min = (self.position.x - self.scale/2 - ray.origin.x) / ray.direction.x
        t_max = (self.position.x + self.scale/2 - ray.origin.x) / ray.direction.x
        if t_min > t_max:
            t_min, t_max = t_max, t_min

        ty_min = (self.position.y - self.scale/2 - ray.origin.y) / ray.direction.y
        ty_max = (self.position.y + self.scale/2 - ray.origin.y) / ray.direction.y
        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min

        if (t_min > ty_max) or (ty_min > t_max):
            return False

        if ty_min > t_min:
            t_min = ty_min
        if ty_max < t_max:
            t_max = ty_max

        tz_min = (self.position.z - self.scale/2 - ray.origin.z) / ray.direction.z
        tz_max = (self.position.z + self.scale/2 - ray.origin.z) / ray.direction.z
        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min

        if (t_min > tz_max) or (tz_min > t_max):
            return False

        if tz_min > t_min:
            t_min = tz_min
        if tz_max < t_max:
            t_max = tz_max

        # Updated logic: If ray starts inside the cube, t_min will be negative.
        # We must choose the smallest POSITIVE t to find the intersection AHEAD of the ray.
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
