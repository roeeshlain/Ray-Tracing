from vector import Vector
class InfinitePlane:
    """
    InfinitePlane class - represents an infinite plane in 3D space.
    Includes a specific method for ray intersection.

    Attributes:
        normal (Vector): The normal vector of the plane.
        offset (float): An offset along the normal vector of the plane.
        material_index (int): The index of the material of the plane.
    """
    def __init__(self, normal: Vector, offset, material_index):
        self.normal = Vector(*normal).normalize()
        self.offset = offset
        self.material_index = material_index

    def ray_interception(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) > 1e-6:  # Ensure the ray is not parallel to the plane or in opposite direction
            t = (self.offset - self.normal.dot(ray.origin)) / denom
            if t >= 0:
                intersection_point = ray.origin + ray.direction * t
                return intersection_point, self.normal  
        return False  # No intersection
