from vector import Vector
class InfinitePlane:
    def __init__(self, normal: Vector, offset, material_index):
        self.normal = Vector(*normal).normalize()
        self.offset = offset
        self.material_index = material_index

    #to implement ray interception
    def ray_interception(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) > 1e-6:  # Ensure the ray is not parallel to the plane or in opposite direction
            t = (self.offset - self.normal.dot(ray.origin)) / denom
            if t >= 0:
                intersection_point = ray.origin + ray.direction * t
                return intersection_point, self.normal  # Added normal to the return value
        return False  # No intersection
