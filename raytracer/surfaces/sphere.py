from vector import Vector

class Sphere:
    def __init__(self, position, radius, material_index):
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("Position must be a list or tuple of three numerical values")
        self.position = Vector(*position)  # Convert position to a Vector
        self.radius = radius
        self.material_index = material_index
        

    #to implement ray interception
    def ray_interception(self, ray):
        a = 1  # since ray.direction is normalized
        b = 2 * ray.direction.dot(ray.origin - self.position)
        c = (ray.origin - self.position).dot(ray.origin - self.position) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False  # No intersection
        else:
            t1 = (-b - discriminant**0.5) / (2 * a)
            t2 = (-b + discriminant**0.5) / (2 * a)
            
            # Logic to handle rays starting inside the sphere
            if t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                return False # Both interactions are behind the ray

            intersection_point = ray.origin + ray.direction * t
            normal = (intersection_point - self.position).normalize()  # Calculate normal
            return intersection_point, normal  # Return normal
