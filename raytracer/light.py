from vector import Vector

class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("Position must be a list or tuple of three numerical values")
        self.position = Vector(*position)
        self.color = Vector(*color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
