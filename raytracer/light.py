from vector import Vector

class Light:
    """
    Light class - represents a light source in 3D space.
    It is used to represent the properties of a light source in 3D space.

    Attributes:
        position (Vector): The position of the light source.
        color (Vector): The color of the light source.
        specular_intensity (float): The specular intensity of the light source.
        shadow_intensity (float): The shadow intensity of the light source.
        radius (float): The radius of the light source.
    """
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("Position must be a list or tuple of three numerical values")
        self.position = Vector(*position)
        self.color = Vector(*color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
