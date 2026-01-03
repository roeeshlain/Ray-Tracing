from vector import Vector

class Material:
    """
    Material class - represents a material in 3D space.
    It is used to represent the properties of a material in 3D space.

    Attributes:
        diffuse_color (Vector): The diffuse color of the material.
        specular_color (Vector): The specular color of the material.
        reflection_color (Vector): The reflection color of the material.
        shininess (float): The shininess of the material.
        transparency (float): The transparency of the material.
    """
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = Vector(*diffuse_color)
        self.specular_color = Vector(*specular_color)
        self.reflection_color = Vector(*reflection_color)
        self.shininess = shininess
        self.transparency = transparency
