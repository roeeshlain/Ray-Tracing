from vector import Vector

class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = Vector(*diffuse_color)
        self.specular_color = Vector(*specular_color)
        self.reflection_color = Vector(*reflection_color)
        self.shininess = shininess
        self.transparency = transparency
