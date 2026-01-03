from vector import Vector
from material import Material
from surfaces.sphere import Sphere
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane

class Ray:
    """
    Ray class - represents a ray in 3D space. 

    Attributes:
        origin (Vector): The origin of the ray.
        direction (Vector): The direction of the ray.
    """
    __slots__ = ('origin', 'direction')

    def __init__(self, origin: Vector, direction: Vector):
        self.origin = origin
        self.direction = direction.normalize() #make direction normalized(length 1)

    def __repr__(self):
        return (self.origin, self.direction)
    


