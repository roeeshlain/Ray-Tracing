from math import sqrt

class Vector: 
    """
    Vector class - represents a 3D vector.
    It is used to represent points in 3D space, as well as direction vectors.
    Includes basic vector calculations.

    Attributes:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
        z (float): The z-coordinate of the vector.
    """
    __slots__ = ['x', 'y', 'z']

    def __init__(self, x=0.0, y=0.0, z=0.0): #initialize vector with given coords
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self): # Return tuple of coords
        return (self.x, self.y, self.z)
    

    def __add__(self, other): # Add each coord with given scalar or vector
        if not isinstance(other, Vector):
            raise TypeError("Operand must be an instance of Vector")    
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other): # Sub each coord with given scalar or vector
        if not isinstance(other, Vector):
            raise TypeError("Operand must be an instance of Vector")    
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other): # Multiple each coord with given scalar or vector
        if isinstance(other, (int, float)): #if other is scalar
            return Vector(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)
        raise TypeError("Operand must be an instance of Vector or a scalar")
        
    def __truediv__(self, other): # Divide each coord with given scalar or vector
        if isinstance(other, (int, float)):
            return Vector(self.x / other, self.y / other, self.z / other)
        elif isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y, self.z / other.z)
        raise TypeError("Operand must be an instance of Vector or a scalar")
    
    def __pow__(self, other): # Calculate each coordinate in self to be itself to the power of the given scalar or vector
        if isinstance(other, (int, float)):
            return Vector(self.x**other, self.y**other, self.z**other)
        elif isinstance(other, Vector):
            return Vector(self.x**other.x, self.y**other.y, self.z**other.z)
        raise TypeError("Operand must be an instance of Vector or a scalar")
    
    def __gt__(self, other): # Compare L1 normal of vectors, or to L1 normal to a scalar
        if isinstance(other, (int, float)):
            return self.magnitude() > other
        elif isinstance(other, Vector):
            return self.magnitude() > other.magnitude()
        raise TypeError("error: You are comparing a vector with an unsupported variable type!")

    
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.magnitude() < other
        if isinstance(other, Vector):
            return self.magnitude() < other.magnitude()
        raise TypeError("error: You are comparing a vector with an unsupported variable type!")
    
    def __ge__(self, other): # Compare L1 normal of vectors, or to L1 normal to a scalar, same,  greater than or equal
        if isinstance(other, (int, float)):
            return self.magnitude() >= other
        if isinstance(other, Vector):
            return self.magnitude() >= other.magnitude()
        raise TypeError("error: You are comparing a vector with an unsupported variable type!")
    
    def __le__(self, other): # Compare L1 normal of vectors, or to L1 normal to a scalar, same,  less than or equal
        if isinstance(other, (int, float)):
            return self.magnitude() <= other
        if isinstance(other, Vector):
            return self.magnitude() <= other.magnitude()
        raise TypeError("error: You are comparing a vector with an unsupported variable type!")
    
    def __eq__(self, other):  # Check equality of L1 normal of vectors, or L1 normal to a scalar
        if isinstance(other, (int, float)):
            return self.magnitude() == other
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False  # Return False for unsupported types
    
    def __neg__(self): # Return negative vector has -1 in all cords
        return Vector(-self.x, -self.y, -self.z)
    
    def __pos__(self): # Return positive vector has 1 in all cords
        return Vector(+self.x, +self.y, +self.z)
    
    def __float__(self): # Return float L1 normal of vector
        return float(self.magnitude())
    
    def __int__(self) -> int: # Return int L1 normal of vector
        return int(self.magnitude())
    
    def dot(self, other): # Return dot product of two vectors, or vector and scalar
        if isinstance(other, (int, float)):
            return self.x * other + self.y * other + self.z * other
        elif isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        raise TypeError("Operand must be an instance of Vector or a scalar")        
    
    def cross(self, other): # Return cross product of two vectors
        if isinstance(other, Vector):
            return Vector((self.y * other.z) - (self.z * other.y), (self.z * other.x) - (self.x * other.z), (self.x * other.y) - (self.y * other.x))
        raise TypeError("Operand must be an instance of Vector")    

    def magnitude(self): # Return the L1 normal value of vector
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self): # Normalize vector
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return self / mag

    
    def reflect(self, other): # Other is a normal vector, return reflection of other with acc to normal
        if isinstance(other, Vector):
            return self - other * (self.dot(other)) * 2
        raise TypeError("Operand must be an instance of Vector")    
    
    def to_rgb(self): # Return tuple of rgb values, given a vector of floats(0-1)
        # This function is used to convert a vector of floats(0-1) to a tuple of rgb values(0-255)
        # It is used to clamp the values of the vector to the range of 0-255
        r = max(0, min(1, self.x))
        g = max(0, min(1, self.y))
        b = max(0, min(1, self.z))
        return (r * 255, g * 255, b * 255)
