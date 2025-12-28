import numpy as np

EPSILON = 1e-4


def normalize(vector):
    """
    Return a unit-length copy of the vector(s); zero vector stays zero.
    Handles both single vectors (3,) and arrays (..., 3).
    """
    vec = np.array(vector, dtype=np.float32)
    # If 1D, norm is scalar. If ND, norm along last axis.
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    else:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        # Avoid division by zero
        return np.divide(vec, norm, out=np.zeros_like(vec), where=norm!=0)


def reflect(direction, normal):
    """
    Reflect an incoming direction around a surface normal.
    Handles arrays.
    """
    d = np.array(direction, dtype=np.float32)
    n = np.array(normal, dtype=np.float32)
    
    # Dot product
    if d.ndim == 1:
        dot = np.dot(d, n)
    else:
        dot = np.sum(d * n, axis=-1, keepdims=True)
        
    return d - 2.0 * dot * n


def clamp_color(color):
    """Clamp color components to [0, 1]."""
    return np.clip(color, 0.0, 1.0)
