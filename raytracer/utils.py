import numpy as np

EPSILON = 1e-4


def normalize(vector):
    """Return a unit-length copy of the vector; zero vector stays zero."""
    vec = np.array(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def reflect(direction, normal):
    """Reflect an incoming direction around a surface normal."""
    d = np.array(direction, dtype=float)
    n = np.array(normal, dtype=float)
    return d - 2.0 * np.dot(d, n) * n


def clamp_color(color):
    """Clamp color components to [0, 1]."""
    return np.clip(color, 0.0, 1.0)
