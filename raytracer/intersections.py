import numpy as np

from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from utils import EPSILON


def intersect_sphere(ray_origins, ray_directions, sphere):
    """
    Vectorized Ray-sphere intersection.
    ray_origins: (H, W, 3)
    ray_directions: (H, W, 3)
    Returns: t (H, W), mask (H, W)
    """
    sphere_pos = np.array(sphere.position, dtype=np.float32)
    radius = float(sphere.radius)
    
    # Vector from ray origin to sphere center
    # ray_origins: (H, W, 3), sphere_pos: (3,) -> oc: (H, W, 3)
    oc = ray_origins - sphere_pos
    
    # Quadratic coefficients
    # b = dot(oc, d)
    b = np.sum(oc * ray_directions, axis=-1)
    
    # c = dot(oc, oc) - r^2
    c = np.sum(oc * oc, axis=-1) - radius * radius
    
    # Discriminant
    discriminant = b * b - c
    
    # Initialize t with infinity
    t = np.full(discriminant.shape, np.inf)
    mask = discriminant >= 0
    
    # Calculate solutions where discriminant >= 0
    sqrt_disc = np.sqrt(np.maximum(0, discriminant))
    
    # Standard solution t1 = -b - sqrt_delta
    t1 = -b - sqrt_disc
    # Second solution t2 = -b + sqrt_delta
    t2 = -b + sqrt_disc
    
    # Logic to pick the smallest positive t > EPSILON
    # We want t such that t > EPSILON.
    # If t1 > EPSILON, pick t1.
    # Else if t2 > EPSILON, pick t2.
    # Else no valid intersection.
    
    # Create mask for valid t1
    mask_t1 = (t1 > EPSILON) & mask
    t[mask_t1] = t1[mask_t1]
    
    # Where t1 was invalid, try t2
    # We only care about pixels where t1 was NOT picked but t2 is valid
    mask_t2 = (~mask_t1) & (t2 > EPSILON) & mask
    t[mask_t2] = t2[mask_t2]
    
    # Final validity mask: pixels where we found a valid t
    final_mask = mask_t1 | mask_t2
    
    return t, final_mask


def intersect_plane(ray_origins, ray_directions, plane):
    """
    Vectorized Ray-plane intersection.
    Returns: t (H, W), mask (H, W)
    """
    normal = np.array(plane.normal, dtype=np.float32)
    offset = float(plane.offset)
    
    # denom = dot(n, d)
    denom = np.sum(normal * ray_directions, axis=-1)
    
    # Check parallel rays (abs(denom) < EPSILON)
    non_parallel_mask = np.abs(denom) > EPSILON
    
    # t = (offset - dot(n, o)) / denom
    numer = offset - np.sum(normal * ray_origins, axis=-1)
    
    # We'll compute t everywhere, but filter with mask
    # Avoid division by zero warnings by using safe division or ignoring invalid results
    # Using np.divide with where clause might be cleaner, but simple calculation + masking is fine
    # if we are careful.
    
    # To be safe, set denom to 1.0 where it is too small, result will be masked out anyway
    safe_denom = np.where(non_parallel_mask, denom, 1.0)
    t = numer / safe_denom
    
    # Valid t must be > EPSILON and not parallel
    valid_mask = non_parallel_mask & (t > EPSILON)
    
    # Reset invalid t to infinity
    t_out = np.full(ray_origins.shape[:-1], np.inf)
    t_out[valid_mask] = t[valid_mask]
    
    return t_out, valid_mask


def intersect_cube(ray_origins, ray_directions, cube):
    """
    Vectorized Ray-AABB intersection.
    Returns: t (H, W), mask (H, W)
    """
    position = np.array(cube.position, dtype=np.float32)
    scale = float(cube.scale)
    half_scale = scale / 2.0
    
    cube_min = position - half_scale
    cube_max = position + half_scale
    
    # 1 / ray_direction
    # Handle division by zero
    inv_dir = 1.0 / np.where(np.abs(ray_directions) < EPSILON, EPSILON * np.sign(ray_directions), ray_directions)
    
    t_min = (cube_min - ray_origins) * inv_dir
    t_max = (cube_max - ray_origins) * inv_dir
    
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    
    # t_near = max(t1.x, t1.y, t1.z)
    t_near = np.max(t1, axis=-1)
    # t_far = min(t2.x, t2.y, t2.z)
    t_far = np.min(t2, axis=-1)
    
    # Valid if t_near <= t_far and t_far > EPSILON and t_near > EPSILON (actually t > EPSILON)
    # The logic in original code:
    # if t_near > t_far or t_far < EPSILON: return None
    # t = t_near if t_near > EPSILON else t_far
    
    valid_slab = (t_near <= t_far) & (t_far > EPSILON)
    
    # Determine t
    # If t_near > EPSILON, use t_near. Else use t_far.
    use_near = t_near > EPSILON
    t = np.where(use_near, t_near, t_far)
    
    final_mask = valid_slab & (t > EPSILON)
    
    t_out = np.full(ray_origins.shape[:-1], np.inf)
    t_out[final_mask] = t[final_mask]
    
    return t_out, final_mask


def find_closest_intersection(ray_origins, ray_directions, geometry):
    """
    Find nearest intersection for a batch of rays.
    Returns:
        min_t: (H, W)
        hit_mask: (H, W) boolean
        hit_normals: (H, W, 3)
        object_indices: (H, W) integer index into geometry list
    """
    shape = ray_origins.shape[:-1]
    min_t = np.full(shape, np.inf)
    hit_mask = np.zeros(shape, dtype=bool)
    hit_normals = np.zeros(ray_origins.shape, dtype=np.float32)
    object_indices = np.full(shape, -1, dtype=int)
    
    for i, obj in enumerate(geometry):
        t = None
        mask = None
        
        if isinstance(obj, Sphere):
            t, mask = intersect_sphere(ray_origins, ray_directions, obj)
            
            # Compute Normal specific to Sphere if hit
            # We delay normal computation until checking if it's the closest hit to save ops,
            # BUT efficient vectorization often means computing everything and masking.
            # However, we can compute normals ONLY for pixels that are the new closest.
            
        elif isinstance(obj, InfinitePlane):
            t, mask = intersect_plane(ray_origins, ray_directions, obj)
            
        elif isinstance(obj, Cube):
            t, mask = intersect_cube(ray_origins, ray_directions, obj)
            
        else:
            continue
            
        # Update closest intersection
        # We need a mask for where the new t is closer than existing min_t
        if t is not None:
            closer_mask = mask & (t < min_t)
            
            # Update t
            min_t = np.where(closer_mask, t, min_t)
            
            # Update index
            object_indices = np.where(closer_mask, i, object_indices)
            
            # Update hit mask (global)
            hit_mask = hit_mask | closer_mask
            
            # Compute/Update normals only for closer hits
            # This is slightly different per object type
            if np.any(closer_mask):
                points = ray_origins[closer_mask] + ray_directions[closer_mask] * t[closer_mask][:, np.newaxis]
                
                if isinstance(obj, Sphere):
                    # Normal = (p - center) / radius
                    sphere_pos = np.array(obj.position, dtype=np.float32)
                    normals = (points - sphere_pos) / obj.radius
                    hit_normals[closer_mask] = normals
                    
                elif isinstance(obj, InfinitePlane):
                    # Normal is constant (or flipped if hitting back)
                    # We need to check alignment with ray direction for current pixels
                    plane_n = np.array(obj.normal, dtype=np.float32)
                    # denom was dot(n, d). if denom > 0 -> flip
                    # We recompute dot(n, d) just for these pixels? Or broadcast.
                    # Let's broadcast plane_n
                    # If dot(n, d) > 0, normal = -n, else n
                    # d is ray_directions[closer_mask]
                    d_subset = ray_directions[closer_mask]
                    denom = np.sum(d_subset * plane_n, axis=-1)
                    
                    # Create array of normals
                    n_subset = np.tile(plane_n, (len(denom), 1))
                    n_subset[denom > 0] = -plane_n
                    
                    hit_normals[closer_mask] = n_subset
                    
                elif isinstance(obj, Cube):
                    # Cube normal depends on which face was hit
                    # M = point - center
                    # Normal is +1/-1 on the axis where abs(coord) is max (closest to face)
                    # OR we can use the "step" logic from slab method.
                    # Simpler is to check which component of |p - center| is approx scale/2
                    
                    center = np.array(obj.position, dtype=np.float32)
                    half_scale = obj.scale / 2.0
                    local_p = points - center
                    
                    # We find argmax of abs(local_p).
                    # Since p is on surface, one component should be approx +/- half_scale
                    
                    # BUT wait, simple epsilon check is robust
                    abs_p = np.abs(local_p)
                    # We want to know if it's x, y, or z face
                    # Distances to faces:
                    dist_to_edge = np.abs(abs_p - half_scale)
                    axis = np.argmin(dist_to_edge, axis=-1)
                    
                    # Construct normals
                    n_subset = np.zeros_like(local_p)
                    # Set +/- 1.0 on the correct axis
                    # numpy advanced indexing
                    rows = np.arange(len(axis))
                    # Sign depends on local_p[axis]
                    signs = np.sign(local_p[rows, axis])
                    
                    n_subset[rows, axis] = signs
                    hit_normals[closer_mask] = n_subset

    return min_t, hit_mask, hit_normals, object_indices
