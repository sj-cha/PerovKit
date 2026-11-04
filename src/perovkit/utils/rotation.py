import numpy as np

def rotation_about_axis(axis: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    """
    Rodrigues rotation by `theta` radians about `axis`, of any length: (3, 3) for one
    angle, (n, 3, 3) for an array of them.
    """
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    s = np.sin(np.asarray(theta))[..., None, None]
    c = np.cos(np.asarray(theta))[..., None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)

def rotation_from_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotation turning the direction of `u` onto that of `v`.
    """
    u = np.asarray(u, dtype=float) / np.linalg.norm(u)
    v = np.asarray(v, dtype=float) / np.linalg.norm(v)
    c = float(np.dot(u, v))

    # already aligned
    if np.isclose(c, 1.0):
        return np.eye(3)

    # opposite: half a turn about any axis perpendicular to u
    if np.isclose(c, -1.0):
        tmp = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(u), np.abs(tmp)):
            tmp = np.array([0.0, 1.0, 0.0])
        return rotation_about_axis(np.cross(u, tmp), np.pi)

    return rotation_about_axis(np.cross(u, v), np.arccos(c))
