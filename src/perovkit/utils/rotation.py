import numpy as np

def rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues' rotation formula for rotation about arbitrary axis.

    axis : (3,) unit vector
    theta : rotation angle in radians
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    x, y, z = axis
    K = np.array([[0, -z, y],
                [z, 0, -x],
                [-y, x, 0]], dtype=float)

    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def rotation_from_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    u: source vector
    v: target vector
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    c = float(np.dot(u, v))

    # already aligned
    if np.isclose(c, 1.0):
        return np.eye(3)
    
    if np.isclose(c, -1.0):
        tmp = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(u), np.abs(tmp)):
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, tmp)
        axis /= np.linalg.norm(axis)
        theta = np.pi
        return rotation_about_axis(axis, theta)

    # general case
    axis = np.cross(u, v)
    axis /= np.linalg.norm(axis)
    theta = np.arccos(c)

    return rotation_about_axis(axis, theta)