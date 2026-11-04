from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from ..utils.rotation import rotation_about_axis


def build_ordered_rotmat(angles: np.ndarray, order: str = "xyz") -> np.ndarray:
    """Turns by `angles` (radians) about x, y and z, in `order` (first letter first)."""
    basis = np.eye(3, dtype=float)
    mats = {c: rotation_about_axis(basis[:, k], float(angles[k])) for k, c in enumerate("xyz")}
    R = np.eye(3)
    for c in order:
        R = mats[c] @ R
    return R


def build_octahedra_rotmat(
    glazer: str,
    angles: Tuple[float, float, float],
    b_ijk: Dict[int, Tuple[int, int, int]],
    b_keys: Iterable[int],
    order: str = "xyz",
) -> Dict[int, np.ndarray]:
    """
    Rotation of the octahedron of every B in `b_keys` for the Glazer tilt `glazer` (e.g.
    "a-b+c0"), by `angles` (degrees) about x, y and z: none for 0; for -, the sign flips
    from one octahedron to the next along every axis, for +, along the other two only.
    """
    glazer = glazer.strip()
    marks = [c for c in glazer if c in ("0", "+", "-")]
    if len(marks) != 3:
        raise ValueError(f"Could not parse Glazer '{glazer}'. Expected exactly 3 of 0/+/-.")
    ang_rad = np.deg2rad(np.array(angles, dtype=float))

    R_b: Dict[int, np.ndarray] = {}
    for b in b_keys:
        b = int(b)
        ijk = b_ijk[b]
        rot_angle = np.zeros(3, dtype=float)
        for axis, mark in enumerate(marks):
            if mark == "0" or abs(ang_rad[axis]) < 1e-16:
                continue
            flips = [int(mark == "-" or a != axis) for a in range(3)]
            n = int(ijk[0] * flips[0] + ijk[1] * flips[1] + ijk[2] * flips[2])
            rot_angle[axis] = (-1.0 if n % 2 else 1.0) * ang_rad[axis]
        R_b[b] = build_ordered_rotmat(rot_angle, order=order)
    return R_b
