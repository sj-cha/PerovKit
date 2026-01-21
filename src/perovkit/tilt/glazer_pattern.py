from __future__ import annotations

from typing import Dict, Tuple, Iterable

import numpy as np

from ..utils.rotation import rotation_about_axis


def parse_glazer(glazer: str) -> Tuple[str, str, str]:
    glazer = glazer.strip()
    marks = [c for c in glazer if c in ("0", "+", "-")]
    if len(marks) != 3:
        raise ValueError(f"Could not parse Glazer '{glazer}'. Expected exactly 3 of 0/+/-.")
    return marks[0], marks[1], marks[2]


def glazer_kvec(pattern: str, axis: int) -> np.ndarray:
    if pattern == "0":
        return np.array([0, 0, 0], dtype=int)
    if pattern == "-":
        return np.array([1, 1, 1], dtype=int)
    if pattern == "+":
        kv = np.array([1, 1, 1], dtype=int)
        kv[axis] = 0
        return kv


def phase_factor_ijk(ijk: Tuple[int, int, int], kvec: np.ndarray) -> float:
    n = int(ijk[0] * kvec[0] + ijk[1] * kvec[1] + ijk[2] * kvec[2])
    return -1.0 if (n % 2) else 1.0


def build_ordered_rotmat(angles: np.ndarray, order: str = "xyz") -> np.ndarray:
    cubic_basis = np.eye(3, dtype=float)
    ax, ay, az = cubic_basis[:, 0], cubic_basis[:, 1], cubic_basis[:, 2]
    Rx = rotation_about_axis(ax, float(angles[0]))
    Ry = rotation_about_axis(ay, float(angles[1]))
    Rz = rotation_about_axis(az, float(angles[2]))
    mats = {"x": Rx, "y": Ry, "z": Rz}

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
    
    pat_x, pat_y, pat_z = parse_glazer(glazer)
    patterns = (pat_x, pat_y, pat_z)
    ang_rad = np.deg2rad(np.array(angles, dtype=float))

    R_b: Dict[int, np.ndarray] = {}
    for b in b_keys:
        b = int(b)
        ijk = b_ijk[b]
        rot_angle = np.zeros(3, dtype=float)

        for axis, pat in enumerate(patterns):
            if pat == "0" or abs(ang_rad[axis]) < 1e-16:
                rot_angle[axis] = 0.0
                continue
            kv = glazer_kvec(pat, axis)
            s = phase_factor_ijk(ijk, kv)
            rot_angle[axis] = s * ang_rad[axis]

        R_b[b] = build_ordered_rotmat(rot_angle, order=order)

    return R_b