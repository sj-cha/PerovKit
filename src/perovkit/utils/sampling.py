from collections import defaultdict
import math
import random
from typing import Dict, List, Tuple

import numpy as np


def farthest_point_sampling(
    coords: np.ndarray,
    n_target: int,
    rng: random.Random,
    initial_idx: int | None = None,
) -> List[int]:
    n = coords.shape[0]
    if n_target >= n:
        return list(range(n))

    if initial_idx is None:
        initial_idx = rng.randint(0, n - 1)

    selected = [initial_idx]

    d = np.linalg.norm(coords - coords[initial_idx], axis=1)
    d[initial_idx] = 0.0

    while len(selected) < n_target:
        next_idx = int(np.argmax(d))
        selected.append(next_idx)

        new_d = np.linalg.norm(coords - coords[next_idx], axis=1)
        d = np.minimum(d, new_d)
        d[selected] = 0.0

    return selected


def per_face_farthest_point_sampling(
    coords: np.ndarray,
    planes: List[Tuple[int, int, int]],
    n_target: int,
    rng: random.Random,
) -> List[int]:
    """FPS run independently per face, with n_target allocated proportionally."""

    face_groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, plane in enumerate(planes):
        # Group by absolute plane (e.g., (1,0,0) and (-1,0,0) are opposite faces)
        face_groups[plane].append(i)

    total = len(coords)
    if n_target >= total:
        return list(range(total))

    selected: List[int] = []
    remainder = 0.0

    faces = sorted(face_groups.keys())
    for face in faces:
        indices = face_groups[face]
        # Proportional allocation with fractional accumulation
        exact = n_target * len(indices) / total + remainder
        n_face = int(math.floor(exact))
        remainder = exact - n_face
        if n_face == 0:
            continue

        face_coords = coords[indices]
        face_selected = farthest_point_sampling(face_coords, n_face, rng)
        selected.extend(indices[j] for j in face_selected)

    # If rounding left us short, fill from unselected sites
    if len(selected) < n_target:
        unselected = [i for i in range(total) if i not in set(selected)]
        rng.shuffle(unselected)
        selected.extend(unselected[:n_target - len(selected)])

    return selected
