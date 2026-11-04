from collections import defaultdict
import math
import random
from typing import Dict, List, Tuple

import numpy as np


def farthest_point_sampling(coords: np.ndarray, n_target: int, rng: random.Random) -> List[int]:
    """
    `n_target` of the points, starting from a random one, each next the farthest from those
    taken.
    """
    n = coords.shape[0]
    if n_target <= 0:
        return []
    if n_target >= n:
        return list(range(n))

    selected = [rng.randint(0, n - 1)]
    d = np.linalg.norm(coords - coords[selected[0]], axis=1)
    while len(selected) < n_target:
        next_idx = int(np.argmax(d))
        selected.append(next_idx)
        d = np.minimum(d, np.linalg.norm(coords - coords[next_idx], axis=1))
    return selected


def per_face_farthest_point_sampling(
    coords: np.ndarray,
    planes: List[Tuple[int, int, int]],
    n_target: int,
    rng: random.Random,
) -> List[int]:
    """Farthest-point sampling on every face apart, with n_target shared out in proportion."""
    # one group per face: (1, 0, 0) and (-1, 0, 0) are opposite faces
    face_groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, plane in enumerate(planes):
        face_groups[plane].append(i)

    total = len(coords)
    if n_target >= total:
        return list(range(total))

    selected: List[int] = []
    remainder = 0.0
    for face in sorted(face_groups):
        indices = face_groups[face]
        # proportional share, the fractions carried on to the next face
        exact = n_target * len(indices) / total + remainder
        n_face = int(math.floor(exact))
        remainder = exact - n_face
        if n_face == 0:
            continue
        selected.extend(indices[j] for j in farthest_point_sampling(coords[indices], n_face, rng))

    # what rounding left short comes from the sites not taken
    if len(selected) < n_target:
        taken = set(selected)
        unselected = [i for i in range(total) if i not in taken]
        rng.shuffle(unselected)
        selected.extend(unselected[:n_target - len(selected)])
    return selected
