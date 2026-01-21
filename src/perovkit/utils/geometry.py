# pync/utils/geometry.py
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial import cKDTree
import random

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

def compute_bounding_spheres(
        coords_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_lig = len(coords_list)
        centers = np.zeros((n_lig, 3), dtype=float)
        radii = np.zeros(n_lig, dtype=float)

        for i, coords in enumerate(coords_list):
            c = coords.mean(axis=0)
            centers[i] = c
            radii[i] = np.linalg.norm(coords - c, axis=1).max()

        return centers, radii

def build_neighbor_map(
    centers: np.ndarray,
    radii: np.ndarray,
    cutoff: float,
) -> Dict[int, List[int]]:
    
    n = len(centers)
    neighbor_map: Dict[int, set[int]] = {i: set() for i in range(n)}

    if n == 0:
        return {}

    max_r = float(radii.max())
    max_pair_radius = 2.0 * max_r + cutoff

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=max_pair_radius)

    for i, j in pairs:
        center_dist = np.linalg.norm(centers[j] - centers[i])
        if center_dist <= radii[i] + radii[j] + cutoff:
            neighbor_map[i].add(j)
            neighbor_map[j].add(i)

    return {i: sorted(neighbor_map[i]) for i in range(n)}
