# pync/utils/geometry.py
import numpy as np
import pandas as pd
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

def compute_b_x_b_angles(structure,
                         atoms=None, 
                         B_indices=None, 
                         layer=None):
    
    octahedra = structure.octahedra
    B_ijk = structure.B_ijk
    if atoms is None:
        atoms = structure.atoms

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    
    # Determine which B atoms to consider
    if B_indices is not None:
        b_set = set(B_indices)
    elif layer is not None:
        assert int(layer) <= max(structure.core.supercell)
        if isinstance(layer, int):
            layer = [layer]
        b_set = {b for b, ijk in B_ijk.items() if ijk[2] in layer}
    else:
        b_set = set(octahedra.keys())

    # Build reverse map: X_index -> list of B indices that contain it
    x_to_b = {}
    for b in b_set:
        if b not in octahedra:
            continue
        for x_idx in octahedra[b]['X']:
            x_to_b.setdefault(x_idx, []).append(b)

    # Compute angles for each shared X atom
    results = []
    seen = set()
    for x_idx, b_list in x_to_b.items():
        if len(b_list) < 2:
            continue
        for i in range(len(b_list)):
            for j in range(i + 1, len(b_list)):
                b1, b2 = b_list[i], b_list[j]
                key = (min(b1, b2), max(b1, b2), x_idx)
                if key in seen:
                    continue
                seen.add(key)

                pos_x = positions[x_idx]
                vec1 = positions[b1] - pos_x
                vec2 = positions[b2] - pos_x

                from ase.geometry import find_mic
                vecs, _ = find_mic(np.array([vec1, vec2]), cell, pbc=pbc)
                vec1, vec2 = vecs[0], vecs[1]

                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))

                results.append({
                    'B1': b1, 'B2': b2, 'X': x_idx,
                    'B1_ijk': B_ijk.get(b1), 'B2_ijk': B_ijk.get(b2),
                    'angle': angle
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['B1', 'B2', 'X']).reset_index(drop=True)
    return df