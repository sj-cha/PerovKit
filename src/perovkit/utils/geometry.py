# pync/utils/geometry.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial import cKDTree, ConvexHull
from ase.geometry import find_mic
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


def per_face_farthest_point_sampling(
    coords: np.ndarray,
    planes: List[Tuple[int, int, int]],
    n_target: int,
    rng: random.Random,
) -> List[int]:
    """FPS run independently per face, with n_target allocated proportionally."""
    from collections import defaultdict
    import math

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


def compute_B_X_B_angles(structure,
                         B_sites:List=None
                         ):
    
    octahedra = structure.octahedra
    B_ijk = structure.B_ijk
    atoms = structure.atoms

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    
    # Determine which B atoms to consider
    if B_sites is not None:
        b_set = set(B_sites)
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
    for x_idx, b_list in x_to_b.items():
        if len(b_list) < 2:
            continue
        for i in range(len(b_list)):
            for j in range(i + 1, len(b_list)):
                b1, b2 = b_list[i], b_list[j]

                pos_x = positions[x_idx]
                vec1 = positions[b1] - pos_x
                vec2 = positions[b2] - pos_x

                vecs, _ = find_mic(np.array([vec1, vec2]), cell, pbc=pbc)
                vec1, vec2 = vecs[0], vecs[1]

                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))

                results.append({
                    'B1': b1, 'B2': b2, 'X': x_idx,
                    'B1_ijk': B_ijk.get(b1), 'B2_ijk': B_ijk.get(b2),
                    'angles': angle
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['B1', 'B2', 'X']).reset_index(drop=True)
    return df


def compute_X_B_X_angles(structure,
                         B_sites:List=None):
    
    octahedra = structure.octahedra
    B_ijk = structure.B_ijk
    atoms = structure.atoms

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    
    # Determine which B atoms to consider
    if B_sites is not None:
        b_set = set(B_sites)
    else:
        b_set = set(octahedra.keys())

    # Compute angles for each shared B atom
    results = []
    for b_idx, entry in octahedra.items():
        if b_idx not in b_set:
            continue
        x_list = entry['X']
        for i in range(len(x_list)):
            for j in range(i + 1, len(x_list)):
                x1, x2 = x_list[i], x_list[j]

                pos_b = positions[b_idx]
                vec1 = positions[x1] - pos_b
                vec2 = positions[x2] - pos_b

                vecs, _ = find_mic(np.array([vec1, vec2]), cell, pbc=pbc)
                vec1, vec2 = vecs[0], vecs[1]

                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))

                results.append({
                    'B': b_idx, 'X1': x1, 'X2': x2,
                    'B_ijk': B_ijk.get(b_idx),
                    'angles': angle
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['B', 'X1', 'X2']).reset_index(drop=True)
    return df


def compute_bond_angle_variance(structure,
                                B_sites:List=None,
                                linear_threshold:float=130.0):

    df = compute_X_B_X_angles(structure, B_sites)
    df = df[df['angles'] < linear_threshold].copy()
    df['angles'] = (df['angles'].astype(float) - 90) ** 2
    variance_df = (
        df.groupby('B')['angles']
          .agg(lambda x: x.sum() / (len(x) - 1))
          .reset_index()
          .rename(columns={'angles': 'bond_angle_variance'})
    )
    variance_df['B_ijk'] = variance_df['B'].map(structure.B_ijk)
    variance_df = variance_df[['B', 'B_ijk', 'bond_angle_variance']]
    return variance_df


def compute_distortion_index(structure,
                             B_sites:List=None):

    octahedra = structure.octahedra
    B_ijk = structure.B_ijk
    atoms = structure.atoms

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    # Determine which B atoms to consider
    if B_sites is not None:
        b_set = set(B_sites)
    else:
        b_set = set(octahedra.keys())
 
    results = []
    for b_idx, entry in octahedra.items():
        if b_idx not in b_set:
            continue
        x_list = entry['X']

        pos_b = positions[b_idx]
        vecs = np.array([positions[x_idx] - pos_b for x_idx in x_list])
        vecs, _ = find_mic(vecs, cell, pbc=pbc)
        dists = np.linalg.norm(vecs, axis=1)

        D = np.mean(np.abs(dists - dists.mean()) / dists.mean())

        results.append({
            'B': b_idx,
            'B_ijk': B_ijk.get(b_idx),
            'distortion_index': D
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('B').reset_index(drop=True)
    return df


def compute_quadratic_elongation(structure,
                                 B_sites:List=None):

    octahedra = structure.octahedra
    B_ijk = structure.B_ijk
    atoms = structure.atoms

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    # Determine which B atoms to consider
    if B_sites is not None:
        b_set = set(B_sites)
    else:
        b_set = set(octahedra.keys())

    results = []
    for b_idx, entry in octahedra.items():
        if b_idx not in b_set:
            continue
        x_list = entry['X']
        if len(x_list) != 6:
            continue

        pos_b = positions[b_idx]
        vecs = np.array([positions[x_idx] - pos_b for x_idx in x_list])
        vecs, _ = find_mic(vecs, cell, pbc=pbc)
        dists = np.linalg.norm(vecs, axis=1)

        # Robinson l_0: center-to-vertex of a regular octahedron with the same volume as the observed one (V = 4/3 * l_0^3)
        vol = ConvexHull(vecs).volume
        l_0 = (3.0 * vol / 4.0) ** (1.0 / 3.0)

        quadratic_elongation = float(np.mean((dists / l_0) ** 2))

        results.append({
            'B': b_idx,
            'B_ijk': B_ijk.get(b_idx),
            'quadratic_elongation': quadratic_elongation
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('B').reset_index(drop=True)
    return df