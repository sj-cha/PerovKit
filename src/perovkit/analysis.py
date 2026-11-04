from typing import List, Optional, Tuple

from ase.geometry import find_mic
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


_MIC_CHUNK = 1 << 16    # vectors per find_mic call; bounds its temporaries to ~50 MB


def _as_frames(structure) -> Tuple[list, bool]:
    """
    Return (frames, is_trajectory); a single structure becomes a one-frame list.
    """
    if isinstance(structure, (list, tuple)):
        if len(structure) == 0:
            raise ValueError("Empty trajectory.")
        return list(structure), True
    return [structure], False


def _mic(vecs: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """
    Minimum-image vectors for an array of any leading shape (..., 3).

    Orthorhombic cells round each periodic axis, which is exact for them; other cells
    go through ase find_mic in chunks.
    """
    pbc = np.asarray(cell).any(axis=1) & np.asarray(pbc, dtype=bool)
    if vecs.size == 0 or not pbc.any():
        return vecs
    if np.count_nonzero(cell - np.diag(np.diag(cell))) == 0:
        out = vecs.copy()
        for ax in np.flatnonzero(pbc):
            L = cell[ax, ax]
            out[..., ax] -= L * np.round(out[..., ax] / L)
        return out

    flat = vecs.reshape(-1, 3)
    out = np.empty_like(flat)
    for start in range(0, len(flat), _MIC_CHUNK):
        out[start:start + _MIC_CHUNK], _ = find_mic(flat[start:start + _MIC_CHUNK], cell, pbc=pbc)
    return out.reshape(vecs.shape)


def _mic_vectors(frames: list, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    Minimum-image vectors r_j - r_i for index arrays `i`, `j` in every frame: (n_frames, n, 3).
    """
    idx, inv = np.unique(np.concatenate([i, j]), return_inverse=True)
    atoms = [getattr(s, "atoms", s) for s in frames]       # Assembly or ASE Atoms
    pos = np.stack([a.positions[idx] for a in atoms])
    vecs = pos[:, inv[len(i):]] - pos[:, inv[:len(i)]]

    cells = np.stack([a.cell.array for a in atoms])
    pbcs = np.stack([a.pbc for a in atoms])
    if (cells == cells[0]).all() and (pbcs == pbcs[0]).all():
        return _mic(vecs, cells[0], pbcs[0])
    return np.stack([_mic(v, c, p) for v, c, p in zip(vecs, cells, pbcs)])


def _octahedra(structure, B_sites) -> Tuple[list, bool, dict, dict, set]:
    """
    The frames, whether they are a trajectory, the octahedra and lattice coordinates of the
    first, and the B atoms to consider: those of `B_sites`, all by default.
    """
    frames, is_traj = _as_frames(structure)
    octahedra = frames[0].octahedra
    b_set = set(B_sites) if B_sites is not None else set(octahedra.keys())
    return frames, is_traj, octahedra, frames[0].B_ijk, b_set


def _table(rows: list, by: list) -> pd.DataFrame:
    """The rows as a DataFrame sorted by the columns `by`."""
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by).reset_index(drop=True)
    return df


def _per_row(values: np.ndarray, is_traj: bool) -> list:
    """
    Split (n_frames, n) values into n row entries: (n_frames,) arrays, or scalars for one structure.
    """
    return list(np.ascontiguousarray(values.T)) if is_traj else list(values[0])


def compute_distances(structure,
                      pairs,
                      vector: bool = False
                      ) -> np.ndarray:
    """
    Minimum-image distances between atom pairs.

    Args:
        structure: Assembly or ASE Atoms, or a list of them (a trajectory).
        pairs: (n, 2) atom indices (i, j).
        vector (bool): Return the vectors r_j - r_i instead of their lengths.

    Returns:
        (n,) distances or (n, 3) vectors for a single structure;
        (n_frames, n) or (n_frames, n, 3) for a trajectory.
    """
    frames, is_traj = _as_frames(structure)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    vecs = _mic_vectors(frames, pairs[:, 0], pairs[:, 1])
    out = vecs if vector else np.linalg.norm(vecs, axis=-1)
    return out if is_traj else out[0]


def compute_angles(structure,
                   triplets
                   ) -> np.ndarray:
    """
    Angles (deg) at the middle atom j of (i, j, k) triplets, minimum image.

    Args:
        structure: Assembly or ASE Atoms, or a list of them (a trajectory).
        triplets: (n, 3) atom indices (i, j, k).

    Returns:
        (n,) for a single structure; (n_frames, n) for a trajectory.
    """
    frames, is_traj = _as_frames(structure)
    t = np.asarray(triplets, dtype=int).reshape(-1, 3)
    n = len(t)
    vecs = _mic_vectors(frames, np.concatenate([t[:, 1], t[:, 1]]), np.concatenate([t[:, 0], t[:, 2]]))
    v1, v2 = vecs[:, :n], vecs[:, n:]
    cos = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    out = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out if is_traj else out[0]


def compute_B_X_B_angles(structure, B_sites: Optional[List[int]] = None) -> pd.DataFrame:
    """
    B-X-B angles (deg) at every X shared by two B sites of `B_sites` (all B by default).

    Returns a DataFrame with columns B1, B2, X, B1_ijk, B2_ijk, angles.
    """
    frames, is_traj, octahedra, B_ijk, b_set = _octahedra(structure, B_sites)

    # Build reverse map: X_index -> list of B indices that contain it
    x_to_b = {}
    for b in b_set:
        if b not in octahedra:
            continue
        for x_idx in octahedra[b]['X']:
            x_to_b.setdefault(x_idx, []).append(b)

    # One row for each shared X atom and pair of B atoms holding it
    results = []
    for x_idx, b_list in x_to_b.items():
        for i in range(len(b_list)):
            for j in range(i + 1, len(b_list)):
                b1, b2 = b_list[i], b_list[j]
                results.append({
                    'B1': b1, 'B2': b2, 'X': x_idx,
                    'B1_ijk': B_ijk.get(b1), 'B2_ijk': B_ijk.get(b2),
                })

    angles = compute_angles(frames, [(r['B1'], r['X'], r['B2']) for r in results])
    for r, angle in zip(results, _per_row(angles, is_traj)):
        r['angles'] = angle
    return _table(results, ['B1', 'B2', 'X'])


def compute_X_B_X_angles(structure, B_sites: Optional[List[int]] = None) -> pd.DataFrame:
    """
    X-B-X angles (deg) inside the octahedron of every B in `B_sites` (all B by default).

    Returns a DataFrame with columns B, X1, X2, B_ijk, angles.
    """
    frames, is_traj, octahedra, B_ijk, b_set = _octahedra(structure, B_sites)

    # One row for each pair of X atoms in an octahedron
    results = []
    for b_idx, entry in octahedra.items():
        if b_idx not in b_set:
            continue
        x_list = entry['X']
        for i in range(len(x_list)):
            for j in range(i + 1, len(x_list)):
                results.append({
                    'B': b_idx, 'X1': x_list[i], 'X2': x_list[j],
                    'B_ijk': B_ijk.get(b_idx),
                })

    angles = compute_angles(frames, [(r['X1'], r['B'], r['X2']) for r in results])
    for r, angle in zip(results, _per_row(angles, is_traj)):
        r['angles'] = angle
    return _table(results, ['B', 'X1', 'X2'])


def compute_B_X_distances(structure, B_sites: Optional[List[int]] = None) -> pd.DataFrame:
    """
    B-X bond lengths (Å) of the octahedron of every B in `B_sites` (all B by default).

    Returns a DataFrame with columns B, X, B_ijk, distances.
    """
    frames, is_traj, octahedra, B_ijk, b_set = _octahedra(structure, B_sites)

    results = [
        {'B': b_idx, 'X': x_idx, 'B_ijk': B_ijk.get(b_idx)}
        for b_idx, entry in octahedra.items() if b_idx in b_set
        for x_idx in entry['X']
    ]

    dists = compute_distances(frames, [(r['B'], r['X']) for r in results])
    for r, d in zip(results, _per_row(dists, is_traj)):
        r['distances'] = d
    return _table(results, ['B', 'X'])


def compute_bond_angle_variance(structure, B_sites: Optional[List[int]] = None,
                                linear_threshold: float = 130.0) -> pd.DataFrame:
    """
    Bond angle variance (deg^2) of every octahedron: sum of (angle - 90)^2 / (n - 1) over
    its n cis X-B-X angles, i.e. those below `linear_threshold`. For a trajectory the cis
    angles are picked in every frame.

    Returns a DataFrame with columns B, B_ijk, bond_angle_variance.
    """
    frames, is_traj = _as_frames(structure)
    df = compute_X_B_X_angles(structure, B_sites)

    if is_traj:
        columns = ['B', 'B_ijk', 'bond_angle_variance']
        if df.empty:
            return pd.DataFrame(columns=columns)
        results = []
        for b_idx, group in df.groupby('B'):
            angles = np.stack(group['angles'].to_list(), axis=1)        # (n_frames, n_angles)
            cis = angles < linear_threshold
            with np.errstate(divide='ignore', invalid='ignore'):
                variance = np.where(cis, (angles - 90) ** 2, 0.0).sum(axis=1) / (cis.sum(axis=1) - 1)
            results.append({'B': b_idx, 'B_ijk': frames[0].B_ijk.get(b_idx),
                            'bond_angle_variance': variance})
        return pd.DataFrame(results, columns=columns)

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


def compute_distortion_index(structure, B_sites: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Distortion index of every octahedron: mean |d - <d>| / <d> over its B-X bond lengths d.

    Returns a DataFrame with columns B, B_ijk, distortion_index.
    """
    frames, _ = _as_frames(structure)
    df = compute_B_X_distances(structure, B_sites)
    if df.empty:
        return pd.DataFrame()

    results = []
    for b_idx, group in df.groupby('B'):
        dists = np.stack(group['distances'].to_list(), axis=-1)        # (n_X,) or (n_frames, n_X)
        mean = dists.mean(axis=-1, keepdims=True)
        D = np.mean(np.abs(dists - mean) / mean, axis=-1)
        results.append({
            'B': b_idx,
            'B_ijk': frames[0].B_ijk.get(b_idx),
            'distortion_index': D
        })

    return pd.DataFrame(results)


def compute_quadratic_elongation(structure, B_sites: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Quadratic elongation of every complete (6-X) octahedron: mean (d / l_0)^2, with l_0 the
    center-to-vertex distance of a regular octahedron of the same volume. For a trajectory the
    volume needs a convex hull per frame, so this one is slow on long trajectories.

    Returns a DataFrame with columns B, B_ijk, quadratic_elongation.
    """
    frames, is_traj, octahedra, B_ijk, b_set = _octahedra(structure, B_sites)

    sites = [(b_idx, entry['X']) for b_idx, entry in octahedra.items()
             if b_idx in b_set and len(entry['X']) == 6]
    vecs = compute_distances(frames, [(b_idx, x_idx) for b_idx, x_list in sites for x_idx in x_list],
                             vector=True)                               # (n_frames, 6 * n_sites, 3)

    results = []
    for k, (b_idx, x_list) in enumerate(sites):
        v = vecs[:, 6 * k:6 * k + 6]
        dists = np.linalg.norm(v, axis=-1)

        # Robinson l_0: center-to-vertex distance of the regular octahedron with the
        # observed volume (V = 4/3 * l_0^3)
        vol = np.array([ConvexHull(vf).volume for vf in v])
        l_0 = (3.0 * vol / 4.0) ** (1.0 / 3.0)

        quadratic_elongation = np.mean((dists / l_0[:, None]) ** 2, axis=1)

        results.append({
            'B': b_idx,
            'B_ijk': B_ijk.get(b_idx),
            'quadratic_elongation': quadratic_elongation if is_traj else float(quadratic_elongation[0])
        })
    return _table(results, 'B')
