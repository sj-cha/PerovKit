from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from ..assembly import Assembly
from ..core import Core
from ..utils.box import Box
from .glazer_pattern import build_octahedra_rotmat

# Some features are adapted from Terumasa Tadano's DistortPerovskite, and modified:
# https://github.com/ttadano/DistortPerovskite/tree/main


def _adjust_network(
    pos0: np.ndarray,
    b_keys: List[int],
    x_to_bs: Dict[int, List[int]],
    R_b: Dict[int, np.ndarray],
    box: Box,
) -> Dict[int, np.ndarray]:
    """
    Translation of every B that best keeps the corners shared once each octahedron is
    turned by its R_b: over every X shared by two B, the two places the turned octahedra
    put it should meet. A least-squares problem on the graph Laplacian of the B, with the
    first B held where it is.
    """
    n = len(b_keys)
    if n == 0:
        return {}
    b_to_idx = {b: i for i, b in enumerate(b_keys)}

    # Laplacian: L[i, i] += 1 and L[i, j] -= 1 for each directed edge i -> j
    deg = np.zeros(n, dtype=np.int32)
    rows, cols, data = [], [], []
    rhs = np.zeros((n, 3), dtype=float)

    def add_directed(i_b: int, j_b: int, delta_ij: np.ndarray):
        i, j = b_to_idx[i_b], b_to_idx[j_b]
        deg[i] += 1
        rows.append(i); cols.append(j); data.append(-1.0)
        rhs[i] -= delta_ij

    for x, bs in x_to_bs.items():
        for a in range(len(bs)):
            for b in range(a + 1, len(bs)):
                bi, bj = bs[a], bs[b]
                pred_i = pos0[bi] + R_b[bi] @ box.mic(pos0[x] - pos0[bi])
                pred_j = pos0[bj] + R_b[bj] @ box.mic(pos0[x] - pos0[bj])
                delta_ij = box.mic(pred_i - pred_j)
                add_directed(bi, bj, delta_ij)
                add_directed(bj, bi, -delta_ij)

    rows.extend(range(n))
    cols.extend(range(n))
    data.extend(deg.astype(float).tolist())
    L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()

    keep = np.arange(n) != 0
    lu = splu(L[keep][:, keep])   # fails if the network falls apart
    t = np.zeros((n, 3), dtype=float)
    for c in range(3):
        t[keep, c] = lu.solve(rhs[keep, c])
    return {b: t[i].copy() for i, b in enumerate(b_keys)}


def apply_tilt(
    structure: Core | Assembly,
    glazer: str,
    angles: Tuple[float, float, float],
    *,
    order: str = "xyz",
    move_ligands: bool = True,
):
    """
    Turn every octahedron by the Glazer pattern (see build_octahedra_rotmat), shift each B
    so that the corners stay shared (_adjust_network), and put each X at the mean of the
    places its B put it. With `move_ligands`, each ligand of an Assembly turns with the
    octahedron it is bound to.
    """
    octahedra = structure.octahedra
    b_keys = sorted(octahedra)
    pos0 = np.array(structure.atoms.positions, dtype=float, copy=True)
    pos_new = pos0.copy()

    R_b = build_octahedra_rotmat(glazer=glazer, angles=angles, b_ijk=structure.B_ijk, b_keys=b_keys,
                                 order=order)
    x_to_bs: Dict[int, List[int]] = {}
    for b, info in octahedra.items():
        for x in info["X"]:
            x_to_bs.setdefault(x, []).append(b)

    # minimum images along the periodic axes of the core
    box = (structure if isinstance(structure, Core) else structure.core).box
    t_b = _adjust_network(pos0, b_keys, x_to_bs, R_b, box)

    for b in b_keys:
        pos_new[b] = pos0[b] + t_b[b]

    # each X at the mean of where its B put it, as images near the first
    for x, bs in x_to_bs.items():
        preds = [(pos0[b] + t_b[b]) + (R_b[b] @ box.mic(pos0[x] - pos0[b])) for b in bs]
        preds[1:] = [preds[0] + box.mic(p - preds[0]) for p in preds[1:]]
        pos_new[x] = np.mean(preds, axis=0)

    # a ligand bound to two octahedra follows the later one
    if isinstance(structure, Assembly) and move_ligands:
        for b in b_keys:
            for lig_id in octahedra[b]["Ligand"]:
                idx = structure.ligands[lig_id].indices
                pos_new[idx] = (pos0[b] + t_b[b]) + (pos0[idx] - pos0[b]) @ R_b[b].T

    structure.atoms.positions[:] = pos_new
