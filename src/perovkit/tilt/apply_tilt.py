from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .glazer_pattern import build_octahedra_rotmat
from perovkit import Core, NanoCrystal, Slab

"""
Some features have beem adapted from Terumasa Tadano's codes and modified.   
Please refer to DistortPerovskite: https://github.com/ttadano/DistortPerovskite/tree/main
"""


def _mic(v, cell_inv, cell_arr, pbc):
    """Minimum-image convention for displacement vector(s) *v*."""
    if cell_inv is None:
        return v
    frac = v @ cell_inv
    for ax in range(3):
        if pbc[ax]:
            frac[..., ax] -= np.round(frac[..., ax])
    return frac @ cell_arr


def _build_X_to_B(octahedra: Dict[int, dict]) -> Dict[int, list[int]]:
    X_to_B = defaultdict(list)
    for b, info in octahedra.items():
        for x in info.get("X", []):
            X_to_B[int(x)].append(int(b))
    return dict(X_to_B)


def _adjust_network(
    pos0: np.ndarray,
    b_keys: np.ndarray,
    x_to_bs: Dict[int, list[int]],
    R_b: Dict[int, np.ndarray],
    cell_inv: np.ndarray | None = None,
    cell_arr: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> Dict[int, np.ndarray]:

    b_keys_list = [int(b) for b in b_keys]
    n = len(b_keys_list)
    if n == 0:
        return {}

    b_to_idx = {b: i for i, b in enumerate(b_keys_list)}

    # Build directed edges i->j with delta_ij and degree counts
    deg = np.zeros(n, dtype=np.int32)
    rows = []
    cols = []
    data = []
    rhs = np.zeros((n, 3), dtype=float)

    def add_directed(i_b: int, j_b: int, delta_ij: np.ndarray):
        i = b_to_idx[i_b]
        j = b_to_idx[j_b]
        # Laplacian: L[i,i]+=1 ; L[i,j]-=1
        deg[i] += 1
        rows.append(i); cols.append(j); data.append(-1.0)
        rhs[i] -= delta_ij

    for x, bs in x_to_bs.items():
        if len(bs) <= 1:
            continue

        x = int(x)
        for a in range(len(bs)):
            for b in range(a + 1, len(bs)):
                bi, bj = int(bs[a]), int(bs[b])

                Ri = R_b.get(bi, np.eye(3))
                Rj = R_b.get(bj, np.eye(3))

                rbi0 = pos0[bi]
                rbj0 = pos0[bj]
                vi = _mic(pos0[x] - rbi0, cell_inv, cell_arr, pbc)
                vj = _mic(pos0[x] - rbj0, cell_inv, cell_arr, pbc)

                pred_i = rbi0 + Ri @ vi
                pred_j = rbj0 + Rj @ vj
                delta_ij = _mic(pred_i - pred_j, cell_inv, cell_arr, pbc)

                add_directed(bi, bj, delta_ij)
                add_directed(bj, bi, -delta_ij)

    # Add diagonal from degrees
    rows.extend(list(range(n)))
    cols.extend(list(range(n)))
    data.extend(deg.astype(float).tolist())

    L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()

    # Gauge fix: remove index 0
    keep = np.arange(n) != 0
    Lr = L[keep][:, keep]          # sparse
    rr = rhs[keep, :]              # dense

    # Factorize once, solve 3 RHS
    lu = splu(Lr)  # will error if graph is disconnected & singular
    t = np.zeros((n, 3), dtype=float)
    for c in range(3):
        t[keep, c] = lu.solve(rr[:, c])

    return {b_keys_list[i]: t[i].copy() for i in range(n)}


def apply_tilt(
    structure: Core | NanoCrystal | Slab,
    glazer: str,
    angles: Tuple[float, float, float],
    *,
    order: str = "xyz",
    move_ligands: bool = True,
):

    octahedra = structure.octahedra
    b_keys = np.array(sorted(octahedra.keys()), dtype=int)

    pos0 = np.array(structure.atoms.positions, dtype=float, copy=True)
    pos_new = np.array(pos0, copy=True)

    # Ligand maps (NanoCrystal / Slab only)
    lig_to_b = {}
    lig_global_indices = {}

    if isinstance(structure, (NanoCrystal, Slab)) and move_ligands:
        for b in b_keys:
            for lig_id in octahedra[int(b)].get("Ligand", []):
                lig_id = int(lig_id)
                lig_to_b[lig_id] = int(b)

                lig = structure.ligands[lig_id]
                lig_global_indices[lig_id] = np.asarray(lig.indices, dtype=int)

    b_ijk = structure.B_ijk

    # Per-B rotation matrices following Glazer phase rule
    R_b = build_octahedra_rotmat(
        glazer=glazer,
        angles=angles,
        b_ijk=b_ijk,     
        b_keys=b_keys,
        order=order,
    )

    x_to_bs = _build_X_to_B(octahedra)

    # Periodic boundary info
    atoms = structure.atoms if isinstance(structure, Core) else structure.core.atoms
    pbc = np.asarray(atoms.pbc)
    if any(pbc):
        cell_arr = np.asarray(atoms.get_cell(), dtype=float)
        cell_inv = np.linalg.inv(cell_arr)
    else:
        cell_arr = cell_inv = None

    # Solve translations to restore corner sharing
    t_b = _adjust_network(pos0, b_keys, x_to_bs, R_b, cell_inv, cell_arr, pbc)

    # 1) B: translate only
    for b in b_keys:
        b = int(b)
        pos_new[b] = pos0[b] + t_b.get(b, np.zeros(3))

    # 2) X: average predictions from all connected B
    for x, bs in x_to_bs.items():
        preds = []
        for b in bs:
            R = R_b.get(int(b), np.eye(3))
            tb = t_b.get(int(b), np.zeros(3))
            rb0 = pos0[int(b)]
            v0 = _mic(pos0[int(x)] - rb0, cell_inv, cell_arr, pbc)
            preds.append((rb0 + tb) + (R @ v0))
        if len(preds) > 1:
            ref = preds[0]
            for i in range(1, len(preds)):
                preds[i] = ref + _mic(preds[i] - ref, cell_inv, cell_arr, pbc)
        pos_new[int(x)] = np.mean(preds, axis=0)

    # 3) Ligands: rigidly follow assigned B (row-vector update)
    if isinstance(structure, (NanoCrystal, Slab)) and move_ligands:
        for lig_id, b in lig_to_b.items():
            b = int(b)
            R = R_b.get(b, np.eye(3))
            tb = t_b.get(b, np.zeros(3))
            rb0 = pos0[b]
            gidx = lig_global_indices[int(lig_id)]

            # row-vector convention
            pos_new[gidx] = (rb0 + tb) + (pos0[gidx] - rb0) @ R.T

    if isinstance(structure, Core):
        structure.atoms.positions[:] = pos_new[: len(structure.atoms)]
        return

    n_core = len(structure.core.atoms)
    structure.core.atoms.positions[:] = pos_new[:n_core]

    if move_ligands:
        for lig_id, gidx in lig_global_indices.items():
            lig = structure.ligands[int(lig_id)]
            lig.atoms.positions[:] = pos_new[np.asarray(gidx, dtype=int)]