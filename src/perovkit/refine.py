"""
Torsion-space refinement of a multi-motif ligand against fixed surroundings.

The ligand moves by its rotatable torsions plus a small rigid-body adjustment, so bond
lengths and angles stay those of its conformer. The objective keeps every anchor on its
target and pushes atoms out of contact with the core, neighboring ligands and the ligand
itself, with soft-sphere penalties; gradients are analytic and the problem goes to scipy's
L-BFGS-B. This is a geometry generator, not a force field: it removes clashes that a rigid
conformer cannot, and says nothing about binding energies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize

_ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def _rotation(u: np.ndarray, phi: float) -> np.ndarray:
    x, y, z = u
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + np.sin(phi) * K + (1.0 - np.cos(phi)) * (K @ K)


def _euler(a: float, b: float, c: float):
    """
    R = Rz(c) Ry(b) Rx(a) and its derivatives with respect to a, b, c.
    """
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    dRx = np.array([[0, 0, 0], [0, -sa, -ca], [0, ca, -sa]])
    dRy = np.array([[-sb, 0, cb], [0, 0, 0], [-cb, 0, -sb]])
    dRz = np.array([[-sc, -cc, 0], [cc, -sc, 0], [0, 0, 0]])
    return Rz @ Ry @ Rx, (Rz @ Ry @ dRx, Rz @ dRy @ Rx, dRz @ Ry @ Rx)


@dataclass
class TorsionModel:
    """
    Rotatable torsions of one ligand type and the pairs it must keep apart.

    Attributes:
        bonds (np.ndarray): (k, 2) atoms (i, j) of each rotatable bond; turning it moves the
            side of j, which never holds the root (the first binding atom).
        masks (np.ndarray): (k, n_atoms) atoms moved by each torsion, deepest bond first.
        pairs (np.ndarray): (q, 2) atom pairs at least four bonds apart.
    """
    bonds: np.ndarray
    masks: np.ndarray
    pairs: np.ndarray

    @classmethod
    def build(cls, mol: Chem.Mol, root: int) -> TorsionModel:
        n = mol.GetNumAtoms()
        heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
        nbrs = [[x.GetIdx() for x in a.GetNeighbors()] for a in mol.GetAtoms()]
        topo = Chem.GetDistanceMatrix(mol)

        def side(start: int, cut: int) -> np.ndarray:
            seen = {start}
            stack = [start]
            while stack:
                a = stack.pop()
                for b in nbrs[a]:
                    if b not in seen and not (a == start and b == cut):
                        seen.add(b)
                        stack.append(b)
            mask = np.zeros(n, dtype=bool)
            mask[list(seen)] = True
            return mask

        found = []
        for i, j in mol.GetSubstructMatches(_ROTATABLE):
            mask = side(j, i)
            if mask[root]:
                i, j = j, i
                mask = side(j, i)
            if not heavy[mask].any():
                continue  # methyl and ammonium rotors move hydrogens only
            found.append((topo[root, i], i, j, mask))
        found.sort(key=lambda f: -f[0])  # deepest first, see forward()

        bonds = np.array([(i, j) for _, i, j, _ in found], dtype=int).reshape(-1, 2)
        masks = np.array([m for *_, m in found], dtype=bool).reshape(-1, n)
        pairs = np.argwhere(np.triu(topo >= 4))
        return cls(bonds=bonds, masks=masks, pairs=pairs)

    @property
    def n_torsions(self) -> int:
        return len(self.bonds)

    def forward(self, params: np.ndarray, x0: np.ndarray, center: np.ndarray):
        """
        Coordinates for params = (translation (3), Euler angles (3), torsion changes (k)).

        Torsions apply deepest first: each later (shallower) one moves its side, deeper
        bonds included, rigidly, so the derivative of every torsion is a rotation about its
        bond as it lies in the final coordinates.
        """
        x = x0.copy()
        for (i, j), mask, phi in zip(self.bonds, self.masks, params[6:]):
            if phi == 0.0:
                continue
            u = x[j] - x[i]
            R = _rotation(u / np.linalg.norm(u), phi)
            x[mask] = (x[mask] - x[i]) @ R.T + x[i]
        R, dR = _euler(*params[3:6])
        xt = x - center
        return xt @ R.T + center + params[:3], xt, dR

    def gradient(self, params: np.ndarray, x: np.ndarray, xt: np.ndarray, dR, G: np.ndarray) -> np.ndarray:
        """
        Chain rule from the Cartesian gradient G (n_atoms, 3) to the parameters.
        """
        g = np.empty(len(params))
        g[:3] = G.sum(axis=0)
        for k in range(3):
            g[3 + k] = np.sum(G * (xt @ dR[k].T))
        if self.n_torsions:
            i, j = self.bonds[:, 0], self.bonds[:, 1]
            u = x[j] - x[i]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            M = self.masks.astype(float)
            torque = M @ np.cross(x, G) - np.cross(x[i], M @ G)
            g[6:] = np.einsum("ki,ki->k", u, torque)
        return g


@dataclass
class Refinement:
    coords: np.ndarray
    anchor_rmsd: float
    converged: bool


def _ligand_terms(model, x, groups, targets, trees, contact, k_anchor, k_contact, intra):
    """
    Anchor, contact and intramolecular penalties of one ligand and their Cartesian gradient.
    """
    G = np.zeros_like(x)
    E = 0.0
    for grp, t in zip(groups, targets):
        d = x[grp].mean(axis=0) - t
        E += k_anchor * float(d @ d)
        G[grp] += 2.0 * k_anchor * d / len(grp)
    for tree in trees:
        if tree is None or tree.n == 0:
            continue
        dist, idx = tree.query(x, k=min(8, tree.n), distance_upper_bound=contact)
        dist = dist.reshape(len(x), -1)
        idx = idx.reshape(len(x), -1)
        atom, slot = np.nonzero(np.isfinite(dist) & (dist > 1e-9))
        if len(atom):
            d = dist[atom, slot]
            v = (x[atom] - tree.data[idx[atom, slot]]) / d[:, None]
            over = contact - d
            E += k_contact * float(over @ over)
            np.add.at(G, atom, -2.0 * k_contact * over[:, None] * v)
    pa, pb = model.pairs[:, 0], model.pairs[:, 1]
    if len(pa):
        v = x[pa] - x[pb]
        d = np.linalg.norm(v, axis=1)
        hit = d < intra
        if hit.any():
            d, v = d[hit], v[hit] / d[hit, None]
            over = intra - d
            E += k_contact * float(over @ over)
            f = -2.0 * k_contact * over[:, None] * v
            np.add.at(G, pa[hit], f)
            np.add.at(G, pb[hit], -f)
    return E, G


def refine(
    model: TorsionModel,
    coords: np.ndarray,
    groups: Sequence[Sequence[int]],
    targets: np.ndarray,
    trees: Sequence,
    contact: float,
    k_anchor: float = 20.0,
    k_contact: float = 40.0,
    intra: float = 1.6,
    w_torsion: float = 0.02,
    max_iter: int = 400,
) -> Refinement:
    """
    Refine one ligand pose.

    Args:
        model (TorsionModel): Torsions and intramolecular pairs of the ligand.
        coords (np.ndarray): (n_atoms, 3) starting pose.
        groups: Binding atoms of each motif.
        targets (np.ndarray): (n_motifs, 3) anchor targets.
        trees: KD-trees of fixed atoms to stay `contact` Å away from (core, neighbors);
            their ``data`` must hold the positions, periodic images included.
        contact (float): Soft-sphere radius (Å) against the fixed atoms.
        k_anchor, k_contact (float): Weights (per Å^2) of anchor and contact penalties.
        intra (float): Soft-sphere radius (Å) between ligand atoms four or more bonds apart.
        w_torsion (float): Weight of 1 - cos(change) per torsion, a mild pull to the conformer.
        max_iter (int): L-BFGS-B iterations.
    """
    center = targets.mean(axis=0)
    groups = [np.asarray(g, dtype=int) for g in groups]

    def fun(params):
        x, xt, dR = model.forward(params, coords, center)
        E, G = _ligand_terms(model, x, groups, targets, trees, contact, k_anchor, k_contact, intra)
        phi = params[6:]
        E += w_torsion * float(np.sum(1.0 - np.cos(phi)))
        g = model.gradient(params, x, xt, dR, G)
        g[6:] += w_torsion * np.sin(phi)
        return E, g

    res = minimize(fun, np.zeros(6 + model.n_torsions), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    x, _, _ = model.forward(res.x, coords, center)
    anchors = np.array([x[g].mean(axis=0) for g in groups])
    rmsd = float(np.sqrt(np.mean(np.sum((anchors - targets) ** 2, axis=1))))
    return Refinement(coords=x, anchor_rmsd=rmsd, converged=bool(res.success))


def refine_joint(
    models: Sequence[TorsionModel],
    coords: Sequence[np.ndarray],
    groups: Sequence[Sequence[Sequence[int]]],
    targets: Sequence[np.ndarray],
    trees: Sequence,
    contact: float,
    pairs: Sequence[tuple],
    mic=lambda d: d,
    k_anchor: float = 20.0,
    k_contact: float = 40.0,
    intra: float = 1.6,
    w_torsion: float = 0.02,
    max_iter: int = 400,
) -> List[Refinement]:
    """
    Refine several ligands at once, so that ligands jammed against each other can all
    give way. Each has the terms of `refine`; every pair (a, b) in `pairs` also pushes the
    two ligands `contact` Å apart, with differences taken through `mic`.
    """
    sizes = [6 + m.n_torsions for m in models]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    centers = [t.mean(axis=0) for t in targets]
    groups = [[np.asarray(g, dtype=int) for g in gs] for gs in groups]

    def fun(P):
        E = 0.0
        state = []
        for l, model in enumerate(models):
            params = P[offsets[l]:offsets[l + 1]]
            x, xt, dR = model.forward(params, coords[l], centers[l])
            e, G = _ligand_terms(model, x, groups[l], targets[l], trees, contact, k_anchor, k_contact, intra)
            E += e + w_torsion * float(np.sum(1.0 - np.cos(params[6:])))
            state.append((params, x, xt, dR, G))
        for a, b in pairs:
            xa, xb = state[a][1], state[b][1]
            v = mic(xa[:, None, :] - xb[None, :, :])
            d = np.linalg.norm(v, axis=-1)
            ia, ib = np.nonzero((d < contact) & (d > 1e-9))
            if len(ia):
                dd = d[ia, ib]
                over = contact - dd
                f = -2.0 * k_contact * over[:, None] * v[ia, ib] / dd[:, None]
                E += k_contact * float(over @ over)
                np.add.at(state[a][4], ia, f)
                np.add.at(state[b][4], ib, -f)
        grad = np.empty_like(P)
        for l, (params, x, xt, dR, G) in enumerate(state):
            g = models[l].gradient(params, x, xt, dR, G)
            g[6:] += w_torsion * np.sin(params[6:])
            grad[offsets[l]:offsets[l + 1]] = g
        return E, grad

    res = minimize(fun, np.zeros(offsets[-1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    out = []
    for l, model in enumerate(models):
        x, _, _ = model.forward(res.x[offsets[l]:offsets[l + 1]], coords[l], centers[l])
        anchors = np.array([x[g].mean(axis=0) for g in groups[l]])
        rmsd = float(np.sqrt(np.mean(np.sum((anchors - targets[l]) ** 2, axis=1))))
        out.append(Refinement(coords=x, anchor_rmsd=rmsd, converged=bool(res.success)))
    return out
