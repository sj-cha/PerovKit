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
from dataclasses import dataclass, field
import math
import os
from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize
from scipy.spatial import cKDTree

_ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

# The torsion loops are compiled with numba when it is installed (PEROVKIT_NUMBA=0 turns
# that off); otherwise the numpy code below runs them.
try:
    import numba
except ImportError:
    numba = None
JIT = numba is not None and os.environ.get("PEROVKIT_NUMBA", "1") != "0"


def _turn_torsions(X, phis, bonds, moving, starts):
    """
    Apply torsion changes `phis` (L, k) in place to poses X (L, n, 3), deepest first:
    torsion b turns atoms moving[starts[b]:starts[b + 1]] about bond (bonds[b, 0], bonds[b, 1]).
    """
    for l in range(X.shape[0]):
        for b in range(bonds.shape[0]):
            phi = phis[l, b]
            if phi == 0.0:
                continue
            i, j = bonds[b, 0], bonds[b, 1]
            px, py, pz = X[l, i, 0], X[l, i, 1], X[l, i, 2]
            ux, uy, uz = X[l, j, 0] - px, X[l, j, 1] - py, X[l, j, 2] - pz
            r = math.sqrt(ux * ux + uy * uy + uz * uz)
            ux, uy, uz = ux / r, uy / r, uz / r
            c, s = math.cos(phi), math.sin(phi)
            t = 1.0 - c
            r00, r01, r02 = c + t * ux * ux, t * ux * uy - s * uz, t * ux * uz + s * uy
            r10, r11, r12 = t * ux * uy + s * uz, c + t * uy * uy, t * uy * uz - s * ux
            r20, r21, r22 = t * ux * uz - s * uy, t * uy * uz + s * ux, c + t * uz * uz
            for m in range(starts[b], starts[b + 1]):
                a = moving[m]
                vx, vy, vz = X[l, a, 0] - px, X[l, a, 1] - py, X[l, a, 2] - pz
                X[l, a, 0] = r00 * vx + r01 * vy + r02 * vz + px
                X[l, a, 1] = r10 * vx + r11 * vy + r12 * vz + py
                X[l, a, 2] = r20 * vx + r21 * vy + r22 * vz + pz


def _torsion_gradient(X, G, bonds, moving, starts, out):
    """
    dE/dphi of every torsion (L, k) into `out`: the torque of the forces G on the atoms
    each torsion moves, about its bond as it lies in X.
    """
    for l in range(X.shape[0]):
        for b in range(bonds.shape[0]):
            i, j = bonds[b, 0], bonds[b, 1]
            px, py, pz = X[l, i, 0], X[l, i, 1], X[l, i, 2]
            ux, uy, uz = X[l, j, 0] - px, X[l, j, 1] - py, X[l, j, 2] - pz
            r = math.sqrt(ux * ux + uy * uy + uz * uz)
            tx = ty = tz = 0.0
            for m in range(starts[b], starts[b + 1]):
                a = moving[m]
                rx, ry, rz = X[l, a, 0] - px, X[l, a, 1] - py, X[l, a, 2] - pz
                gx, gy, gz = G[l, a, 0], G[l, a, 1], G[l, a, 2]
                tx += ry * gz - rz * gy
                ty += rz * gx - rx * gz
                tz += rx * gy - ry * gx
            out[l, b] = (ux * tx + uy * ty + uz * tz) / r


if JIT:
    _turn_torsions = numba.njit(cache=True)(_turn_torsions)
    _torsion_gradient = numba.njit(cache=True)(_torsion_gradient)


def _rotation(u: np.ndarray, phi: float) -> np.ndarray:
    """
    Rotation by `phi` about the unit axis `u` (Rodrigues), built from scalars: it runs once
    per torsion per evaluation, where numpy's per-call overhead dominated the refinement.
    """
    x, y, z = float(u[0]), float(u[1]), float(u[2])
    c, s = math.cos(phi), math.sin(phi)
    t = 1.0 - c
    return np.array([[c + t * x * x, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, c + t * y * y, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, c + t * z * z]])


def _rotations(u: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Rotations by `phi` (L,) about the unit axes `u` (L, 3): (L, 3, 3).
    """
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    t = 1.0 - c
    return np.stack([
        np.stack([c + t * x * x, t * x * y - s * z, t * x * z + s * y], axis=-1),
        np.stack([t * x * y + s * z, c + t * y * y, t * y * z - s * x], axis=-1),
        np.stack([t * x * z - s * y, t * y * z + s * x, c + t * z * z], axis=-1),
    ], axis=1)


def _euler_many(angles: np.ndarray):
    """
    _euler for (L, 3) angles: R (L, 3, 3) and its three derivatives (3, L, 3, 3).
    """
    L = len(angles)
    ca, sa = np.cos(angles[:, 0]), np.sin(angles[:, 0])
    cb, sb = np.cos(angles[:, 1]), np.sin(angles[:, 1])
    cc, sc = np.cos(angles[:, 2]), np.sin(angles[:, 2])
    zero, one = np.zeros(L), np.ones(L)

    def m(rows):
        return np.stack([np.stack(r, axis=-1) for r in rows], axis=1)
    Rx = m([[one, zero, zero], [zero, ca, -sa], [zero, sa, ca]])
    Ry = m([[cb, zero, sb], [zero, one, zero], [-sb, zero, cb]])
    Rz = m([[cc, -sc, zero], [sc, cc, zero], [zero, zero, one]])
    dRx = m([[zero, zero, zero], [zero, -sa, -ca], [zero, ca, -sa]])
    dRy = m([[-sb, zero, cb], [zero, zero, zero], [-cb, zero, -sb]])
    dRz = m([[-sc, -cc, zero], [cc, -sc, zero], [zero, zero, zero]])
    return Rz @ Ry @ Rx, np.stack([Rz @ Ry @ dRx, Rz @ dRy @ Rx, dRz @ Ry @ Rx])


def _dihedral(p: np.ndarray) -> float:
    """
    Dihedral angle (radians) of four points, positive for a right-handed turn of p[3]
    about the axis p[1] -> p[2].
    """
    b1 = p[2] - p[1]
    b1 = b1 / np.linalg.norm(b1)
    v = (p[0] - p[1]) - np.dot(p[0] - p[1], b1) * b1
    w = (p[3] - p[2]) - np.dot(p[3] - p[2], b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


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
        far (np.ndarray): (n_atoms, n_atoms) whether two atoms are at least four bonds
            apart, the pairs kept apart within the ligand.
        quads (np.ndarray): (k, 4) atoms (a, i, j, b) defining each torsion's dihedral.
        tails (np.ndarray): (k,) torsions whose moving side holds no binding atom, so they
            turn a dangling chain without moving any anchor.
        heavy (np.ndarray): (n_atoms,) non-hydrogen atoms.
    """
    bonds: np.ndarray
    masks: np.ndarray
    far: np.ndarray
    quads: np.ndarray
    tails: np.ndarray
    heavy: np.ndarray
    _moving: list = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # forward() and gradient() run thousands of times per refinement: index once
        self._moving = [(int(i), int(j), np.flatnonzero(m)) for (i, j), m in zip(self.bonds, self.masks)]
        self._weights = self.masks.astype(float)
        # the same, flat, for the compiled loops
        self._bonds = np.ascontiguousarray(self.bonds, dtype=np.int64).reshape(-1, 2)
        self._flat_moving = np.concatenate([m for *_, m in self._moving] + [np.zeros(0, dtype=np.int64)]).astype(np.int64)
        self._starts = np.concatenate([[0], np.cumsum([len(m) for *_, m in self._moving])]).astype(np.int64)

    @classmethod
    def build(cls, mol: Chem.Mol, root: int, binding: Sequence[int] = ()) -> TorsionModel:
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

        def outer(atom: int, other: int) -> int:
            cand = [x for x in nbrs[atom] if x != other]
            return max(cand, key=lambda x: heavy[x])  # a heavy neighbor when there is one
        quads = np.array([(outer(i, j), i, j, outer(j, i)) for i, j in bonds], dtype=int).reshape(-1, 4)
        binding = np.asarray(list(binding), dtype=int)
        tails = ~masks[:, binding].any(axis=1) if len(binding) else np.zeros(len(bonds), dtype=bool)
        return cls(bonds=bonds, masks=masks, far=topo >= 4, quads=quads, tails=tails, heavy=heavy)

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
        if JIT:
            x = np.array(x0, dtype=float)[None]
            _turn_torsions(x, np.ascontiguousarray(params[None, 6:], dtype=float),
                           self._bonds, self._flat_moving, self._starts)
            x = x[0]
        else:
            x = x0.copy()
            for (i, j, moving), phi in zip(self._moving, params[6:]):
                if phi == 0.0:
                    continue
                xi = x[i]
                u = x[j] - xi
                R = _rotation(u / math.sqrt(u @ u), phi)
                x[moving] = (x[moving] - xi) @ R.T + xi
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
        if self.n_torsions and JIT:
            out = np.empty((1, self.n_torsions))
            _torsion_gradient(np.ascontiguousarray(x[None]), np.ascontiguousarray(G[None]),
                              self._bonds, self._flat_moving, self._starts, out)
            g[6:] = out[0]
        elif self.n_torsions:
            i, j = self.bonds[:, 0], self.bonds[:, 1]
            u = x[j] - x[i]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            M = self._weights
            torque = M @ np.cross(x, G) - np.cross(x[i], M @ G)
            g[6:] = np.einsum("ki,ki->k", u, torque)
        return g

    def forward_many(self, P: np.ndarray, X0: np.ndarray, C: np.ndarray):
        """
        forward() for L poses of this ligand at once: params P (L, 6 + k), starting
        coordinates X0 (L, n, 3) and centers C (L, 3). Each torsion is applied to all L
        poses in one array operation, so the Python loop runs over the torsions only.
        """
        X = np.array(X0, dtype=float)
        if JIT:
            _turn_torsions(X, np.ascontiguousarray(P[:, 6:], dtype=float),
                           self._bonds, self._flat_moving, self._starts)
        else:
            for b, (i, j, moving) in enumerate(self._moving):
                phi = P[:, 6 + b]
                if not phi.any():
                    continue
                xi = X[:, i].copy()
                u = X[:, j] - xi
                u /= np.linalg.norm(u, axis=1, keepdims=True)
                R = _rotations(u, phi)
                X[:, moving] = np.matmul(X[:, moving] - xi[:, None], R.transpose(0, 2, 1)) + xi[:, None]
        R, dR = _euler_many(P[:, 3:6])
        Xt = X - C[:, None]
        return np.matmul(Xt, R.transpose(0, 2, 1)) + C[:, None] + P[:, None, :3], Xt, dR

    def gradient_many(self, P: np.ndarray, X: np.ndarray, Xt: np.ndarray, dR: np.ndarray,
                      G: np.ndarray) -> np.ndarray:
        """
        gradient() for L poses at once, after forward_many: (L, 6 + k).
        """
        g = np.empty_like(P)
        g[:, :3] = G.sum(axis=1)
        for k in range(3):
            g[:, 3 + k] = np.einsum("lnd,lnd->l", G, np.matmul(Xt, dR[k].transpose(0, 2, 1)))
        if self.n_torsions and JIT:
            out = np.empty((len(P), self.n_torsions))
            _torsion_gradient(np.ascontiguousarray(X), np.ascontiguousarray(G),
                              self._bonds, self._flat_moving, self._starts, out)
            g[:, 6:] = out
        elif self.n_torsions:
            i, j = self.bonds[:, 0], self.bonds[:, 1]
            u = X[:, j] - X[:, i]
            u /= np.linalg.norm(u, axis=2, keepdims=True)
            M = self._weights
            torque = (np.einsum("kn,lnd->lkd", M, np.cross(X, G))
                      - np.cross(X[:, i], np.einsum("kn,lnd->lkd", M, G)))
            g[:, 6:] = np.einsum("lkd,lkd->lk", u, torque)
        return g


@dataclass
class Refinement:
    coords: np.ndarray
    anchor_rmsd: float
    converged: bool


def _ligand_terms(model, x, groups, targets, trees, contact, k_anchor, k_contact, intra):
    """
    Anchor, contact and intramolecular penalties of one ligand and their Cartesian gradient.

    Returns (E, G, closest, tangled): the closest contact found with the fixed atoms (inf if
    none within `contact`) and whether any intramolecular pair is closer than `intra`.
    """
    G = np.zeros_like(x)
    E = 0.0
    closest = float("inf")
    tangled = False
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
            closest = min(closest, float(d.min()))
            v = (x[atom] - tree.data[idx[atom, slot]]) / d[:, None]
            over = contact - d
            E += k_contact * float(over @ over)
            np.add.at(G, atom, -2.0 * k_contact * over[:, None] * v)
    # close pairs from a tree rather than every pair four bonds apart (thousands of them)
    close = cKDTree(x).query_pairs(intra, output_type="ndarray")
    if len(close):
        pa, pb = close[:, 0], close[:, 1]
        keep = model.far[pa, pb]
        pa, pb = pa[keep], pb[keep]
        v = x[pa] - x[pb]
        d = np.linalg.norm(v, axis=1)
        hit = (d < intra) & (d > 1e-9)
        if hit.any():
            tangled = True
            pa, pb, d, v = pa[hit], pb[hit], d[hit], v[hit] / d[hit, None]
            over = intra - d
            E += k_contact * float(over @ over)
            f = -2.0 * k_contact * over[:, None] * v
            np.add.at(G, pa, f)
            np.add.at(G, pb, -f)
    return E, G, closest, tangled


def _many_terms(model, X, groups, T, trees, contact, k_anchor, k_contact, intra):
    """
    _ligand_terms for L poses of one ligand type, X (L, n, 3) and targets T (L, m, 3), with
    one tree query for all their atoms. Returns (E, G (L, n, 3), closest, tangled, rmsd (L,)).
    """
    L, n, _ = X.shape
    G = np.zeros_like(X)
    E = 0.0
    sq = np.zeros(L)
    for k, grp in enumerate(groups):
        d = X[:, grp].mean(axis=1) - T[:, k]
        E += k_anchor * float(np.sum(d * d))
        sq += np.sum(d * d, axis=1)
        G[:, grp] += (2.0 * k_anchor * d / len(grp))[:, None, :]
    flat, G_flat = X.reshape(-1, 3), G.reshape(-1, 3)
    closest, tangled = float("inf"), False
    for tree in trees:
        if tree is None or tree.n == 0:
            continue
        dist, idx = tree.query(flat, k=min(8, tree.n), distance_upper_bound=contact)
        dist = dist.reshape(len(flat), -1)
        idx = idx.reshape(len(flat), -1)
        atom, slot = np.nonzero(np.isfinite(dist) & (dist > 1e-9))
        if len(atom):
            d = dist[atom, slot]
            closest = min(closest, float(d.min()))
            v = (flat[atom] - tree.data[idx[atom, slot]]) / d[:, None]
            over = contact - d
            E += k_contact * float(over @ over)
            np.add.at(G_flat, atom, -2.0 * k_contact * over[:, None] * v)
    # one tree for the intramolecular pairs of all poses, each pose shifted far apart
    shifted = X + (np.arange(L) * 1.0e4)[:, None, None] * np.array([1.0, 0.0, 0.0])
    close = cKDTree(shifted.reshape(-1, 3)).query_pairs(intra, output_type="ndarray")
    if len(close):
        pa, pb = close[:, 0], close[:, 1]
        keep = model.far[pa % n, pb % n]
        pa, pb = pa[keep], pb[keep]
        v = flat[pa] - flat[pb]
        d = np.linalg.norm(v, axis=1)
        hit = (d < intra) & (d > 1e-9)
        if hit.any():
            tangled = True
            pa, pb, d, v = pa[hit], pb[hit], d[hit], v[hit] / d[hit, None]
            over = intra - d
            E += k_contact * float(over @ over)
            f = -2.0 * k_contact * over[:, None] * v
            np.add.at(G_flat, pa, f)
            np.add.at(G_flat, pb, -f)
    return E, G, closest, tangled, np.sqrt(sq / len(groups))


def _anchor_rmsd(x, groups, targets) -> float:
    anchors = np.array([x[g].mean(axis=0) for g in groups])
    return float(np.sqrt(np.mean(np.sum((anchors - targets) ** 2, axis=1))))


def _minimize(fun, n, bounds, max_iter, done):
    """
    L-BFGS-B that also stops at the first iterate `done` accepted; `done` holds the
    parameters of the last evaluation and whether that point was good enough.
    """
    def callback(intermediate_result):
        if done["ok"] and np.array_equal(done["x"], intermediate_result.x):
            raise StopIteration
    return minimize(fun, np.zeros(n), jac=True, method="L-BFGS-B", bounds=bounds,
                    callback=callback, options={"maxiter": max_iter})


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
    pull: Optional[tuple] = None,
    frozen: Optional[np.ndarray] = None,
    stop_at: Optional[float] = None,
    stop_rmsd: float = float("inf"),
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
        pull: Optional (direction, atom mask, k): lowers the energy by k per Å that the
            masked atoms move along the unit direction.
        frozen (np.ndarray): Optional (6 + k,) mask of parameters held at zero.
        stop_at, stop_rmsd: Stop as soon as every contact with the fixed atoms is at least
            `stop_at` Å, no intramolecular pair is tangled and the anchors are within
            `stop_rmsd` (RMS, Å): the pose is good enough, and the remaining iterations
            would only polish it.
    """
    center = targets.mean(axis=0)
    groups = [np.asarray(g, dtype=int) for g in groups]
    done = {"ok": False, "x": None}

    def fun(params):
        x, xt, dR = model.forward(params, coords, center)
        E, G, closest, tangled = _ligand_terms(model, x, groups, targets, trees, contact,
                                               k_anchor, k_contact, intra)
        done["x"] = params.copy()
        done["ok"] = (stop_at is not None and closest >= stop_at and not tangled
                      and _anchor_rmsd(x, groups, targets) <= stop_rmsd)
        if pull is not None:
            direction, mask, k_pull = pull
            E -= k_pull * float(np.sum((x[mask] - center) @ direction))
            G[mask] -= k_pull * direction
        phi = params[6:]
        E += w_torsion * float(np.sum(1.0 - np.cos(phi)))
        g = model.gradient(params, x, xt, dR, G)
        g[6:] += w_torsion * np.sin(phi)
        return E, g

    bounds = None
    if frozen is not None:
        bounds = [(0.0, 0.0) if f else (None, None) for f in frozen]
    res = _minimize(fun, 6 + model.n_torsions, bounds, max_iter, done)
    x, _, _ = model.forward(res.x, coords, center)
    return Refinement(coords=x, anchor_rmsd=_anchor_rmsd(x, groups, targets),
                      converged=bool(res.success))


def stretch(
    model: TorsionModel,
    coords: np.ndarray,
    groups: Sequence[Sequence[int]],
    targets: np.ndarray,
    normal: np.ndarray,
    trees: Sequence,
    contact: float,
    k_pull: float = 1.0,
    max_iter: int = 200,
    stop_at: Optional[float] = None,
) -> Optional[Refinement]:
    """
    Straighten the dangling chains of a ligand and raise them along `normal`.

    Every tail torsion is set anti (180°); then only the tail torsions are optimized, so no
    anchor moves, to lift the tail's heavy atoms along `normal` while keeping clear of the
    fixed atoms in `trees` and of the ligand itself. With `stop_at` it stops as soon as
    every contact with the fixed atoms is that far, when the point is only to unstick a
    chain from the surface rather than to stand it up. Returns None for a ligand without
    tails.
    """
    if not model.tails.any():
        return None
    x = coords.copy()
    for (i, j), mask, quad, tail in zip(model.bonds, model.masks, model.quads, model.tails):
        if not tail:
            continue
        u = x[j] - x[i]
        u /= np.linalg.norm(u)
        R = _rotation(u, np.pi - _dihedral(x[quad]))
        x[mask] = (x[mask] - x[i]) @ R.T + x[i]
    lifted = model.masks[model.tails].any(axis=0) & model.heavy
    frozen = np.concatenate([np.ones(6, dtype=bool), ~model.tails])
    return refine(model, x, groups, targets, trees, contact, max_iter=max_iter,
                  pull=(normal, lifted, k_pull), frozen=frozen, stop_at=stop_at)


def refine_joint(
    models: Sequence[TorsionModel],
    coords: Sequence[np.ndarray],
    groups: Sequence[Sequence[Sequence[int]]],
    targets: Sequence[np.ndarray],
    trees: Sequence,
    contact: float,
    tree_builder,
    k_anchor: float = 20.0,
    k_contact: float = 40.0,
    intra: float = 1.6,
    w_torsion: float = 0.02,
    max_iter: int = 400,
    stop_at: Optional[float] = None,
    stop_rmsd: float = float("inf"),
) -> List[Refinement]:
    """
    Refine several ligands at once, so that ligands jammed against each other can all
    give way. Each has the terms of `refine`; atoms of different ligands are also pushed
    `contact` Å apart. Those contacts are searched afresh at every evaluation, in a tree
    from `tree_builder` (periodic images included), since the ligands move. `stop_at` and
    `stop_rmsd` end it early as in `refine`, with every ligand meeting them. Ligands of one
    type are computed together, so a shell of 40 lecithins costs about as many Python-level
    steps per evaluation as a single one.
    """
    sizes = [6 + m.n_torsions for m in models]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    groups = [[np.asarray(g, dtype=int) for g in gs] for gs in groups]
    owner = np.concatenate([np.full(len(c), l) for l, c in enumerate(coords)])
    starts = np.concatenate([[0], np.cumsum([len(c) for c in coords])])
    done = {"ok": False, "x": None}

    # ligands of one type are computed together (forward_many and _many_terms)
    batches = []
    for model in {id(m): m for m in models}.values():
        members = [l for l, m in enumerate(models) if m is model]
        k = 6 + model.n_torsions
        batches.append(dict(
            model=model, groups=groups[members[0]],
            params=np.array([np.arange(offsets[l], offsets[l] + k) for l in members]),
            atoms=np.concatenate([np.arange(starts[l], starts[l + 1]) for l in members]),
            X0=np.array([coords[l] for l in members]),
            T=np.array([targets[l] for l in members]),
            C=np.array([targets[l].mean(axis=0) for l in members]),
        ))

    def fun(P):
        E = 0.0
        closest, tangled, worst = float("inf"), False, 0.0
        X = np.empty((starts[-1], 3))
        state = []
        for b in batches:
            Pb = P[b["params"]]
            Xb, Xt, dR = b["model"].forward_many(Pb, b["X0"], b["C"])
            e, G, c, t, rmsd = _many_terms(b["model"], Xb, b["groups"], b["T"], trees, contact,
                                           k_anchor, k_contact, intra)
            closest, tangled, worst = min(closest, c), tangled or t, max(worst, float(rmsd.max()))
            E += e + w_torsion * float(np.sum(1.0 - np.cos(Pb[:, 6:])))
            X[b["atoms"]] = Xb.reshape(-1, 3)
            state.append((Pb, Xb, Xt, dR, G))
        G_all = np.zeros_like(X)
        tree, image_of = tree_builder(X)
        dist, idx = tree.query(X, k=min(16, tree.n), distance_upper_bound=contact)
        p, slot = np.nonzero(np.isfinite(dist))
        q = image_of[idx[p, slot]]
        keep = owner[p] < owner[q]          # each inter-ligand contact once
        if keep.any():
            p, slot, q = p[keep], slot[keep], q[keep]
            dd = dist[p, slot]
            closest = min(closest, float(dd.min()))
            v = (X[p] - tree.data[idx[p, slot]]) / dd[:, None]
            over = contact - dd
            f = -2.0 * k_contact * over[:, None] * v
            E += k_contact * float(over @ over)
            np.add.at(G_all, p, f)
            np.add.at(G_all, q, -f)
        grad = np.empty_like(P)
        for b, (Pb, Xb, Xt, dR, G) in zip(batches, state):
            G += G_all[b["atoms"]].reshape(G.shape)
            g = b["model"].gradient_many(Pb, Xb, Xt, dR, G)
            g[:, 6:] += w_torsion * np.sin(Pb[:, 6:])
            grad[b["params"]] = g
        done["x"] = P.copy()
        done["ok"] = stop_at is not None and closest >= stop_at and not tangled and worst <= stop_rmsd
        return E, grad

    res = _minimize(fun, int(offsets[-1]), None, max_iter, done)
    out = []
    for l, model in enumerate(models):
        x, _, _ = model.forward(res.x[offsets[l]:offsets[l + 1]], coords[l], targets[l].mean(axis=0))
        out.append(Refinement(coords=x, anchor_rmsd=_anchor_rmsd(x, groups[l], targets[l]),
                              converged=bool(res.success)))
    return out
