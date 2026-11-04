from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

from ase.data import vdw_radii
import numba
import numpy as np
from rdkit import Chem

# acyclic single bonds between atoms neither terminal nor triple-bonded
ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


@numba.njit(cache=True)
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


@numba.njit(cache=True)
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


def _euler_many(angles: np.ndarray):
    """
    _euler for (L, 3) angles: R (L, 3, 3) and its three derivatives (3, L, 3, 3).
    """
    L = len(angles)
    ca, sa = np.cos(angles[:, 0]), np.sin(angles[:, 0])
    cb, sb = np.cos(angles[:, 1]), np.sin(angles[:, 1])
    cc, sc = np.cos(angles[:, 2]), np.sin(angles[:, 2])
    Rx, Ry, Rz, dRx, dRy, dRz = (np.zeros((L, 3, 3)) for _ in range(6))
    Rx[:, 0, 0], Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = 1.0, ca, -sa, sa, ca
    Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 1, 1], Ry[:, 2, 0], Ry[:, 2, 2] = cb, sb, 1.0, -sb, cb
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cc, -sc, sc, cc, 1.0
    dRx[:, 1, 1], dRx[:, 1, 2], dRx[:, 2, 1], dRx[:, 2, 2] = -sa, -ca, ca, -sa
    dRy[:, 0, 0], dRy[:, 0, 2], dRy[:, 2, 0], dRy[:, 2, 2] = -sb, cb, -cb, -sb
    dRz[:, 0, 0], dRz[:, 0, 1], dRz[:, 1, 0], dRz[:, 1, 1] = -sc, -cc, cc, -sc
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
        radii (np.ndarray): (n_atoms,) van der Waals radii (Å, ase.data.vdw_radii).
    """
    bonds: np.ndarray
    masks: np.ndarray
    far: np.ndarray
    quads: np.ndarray
    tails: np.ndarray
    heavy: np.ndarray
    radii: np.ndarray

    def __post_init__(self):
        # the bonds and the atoms each torsion moves, flat, for the compiled loops
        moving = [np.flatnonzero(m) for m in self.masks]
        self._bonds = np.ascontiguousarray(self.bonds, dtype=np.int64).reshape(-1, 2)
        self._flat_moving = np.concatenate(moving + [np.zeros(0, dtype=np.int64)]).astype(np.int64)
        self._starts = np.concatenate([[0], np.cumsum([len(m) for m in moving])]).astype(np.int64)

    @property
    def frame(self) -> np.ndarray:
        """
        (n_atoms,) atoms that no tail torsion moves: what holds the binding groups together,
        which turning the dangling chains and rings cannot take out of a neighbor's way.
        """
        return ~self.masks[self.tails].any(axis=0)

    @classmethod
    def build(cls, mol: Chem.Mol, binding: Sequence[int]) -> TorsionModel:
        """Torsions of `mol` rooted at its first binding atom, with `binding` its binding atoms."""
        root = binding[0]
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
        for i, j in mol.GetSubstructMatches(ROTATABLE):
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
        tails = ~masks[:, binding].any(axis=1)
        # 1.8 Å for the elements ASE has no van der Waals radius for
        radii = np.nan_to_num(vdw_radii[[a.GetAtomicNum() for a in mol.GetAtoms()]], nan=1.8)
        return cls(bonds=bonds, masks=masks, far=topo >= 4, quads=quads, tails=tails, heavy=heavy,
                   radii=radii)

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
        x = np.array(x0, dtype=float)[None]
        _turn_torsions(x, np.ascontiguousarray(params[None, 6:], dtype=float),
                       self._bonds, self._flat_moving, self._starts)
        x = x[0]
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
            out = np.empty((1, self.n_torsions))
            _torsion_gradient(np.ascontiguousarray(x[None]), np.ascontiguousarray(G[None]),
                              self._bonds, self._flat_moving, self._starts, out)
            g[6:] = out[0]
        return g

    def forward_many(self, P: np.ndarray, X0: np.ndarray, C: np.ndarray):
        """
        forward() for L poses of this ligand at once: params P (L, 6 + k), starting
        coordinates X0 (L, n, 3) and centers C (L, 3).
        """
        X = np.array(X0, dtype=float)
        _turn_torsions(X, np.ascontiguousarray(P[:, 6:], dtype=float),
                       self._bonds, self._flat_moving, self._starts)
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
            Y = np.matmul(Xt, dR[k].transpose(0, 2, 1))
            g[:, 3 + k] = (G.reshape(len(P), 1, -1) @ Y.reshape(len(P), -1, 1))[:, 0, 0]
        if self.n_torsions:
            out = np.empty((len(P), self.n_torsions))
            _torsion_gradient(np.ascontiguousarray(X), np.ascontiguousarray(G),
                              self._bonds, self._flat_moving, self._starts, out)
            g[:, 6:] = out
        return g
