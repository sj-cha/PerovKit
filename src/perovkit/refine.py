from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

from ase.data import vdw_radii
import numba
import numpy as np
from rdkit import Chem
from scipy.optimize import minimize
from scipy.spatial import cKDTree

_ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

# Atoms of one ligand four or more bonds apart are kept INTRA_SCALE x the sum of their van
# der Waals radii apart (1.68 A for H...H, 2.38 for C...C); a larger scale jams the densest
# packings.
INTRA_SCALE = 0.7
K_CONTACT = 40.0   # per Å^2, soft-sphere weight of a contact
# Intramolecular overlaps weigh INTRA_WEIGHT x a contact, so that pulling the anchors onto
# their sites does not fold a ligand onto itself.
INTRA_WEIGHT = 10.0


def _intra_pairs(model, X, scale, skin=0.0):
    """
    Pairs (a, b) of atoms of one pose, four or more bonds apart, closer than scale x their
    summed radii (+ `skin`), in the flattened (L * n, 3) poses X, with those limits.
    """
    L, n, _ = X.shape
    flat = X.reshape(-1, 3)
    shifted = X + (np.arange(L) * 1.0e4)[:, None, None] * np.array([1.0, 0.0, 0.0])
    reach = 2.0 * scale * float(model.radii.max()) + skin
    close = cKDTree(shifted.reshape(-1, 3)).query_pairs(reach, output_type="ndarray")
    if not len(close):
        return close[:, 0], close[:, 1], np.zeros(0), np.zeros(0)
    pa, pb = close[:, 0], close[:, 1]
    keep = model.far[pa % n, pb % n]
    pa, pb = pa[keep], pb[keep]
    limit = scale * (model.radii[pa % n] + model.radii[pb % n])
    d = np.linalg.norm(flat[pa] - flat[pb], axis=1)
    hit = (d < limit + skin) & (d > 1e-9)
    return pa[hit], pb[hit], d[hit], limit[hit]

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


def _rotation(u: np.ndarray, phi: float) -> np.ndarray:
    """
    Rotation by `phi` about the unit axis `u` (Rodrigues).
    """
    x, y, z = float(u[0]), float(u[1]), float(u[2])
    c, s = math.cos(phi), math.sin(phi)
    t = 1.0 - c
    return np.array([[c + t * x * x, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, c + t * y * y, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, c + t * z * z]])


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
        radii (np.ndarray): (n_atoms,) van der Waals radii (Bondi, Å).
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


@dataclass
class Refinement:
    coords: np.ndarray
    anchor_rmsd: float


# Verlet lists: contacts are searched within their reach + SKIN, and searched again only for
# the ligands that have moved far enough since that every pair within reach stays listed.
SKIN = 1.0


@numba.njit(cache=True)
def _largest_steps(X, ref, starts, out):
    """out[b]: largest squared displacement from `ref` of the atoms starts[b]:starts[b + 1]."""
    for b in range(starts.shape[0] - 1):
        m = 0.0
        for a in range(starts[b], starts[b + 1]):
            dx = X[a, 0] - ref[a, 0]
            dy = X[a, 1] - ref[a, 1]
            dz = X[a, 2] - ref[a, 2]
            m = max(m, dx * dx + dy * dy + dz * dz)
        out[b] = m


@numba.njit(cache=True)
def _fixed_terms(x, atom, partner, contact, k, G):
    """Soft-sphere penalty of the listed (atom, fixed partner) pairs within `contact`; adds
    its gradient to G. Returns (E, closest contact)."""
    E = 0.0
    closest = np.inf
    for t in range(atom.shape[0]):
        a = atom[t]
        dx = x[a, 0] - partner[t, 0]
        dy = x[a, 1] - partner[t, 1]
        dz = x[a, 2] - partner[t, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d < contact and d > 1e-9:
            closest = min(closest, d)
            over = contact - d
            E += k * over * over
            c = -2.0 * k * over / d
            G[a, 0] += c * dx
            G[a, 1] += c * dy
            G[a, 2] += c * dz
    return E, closest


@numba.njit(cache=True)
def _pair_terms(x, pa, pb, shift, limit, k, G):
    """Soft-sphere penalty of the listed pairs of moving atoms, pa against the image of pb at
    x[pb] + shift, within their `limit`; adds its gradient to G. Returns (E, closest, hits)."""
    E = 0.0
    closest = np.inf
    hits = 0
    for t in range(pa.shape[0]):
        a, b = pa[t], pb[t]
        dx = x[a, 0] - x[b, 0] - shift[t, 0]
        dy = x[a, 1] - x[b, 1] - shift[t, 1]
        dz = x[a, 2] - x[b, 2] - shift[t, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d < limit[t] and d > 1e-9:
            hits += 1
            closest = min(closest, d)
            over = limit[t] - d
            E += k * over * over
            c = -2.0 * k * over / d
            G[a, 0] += c * dx
            G[a, 1] += c * dy
            G[a, 2] += c * dz
            G[b, 0] -= c * dx
            G[b, 1] -= c * dy
            G[b, 2] -= c * dz
    return E, closest, hits


class _Neighbors:
    """
    Verlet lists of L poses of one ligand type, X (L, n, 3): the fixed atoms of `trees`
    (their ``data`` holds the positions, periodic images included) within `contact` + SKIN
    of each atom, and the pairs of one pose four or more bonds apart within their limit
    + SKIN. A pose is searched again once one of its atoms has moved SKIN / 2.
    """

    def __init__(self, model, trees, contact, intra):
        self.model, self.contact, self.intra = model, contact, intra
        self.trees = [t for t in trees if t is not None and t.n > 0]
        self.ref = None

    def update(self, X):
        L, n, _ = X.shape
        if self.ref is None or self.ref.shape != X.shape:
            self.ref = X.copy()
            self.starts = np.arange(L + 1, dtype=np.int64) * n
            stale = np.ones(L, dtype=bool)
            self.atom = self.pa = self.pb = np.zeros(0, dtype=np.int64)
            self.partner, self.limit = np.zeros((0, 3)), np.zeros(0)
        else:
            moved = np.empty(L)
            _largest_steps(X.reshape(-1, 3), self.ref.reshape(-1, 3), self.starts, moved)
            stale = moved > (0.5 * SKIN) ** 2
            if not stale.any():
                return
            self.ref[stale] = X[stale]
        poses = np.flatnonzero(stale)
        atoms = (poses[:, None] * n + np.arange(n)).ravel()   # flat index of their atoms
        keep = ~stale[self.atom // n]
        found_atom, found_partner = [self.atom[keep]], [self.partner[keep]]
        if self.trees:
            own = cKDTree(X[poses].reshape(-1, 3))
            for tree in self.trees:
                m = own.sparse_distance_matrix(tree, self.contact + SKIN, output_type="ndarray")
                found_atom.append(atoms[m["i"]])
                found_partner.append(tree.data[m["j"]])
        self.atom, self.partner = np.concatenate(found_atom), np.concatenate(found_partner)
        pa, pb, _, limit = _intra_pairs(self.model, X[poses], self.intra, SKIN)
        keep = ~stale[self.pa // n]
        self.pa = np.concatenate([self.pa[keep], atoms[pa]])
        self.pb = np.concatenate([self.pb[keep], atoms[pb]])
        self.limit = np.concatenate([self.limit[keep], limit])
        self.no_shift = np.zeros((len(self.pa), 3))


class _Mutual:
    """
    Verlet list of the contacts between atoms of different ligands, X (N, 3) with the
    atoms of ligand l at starts[l]:starts[l + 1], within `contact` + SKIN through the
    periodic images of `tree_builder`: pairs (p, q) with the image of q at X[q] + shift.
    A ligand's pairs are searched again once one of its atoms has moved SKIN / 4: the pairs
    of two ligands date from the later of their searches, so together they have moved at
    most 3/4 SKIN since.
    """

    def __init__(self, starts, contact, tree_builder):
        self.starts = np.asarray(starts, dtype=np.int64)
        self.contact, self.tree_builder = contact, tree_builder
        self.owner = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
        self.ref = None
        self.p = self.q = np.zeros(0, dtype=np.int64)
        self.shift = np.zeros((0, 3))

    def update(self, X):
        if self.ref is None:
            self.ref = X.copy()
            stale = np.ones(len(self.starts) - 1, dtype=bool)
        else:
            moved = np.empty(len(self.starts) - 1)
            _largest_steps(X, self.ref, self.starts, moved)
            stale = moved > (0.25 * SKIN) ** 2
            if not stale.any():
                return
        atoms = np.flatnonzero(stale[self.owner])
        self.ref[atoms] = X[atoms]
        tree, image_of = self.tree_builder(X)
        m = cKDTree(X[atoms]).sparse_distance_matrix(tree, self.contact + SKIN, output_type="ndarray")
        p, j = atoms[m["i"]], m["j"].astype(np.int64)
        q = image_of[j]
        op, oq = self.owner[p], self.owner[q]
        # a pair of two searched ligands is found from both sides: keep it once
        found = (op != oq) & (~stale[oq] | (op < oq))
        keep = ~stale[self.owner[self.p]] & ~stale[self.owner[self.q]]
        self.p = np.concatenate([self.p[keep], p[found]])
        self.q = np.concatenate([self.q[keep], q[found]])
        self.shift = np.concatenate([self.shift[keep], tree.data[j[found]] - X[q[found]]])
        self.limit = np.full(len(self.p), self.contact)


def _terms(model, X, groups, T, nb, k_anchor, k_contact):
    """
    Anchor, contact and intramolecular penalties of L poses of one ligand type, X (L, n, 3)
    with anchor targets T (L, m, 3), and their Cartesian gradient, from the lists of `nb`
    (a _Neighbors). Returns (E, G (L, n, 3), closest, tangled, rmsd (L,)): the closest
    contact with the fixed atoms (inf if none within `contact`), whether any intramolecular
    pair is within its limit, and the anchor RMSD of each pose.
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
    nb.update(X)
    flat, G_flat = X.reshape(-1, 3), G.reshape(-1, 3)
    e, closest = _fixed_terms(flat, nb.atom, nb.partner, nb.contact, k_contact, G_flat)
    e_in, _, hits = _pair_terms(flat, nb.pa, nb.pb, nb.no_shift, nb.limit,
                                INTRA_WEIGHT * k_contact, G_flat)
    E += e + e_in
    tangled = hits > 0
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
    k_contact: float = K_CONTACT,
    intra: float = INTRA_SCALE,
    w_torsion: float = 0.02,
    max_iter: int = 400,
    pull: Optional[tuple] = None,
    frozen: Optional[np.ndarray] = None,
    stop_at: Optional[float] = None,
    stop_rmsd: float = float("inf"),
) -> Refinement:
    """
    Refine one ligand pose in torsion space: its rotatable torsions turn and the whole moves
    a little as a rigid body, so bond lengths and angles stay those of the starting pose,
    while soft-sphere penalties keep its anchors on their targets and its atoms out of
    contact (L-BFGS-B with analytic gradients). A geometry generator, not a force field: it
    clears clashes and says nothing about binding energies.

    Args:
        model (TorsionModel): Torsions and intramolecular pairs of the ligand.
        coords (np.ndarray): (n_atoms, 3) starting pose.
        groups: Binding atoms of each motif.
        targets (np.ndarray): (n_motifs, 3) anchor targets.
        trees: KD-trees of fixed atoms to stay `contact` Å away from (core, neighbors);
            their ``data`` must hold the positions, periodic images included.
        contact (float): Soft-sphere radius (Å) against the fixed atoms.
        k_anchor, k_contact (float): Weights (per Å^2) of anchor and contact penalties.
        intra (float): Ligand atoms four or more bonds apart are pushed this times their
            summed van der Waals radii apart (see INTRA_SCALE).
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
    nb = _Neighbors(model, trees, contact, intra)

    def fun(params):
        x, xt, dR = model.forward(params, coords, center)
        E, G, closest, tangled, rmsd = _terms(model, x[None], groups, targets[None], nb,
                                              k_anchor, k_contact)
        G = G[0]
        done["x"] = params.copy()
        done["ok"] = (stop_at is not None and closest >= stop_at and not tangled
                      and float(rmsd[0]) <= stop_rmsd)
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
    return Refinement(coords=x, anchor_rmsd=_anchor_rmsd(x, groups, targets))


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

    Every tail torsion is set anti (180°), from the anchors outward, or gauche (±60°) where
    anti would run the chain into the rest of the ligand or the fixed atoms; then only the
    tail torsions are optimized, so no anchor moves, to lift the tail's heavy atoms along
    `normal` while keeping clear of the fixed atoms in `trees` and of the ligand itself.
    With `stop_at` it stops as soon as every contact with the fixed atoms is that far, when
    the point is only to unstick a chain from the surface. Returns None for a ligand without
    tails.
    """
    if not model.tails.any():
        return None
    x = coords.copy()
    fixed = [t for t in trees if t is not None and t.n > 0]
    # tail torsions, anchors outward: anti, or gauche where anti hits the ligand or fixed atoms
    order = [b for b in range(model.n_torsions - 1, -1, -1) if model.tails[b]]
    for b in order:
        (i, j), mask, quad = model.bonds[b], model.masks[b], model.quads[b]
        u = x[j] - x[i]
        u /= np.linalg.norm(u)
        phi = _dihedral(x[quad])
        moved, rest = np.flatnonzero(mask), np.flatnonzero(~mask)
        rest_tree = cKDTree(x[rest])
        reach = 2.0 * INTRA_SCALE * float(model.radii.max())
        best, best_overlap = None, np.inf
        for target in (np.pi, np.pi / 3, -np.pi / 3):
            R = _rotation(u, target - phi)
            y = (x[mask] - x[i]) @ R.T + x[i]
            d, k = rest_tree.query(y, k=min(8, len(rest)), distance_upper_bound=reach)
            other = rest[np.minimum(k, len(rest) - 1)]
            limit = INTRA_SCALE * (model.radii[moved][:, None] + model.radii[other])
            short = np.where(np.isfinite(d) & model.far[moved[:, None], other], limit - d, 0.0)
            overlap = float(np.sum(np.maximum(short, 0.0) ** 2))
            for tree in fixed:
                d = tree.query(y, distance_upper_bound=contact)[0]
                overlap += float(np.sum((contact - d[np.isfinite(d)]) ** 2))
            if overlap == 0.0:
                best = y
                break
            if overlap < best_overlap:
                best, best_overlap = y, overlap
        x[mask] = best
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
    k_contact: float = K_CONTACT,
    intra: float = INTRA_SCALE,
    w_torsion: float = 0.02,
    max_iter: int = 400,
    stop_at: Optional[float] = None,
    stop_rmsd: float = float("inf"),
) -> List[Refinement]:
    """
    Refine several ligands at once, so that ligands jammed against each other can all give
    way. Each has the terms of `refine`; atoms of different ligands are also pushed
    `contact` Å apart, listed through a tree from `tree_builder` (periodic images included).
    `stop_at` and `stop_rmsd` end it early as in `refine`, with every ligand meeting them.
    """
    sizes = [6 + m.n_torsions for m in models]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    groups = [[np.asarray(g, dtype=int) for g in gs] for gs in groups]
    starts = np.concatenate([[0], np.cumsum([len(c) for c in coords])])
    done = {"ok": False, "x": None}

    # ligands of one type are computed together (forward_many and _terms)
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
            nb=_Neighbors(model, trees, contact, intra),
        ))
    mutual = _Mutual(starts, contact, tree_builder)

    def fun(P):
        E = 0.0
        closest, tangled, worst = float("inf"), False, 0.0
        X = np.empty((starts[-1], 3))
        state = []
        for b in batches:
            Pb = P[b["params"]]
            Xb, Xt, dR = b["model"].forward_many(Pb, b["X0"], b["C"])
            e, G, c, t, rmsd = _terms(b["model"], Xb, b["groups"], b["T"], b["nb"],
                                      k_anchor, k_contact)
            closest, tangled, worst = min(closest, c), tangled or t, max(worst, float(rmsd.max()))
            E += e + w_torsion * float(np.sum(1.0 - np.cos(Pb[:, 6:])))
            X[b["atoms"]] = Xb.reshape(-1, 3)
            state.append((Pb, Xb, Xt, dR, G))
        G_all = np.zeros_like(X)
        mutual.update(X)
        e, c, _ = _pair_terms(X, mutual.p, mutual.q, mutual.shift, mutual.limit, k_contact, G_all)
        E += e
        closest = min(closest, c)
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
        out.append(Refinement(coords=x, anchor_rmsd=_anchor_rmsd(x, groups[l], targets[l])))
    return out
