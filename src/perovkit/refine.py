from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numba
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .torsion import TorsionModel, _dihedral
from .utils.rotation import rotation_about_axis


# Atoms of one ligand four or more bonds apart are kept INTRA_SCALE x the sum of their van
# der Waals radii apart (1.68 A for H...H, 2.38 for C...C); a larger scale jams the densest
# packings.
INTRA_SCALE = 0.7


K_CONTACT = 40.0   # per Å^2, soft-sphere weight of a contact


# Intramolecular overlaps weigh INTRA_WEIGHT x a contact, so that pulling the anchors onto
# their sites does not fold a ligand onto itself.
INTRA_WEIGHT = 10.0


def intra_pairs(model, X, scale, skin=0.0):
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
        pa, pb, _, limit = intra_pairs(self.model, X[poses], self.intra, SKIN)
        keep = ~stale[self.pa // n]
        self.pa = np.concatenate([self.pa[keep], atoms[pa]])
        self.pb = np.concatenate([self.pb[keep], atoms[pb]])
        self.limit = np.concatenate([self.limit[keep], limit])
        self.no_shift = np.zeros((len(self.pa), 3))


class _Mutual:
    """
    Verlet list of the contacts between atoms of different ligands, X (N, 3) with the
    atoms of ligand l at starts[l]:starts[l + 1], within `contact` + SKIN through the
    periodic images of `box`: pairs (p, q) with the image of q at X[q] + shift.
    A ligand's pairs are searched again once one of its atoms has moved SKIN / 4: the pairs
    of two ligands date from the later of their searches, so together they have moved at
    most 3/4 SKIN since.
    """

    def __init__(self, starts, contact, box):
        self.starts = np.asarray(starts, dtype=np.int64)
        self.contact, self.box = contact, box
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
        tree, image_of = self.box.tree(X)
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
        phi = _dihedral(x[quad])
        moved, rest = np.flatnonzero(mask), np.flatnonzero(~mask)
        rest_tree = cKDTree(x[rest])
        reach = 2.0 * INTRA_SCALE * float(model.radii.max())
        best, best_overlap = None, np.inf
        for target in (np.pi, np.pi / 3, -np.pi / 3):
            R = rotation_about_axis(u, target - phi)
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
    box,
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
    `contact` Å apart, listed through a tree from `box` (periodic images included).
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
    mutual = _Mutual(starts, contact, box)

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
