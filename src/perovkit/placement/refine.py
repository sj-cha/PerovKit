from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from ..torsion import TorsionModel, dihedral
from ..utils.rotation import rotation_about_axis
from .kernels import SKIN, System, _nearest, evaluate, forward, make_cells


CONTACT_MARGINS = (0.2, 0.05)  # Å, contacts are pushed to the cutoff plus each, in turn
INTRA_SCALE = 0.7              # atoms four or more bonds apart keep this x their summed vdW radii
K_CONTACT = 40.0               # per Å^2, soft-sphere weight of a contact


def _starts(lengths) -> np.ndarray:
    """Where runs of these `lengths` start, one after another, and where the last ends."""
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def build_system(
    models: Sequence[TorsionModel],
    coords: Sequence[np.ndarray],
    groups: Sequence[Sequence[Sequence[int]]],
    targets: Sequence[np.ndarray],
    frames: Sequence[Optional[np.ndarray]],
    fixed: Sequence[np.ndarray],
    contact_distance: float,
    k_contact: float,
    shifts: Optional[np.ndarray] = None,
    interior: Optional[tuple] = None,
    pull: Optional[tuple] = None,
    links: Optional[tuple] = None,
) -> System:
    """
    The arrays of L poses for `evaluate`. `fixed` are arrays of fixed atoms (periodic images
    included), `shifts` the periodic images of the poses, `pull`, for one pose, (direction,
    atom mask, k), and `links` the bonds being made between poses, (rows, scale, k) as
    System holds them, atoms numbered across the poses.
    """
    L = len(models)
    atom_start = _starts([len(c) for c in coords])
    rows, scale, k_link = (np.zeros((0, 4), dtype=np.int64), np.zeros(0), 0.0) if links is None else links
    types = {}
    for m in models:
        types.setdefault(id(m), m)
    index = {key: i for i, key in enumerate(types)}
    far = [m._far_pairs for m in types.values()]

    group_pose, group_lengths, group_atoms = [], [], []
    for l, gs in enumerate(groups):
        for g in gs:
            group_pose.append(l)
            group_lengths.append(len(g))
            group_atoms.append(np.asarray(g, dtype=np.int64) + atom_start[l])
    lo, hi = (np.zeros(3), np.zeros(3)) if interior is None else interior
    direction, mask, k_pull = (np.zeros(3), None, 0.0) if pull is None else pull
    return System(
        X0=np.concatenate(coords),
        atom_start=atom_start,
        param_start=_starts([6 + m.n_torsions for m in models]),
        torsion_start=_starts([m.n_torsions for m in models]),
        bonds=np.concatenate([m._bonds + o for m, o in zip(models, atom_start)]),
        moving_start=_starts(np.concatenate([m._n_moving for m in models])),
        moving=np.concatenate([m._flat_moving + o for m, o in zip(models, atom_start)]),
        center=np.array([t.mean(axis=0) for t in targets], dtype=float),
        frame=np.array([np.eye(3) if f is None else f for f in frames], dtype=float),
        framed=np.array([f is not None for f in frames]),
        group_pose=np.array(group_pose, dtype=np.int64),
        group_start=_starts(group_lengths),
        group_atoms=np.concatenate(group_atoms),
        targets=np.concatenate(targets),
        n_groups=np.array([len(gs) for gs in groups], dtype=np.int64),
        pose_model=np.array([index[id(m)] for m in models], dtype=np.int64),
        far_start=_starts([len(a) for a, _ in far]),
        far_a=np.concatenate([a for a, _ in far]),
        far_b=np.concatenate([b for _, b in far]),
        far_limit=np.concatenate([INTRA_SCALE * (m.radii[a] + m.radii[b])
                                  for m, (a, b) in zip(types.values(), far)]),
        owner=np.repeat(np.arange(L, dtype=np.int64), np.diff(atom_start)),
        fixed=make_cells(np.concatenate([np.zeros((0, 3)), *fixed]), contact_distance + SKIN),
        shifts=np.zeros((1, 3)) if shifts is None else shifts,
        contact_distance=contact_distance, k_anchor=20.0, k_contact=k_contact,
        w_torsion=0.02,   # of 1 - cos(change) per torsion, a mild pull to the conformer
        confined=interior is not None, lo=lo, hi=hi,
        pull_atoms=np.zeros(0, dtype=np.int64) if mask is None else np.flatnonzero(mask),
        pull_direction=np.asarray(direction, dtype=float), k_pull=k_pull,
        links=np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 4),
        link_scale=np.asarray(scale, dtype=float), k_link=float(k_link),
    )


@dataclass
class Refinement:
    coords: np.ndarray
    anchor_rmsd: float


def _anchor_rmsd(x, groups, targets) -> float:
    anchors = np.array([x[g].mean(axis=0) for g in groups])
    return float(np.sqrt(np.mean(np.sum((anchors - targets) ** 2, axis=1))))


def _spin_frame(axis: np.ndarray) -> np.ndarray:
    """
    Rotation matrix whose rows are two unit axes across `axis` and the unit `axis`: in it
    the last Euler angle turns about `axis` (see kernels.forward).
    """
    n = axis / np.linalg.norm(axis)
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(n, e1), n])


def _held(model: TorsionModel, spin_axis: Optional[np.ndarray]) -> np.ndarray:
    """
    Parameters held at zero for a ligand with a spin axis: the two Euler angles across the
    axis, so that it keeps its tilt; none without one.
    """
    held = np.zeros(6 + model.n_torsions, dtype=bool)
    if spin_axis is not None:
        held[3:5] = True
    return held


def _solve(system, held, max_iters, stop_at, stop_rmsd, stop_link=float("inf")) -> np.ndarray:
    """
    Minimize the objective of `system` by L-BFGS-B from zero with the `held` parameters kept
    there, stopping at the first iterate with every contact `stop_at` Å, nothing tangled, the
    anchors within `stop_rmsd` and every link point within `stop_link` of its partner.
    Returns the coordinates.
    """
    none = np.zeros(0, dtype=np.int64)
    unset = lambda: np.full((len(system.X0), 3), np.inf)    # every pose counts as moved
    lists = unset(), none, none, unset(), none, none, np.zeros(0), unset(), none, none, none
    good_enough = None

    def fun(params):
        nonlocal lists, good_enough
        E, g, min_distance, tangled, rmsd, link_miss, lists = evaluate(params, system, lists)
        good_enough = (params.copy() if stop_at is not None and min_distance >= stop_at
                       and not tangled and rmsd <= stop_rmsd and link_miss <= stop_link else None)
        return E, g

    def callback(intermediate_result):
        if good_enough is not None and np.array_equal(good_enough, intermediate_result.x):
            raise StopIteration

    bounds = [(0.0, 0.0) if h else (None, None) for h in held] if held.any() else None
    res = minimize(fun, np.zeros(len(held)), jac=True, method="L-BFGS-B", bounds=bounds,
                   callback=callback, options={"maxiter": max_iters})
    return forward(res.x, system)[0]


def refine(
    model: TorsionModel,
    coords: np.ndarray,
    groups: Sequence[Sequence[int]],
    targets: np.ndarray,
    fixed: Sequence[Optional[np.ndarray]],
    contact_distance: float,
    max_iters: int = 400,
    pull: Optional[tuple] = None,
    frozen: Optional[np.ndarray] = None,
    spin_axis: Optional[np.ndarray] = None,
    stop_at: Optional[float] = None,
    stop_rmsd: float = float("inf"),
    interior: Optional[tuple] = None,
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
        fixed: Arrays of fixed atoms to stay `contact_distance` Å away from (core, neighbors),
            periodic images included.
        contact_distance (float): Distance (Å) to which the fixed atoms are pushed out:
            closer pairs are penalized.
        max_iters (int): L-BFGS-B iterations.
        pull: Optional (direction, atom mask, k): lowers the energy by k per Å that the
            masked atoms move along the unit direction.
        frozen (np.ndarray): Optional (6 + k,) mask of parameters held at zero.
        spin_axis (np.ndarray): Optional surface normal: the ligand then keeps its tilt and
            turns only about it (the two Euler angles across it are held at zero).
        stop_at, stop_rmsd: Stop as soon as every contact with the fixed atoms is at least
            `stop_at` Å, no intramolecular pair is tangled and the anchors are within
            `stop_rmsd` (RMS, Å): the pose is good enough, and the remaining iterations
            would only polish it.
        interior (tuple): Optional corners (lo, hi) of a box that atoms are pushed out of,
            10x as hard as out of contacts: the free volume between the atoms of a core
            would otherwise take in any chain that a long step lands there.
    """
    frame = None if spin_axis is None else _spin_frame(spin_axis)
    system = build_system([model], [coords], [groups], [targets], [frame], fixed, contact_distance,
                          K_CONTACT, interior=interior, pull=pull)
    held = _held(model, spin_axis)
    if frozen is not None:
        held |= frozen
    x = _solve(system, held, max_iters, stop_at, stop_rmsd)
    return Refinement(coords=x, anchor_rmsd=_anchor_rmsd(x, groups, targets))


def stretch(
    model: TorsionModel,
    coords: np.ndarray,
    groups: Sequence[Sequence[int]],
    targets: np.ndarray,
    normal: np.ndarray,
    fixed: Sequence[np.ndarray],
    contact_distance: float,
    max_iters: int = 200,
    stop_at: Optional[float] = None,
) -> Optional[Refinement]:
    """
    Straighten the dangling chains of a ligand and raise them along `normal`.

    Every tail torsion is set anti (180°), from the anchors outward, or gauche (±60°) where
    anti would run the chain into the rest of the ligand or the fixed atoms; then only the
    tail torsions are optimized, so no anchor moves, to lift the tail's heavy atoms along
    `normal` while keeping clear of the `fixed` atoms (arrays) and of the ligand itself.
    With `stop_at` it stops as soon as every contact with the fixed atoms is that far, when
    the point is only to unstick a chain from the surface. Returns None for a ligand without
    tails.
    """
    if not model.tails.any():
        return None
    x = coords.copy()
    fixed = [f for f in fixed if len(f)]
    grids = [make_cells(f, contact_distance) for f in fixed]
    # tail torsions, anchors outward: anti, or gauche where anti hits the ligand or fixed atoms
    order = [b for b in range(model.n_torsions - 1, -1, -1) if model.tails[b]]
    reach = 2.0 * INTRA_SCALE * float(model.radii.max())
    for b in order:
        (i, j), mask, quad = model.bonds[b], model.masks[b], model.quads[b]
        u = x[j] - x[i]
        phi = dihedral(x[quad])
        moved, rest = np.flatnonzero(mask), np.flatnonzero(~mask)
        rest_tree = cKDTree(x[rest])
        best, best_overlap = None, np.inf
        for target in (np.pi, np.pi / 3, -np.pi / 3):
            R = rotation_about_axis(u, target - phi)
            y = (x[mask] - x[i]) @ R.T + x[i]
            d, k = rest_tree.query(y, k=min(8, len(rest)), distance_upper_bound=reach)
            other = rest[np.minimum(k, len(rest) - 1)]
            limit = INTRA_SCALE * (model.radii[moved][:, None] + model.radii[other])
            short = np.where(np.isfinite(d) & model.far[moved[:, None], other], limit - d, 0.0)
            overlap = float(np.sum(np.maximum(short, 0.0) ** 2))
            for grid in grids:
                d = _nearest(y, grid, contact_distance)
                overlap += float(np.sum((contact_distance - d[np.isfinite(d)]) ** 2))
            if overlap == 0.0:
                best = y
                break
            if overlap < best_overlap:
                best, best_overlap = y, overlap
        x[mask] = best
    lifted = model.masks[model.tails].any(axis=0) & model.heavy
    frozen = np.concatenate([np.ones(6, dtype=bool), ~model.tails])
    return refine(model, x, groups, targets, fixed, contact_distance,
                  max_iters=max_iters, pull=(normal, lifted, 1.0), frozen=frozen,
                  stop_at=stop_at)


def refine_joint(
    models: Sequence[TorsionModel],
    coords: Sequence[np.ndarray],
    groups: Sequence[Sequence[Sequence[int]]],
    targets: Sequence[np.ndarray],
    fixed: Sequence[Optional[np.ndarray]],
    contact_distance: float,
    shifts: np.ndarray,
    k_contact: float = K_CONTACT,
    max_iters: int = 400,
    stop_at: Optional[float] = None,
    stop_rmsd: float = float("inf"),
    spin_axes: Optional[Sequence[Optional[np.ndarray]]] = None,
    interior: Optional[tuple] = None,
    links: Optional[tuple] = None,
    stop_link: float = float("inf"),
    held_poses: Optional[Sequence[bool]] = None,
) -> List[Refinement]:
    """
    Refine several ligands at once, so that ligands jammed against each other can all give
    way. Each has the terms of `refine`; atoms of different ligands are also pushed
    `contact_distance` Å apart, their images at `shifts` included.
    `stop_at` and `stop_rmsd` end it early as in `refine`, with every ligand meeting them.
    `spin_axes` gives each ligand a spin axis as in `refine`, or None; `interior` is that
    of `refine`; `links` are bonds being made between them (see build_system), and
    `stop_link` how near their points must be for stop_at to end it. Ligands marked in
    `held_poses` stay put, there only for the links to them.
    """
    spin_axes = [None] * len(models) if spin_axes is None else list(spin_axes)
    system = build_system(models, coords, groups, targets,
                          [None if a is None else _spin_frame(a) for a in spin_axes],
                          fixed, contact_distance, k_contact, shifts=shifts, interior=interior, links=links)
    held_poses = [False] * len(models) if held_poses is None else held_poses
    held = np.concatenate([np.ones(6 + m.n_torsions, dtype=bool) if still else _held(m, a)
                           for m, a, still in zip(models, spin_axes, held_poses)])
    x = _solve(system, held, max_iters, stop_at, stop_rmsd, stop_link)
    out = []
    for l in range(len(models)):
        y = x[system.atom_start[l]:system.atom_start[l + 1]]
        out.append(Refinement(coords=y, anchor_rmsd=_anchor_rmsd(y, groups[l], targets[l])))
    return out
