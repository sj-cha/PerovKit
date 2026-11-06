from __future__ import annotations
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from ..kernels import _best_spin, _closest, make_cells
from ..refine import CONTACT_MARGINS, INTRA_SCALE, K_CONTACT, refine, refine_joint
from ..utils.box import Box
from .poses import MultiMotifPose, Pose
from .stretch import stretch_poses


# Å beyond the cutoff: contacts between ligands are searched this far, farther ones never matter
CONTACT_REACH = 1.0


def bend_pose(
    pose: Pose,
    coords: np.ndarray,
    neighbors_coords: List[np.ndarray],
    contacts: Contacts,
    max_rmsd: float,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Refine a ligand in torsion space against the core and `neighbors_coords`; one with a
    spin axis (a single-motif ligand) keeps its tilt and turns only about it.

    Returns the refined coordinates whose closest contact is farthest, and that contact,
    stopping at the first margin that clears the cutoff; (None, -inf) when every attempt
    leaves the anchors farther than `max_rmsd` (RMS, Å) from their sites.
    """
    cutoff = contacts.cutoff
    around = contacts.around(neighbors_coords)
    fixed = [contacts.core_cells[0], around[0]]
    best, best_d = None, -float("inf")
    for margin in CONTACT_MARGINS:
        res = refine(pose.ligand.torsion_model(), coords, pose.ligand.binding_groups,
                     pose.targets, fixed, contact_distance=cutoff + margin, spin_axis=pose.spin_axis,
                     stop_at=cutoff + margin / 2, stop_rmsd=max_rmsd, interior=contacts.interior)
        if res.anchor_rmsd > max_rmsd:
            continue
        d = _closest(res.coords, contacts.core_cells, around, contacts.reach)
        if d > best_d:
            best, best_d = res.coords, d
        if best_d >= cutoff:
            break
    return best, best_d


def intramolecular_contacts(poses: Sequence, coords_list: List[np.ndarray]) -> Tuple[float, int]:
    """
    Over the multi-motif ligands: the closest pair of atoms of one ligand four or more bonds
    apart (Å), and how many such pairs are closer than INTRA_SCALE x their summed radii.
    """
    min_distance, n = float("inf"), 0
    for pose, x in zip(poses, coords_list):
        if not isinstance(pose, MultiMotifPose):
            continue
        model = pose.ligand.torsion_model()
        a, b = np.nonzero(np.triu(model.far))
        d = np.linalg.norm(x[a] - x[b], axis=1)
        n += int(np.sum((d < INTRA_SCALE * (model.radii[a] + model.radii[b])) & (d > 1e-9)))
        min_distance = min(min_distance, float(d.min()) if d.size else min_distance)
    return min_distance, n


def bounding_spheres(poses: Sequence, coords_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Neighbor spheres of the ligands: center (mean) and radius of the atoms. A multi-motif
    ligand gets the sphere about its pivot that holds every spin and variant it may take,
    so that its neighbors stay known while it turns; a long ligand turning about its anchors
    sweeps far beyond its current sphere.
    """
    centers = np.array([c.mean(axis=0) for c in coords_list])
    radii = np.array([np.linalg.norm(c - m, axis=1).max() for c, m in zip(coords_list, centers)])
    for i, pose in enumerate(poses):
        if isinstance(pose, MultiMotifPose):
            centers[i] = pose.center
            radii[i] = pose.reach
    return centers, radii


def turn_pose(
    pose,
    contacts: Contacts,
    neighbors,
    rng: random.Random,
    coarse_step_deg: float,
    fine_step_deg: float,
    window_deg: float,
) -> Tuple[int, np.ndarray]:
    """
    Search the spin (and the fit variant) that keeps the closest contact of the coordinates
    with the core and `neighbors` (cells) farthest.

    Variants are scanned in order and the search stops at the first one whose coarse scan
    clears the cutoff, so better-fitting conformers win whenever they are clash-free. The
    fine scan refines the best angle of the best variant.

    Returns:
        (variant, coords) of the best pose found.
    """
    cutoff, bound = contacts.cutoff, contacts.reach
    theta0 = 2.0 * math.pi * rng.random()
    if pose.can_spin:
        axis = np.asarray(pose.axis, dtype=float) / np.linalg.norm(pose.axis)
        n_coarse = max(int(round(360.0 / coarse_step_deg)), 1)
        spins, spin0 = 2.0 * math.pi * np.arange(n_coarse) / n_coarse, theta0
    else:
        axis, spins, spin0 = np.zeros(3), np.zeros(1), 0.0
    center = np.asarray(pose.center, dtype=float)

    def scan(variant, thetas, reference, follow, best):
        return _best_spin(pose.variants[variant], center, axis, thetas, reference, follow, best,
                          contacts.core_cells, neighbors, bound)

    best_theta, best_variant, best_d = 0.0, 0, -float("inf")
    for v in range(pose.n_variants):
        theta, d = scan(v, spins, spin0, False, -1.0)
        if d > best_d:
            best_theta, best_variant, best_d = theta, v, d
        if best_d >= cutoff:
            return best_variant, pose.coords(best_theta, best_variant)

    if pose.can_spin:
        # the window follows the best angle as it improves
        window_rad = math.radians(window_deg)
        n_fine = int(round(2.0 * window_rad / math.radians(fine_step_deg))) + 1
        best_theta, _ = scan(best_variant, np.linspace(-window_rad, window_rad, n_fine),
                             best_theta, True, best_d)
    return best_variant, pose.coords(best_theta, best_variant)


def bend_clusters(
    poses: Sequence,
    coords_list: List[np.ndarray],
    contacts: Contacts,
    max_rmsd: float,
    stop_at: Optional[float] = None,
) -> bool:
    """
    Refine jammed ligands together, for when no ligand can clear its contacts alone.

    Ligands closer than the cutoff form a contact graph. In every connected component with a
    contact below the cutoff (a single ligand touching the core included), the members that
    can bend are refined jointly against the core and all other ligands and pushed apart
    from each other, with the margins of CONTACT_MARGINS and then stiffer contacts (2x and 4x).
    Members whose anchors end farther than `max_rmsd` from their sites keep their poses;
    the others take theirs if the closest contact of the component improves. Each
    refinement stops once every contact is `stop_at` Å (by default halfway into the
    margin). Updates `coords_list` and the poses in place.

    Returns:
        Whether any component improved.
    """
    cutoff = contacts.cutoff
    n = len(poses)
    # components do not touch, so what one finds here holds while the others move
    dists, a, b = contacts.closest(coords_list)
    graph = coo_matrix((np.ones(len(a), dtype=bool), (a, b)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)

    improved = False
    for component in range(n_components):
        members = [i for i in np.flatnonzero(labels == component).tolist() if poses[i].can_bend]
        if not members:
            continue
        before = float(dists[members].min())
        if before >= cutoff:
            continue  # nothing to clear: a lone ligand with no contact below the cutoff
        others = [c for j, c in enumerate(coords_list) if j not in members]
        fixed = [contacts.core_cells[0], contacts.around(others)[0]]
        for stiff, margin in [(k, m) for k in (1.0, 2.0, 4.0) for m in CONTACT_MARGINS]:
            results = refine_joint(
                [poses[i].ligand.torsion_model() for i in members],
                [coords_list[i] for i in members],
                [poses[i].ligand.binding_groups for i in members],
                [poses[i].targets for i in members],
                fixed, cutoff + margin, contacts.box.shifts, k_contact=K_CONTACT * stiff,
                stop_at=cutoff + margin / 2 if stop_at is None else stop_at, stop_rmsd=max_rmsd,
                spin_axes=[poses[i].spin_axis for i in members], interior=contacts.interior,
            )
            moved = [(i, r) for i, r in zip(members, results) if r.anchor_rmsd <= max_rmsd]
            if not moved:
                continue
            trial = list(coords_list)
            for i, r in moved:
                trial[i] = r.coords
            after = float(contacts.closest(trial, members)[0][members].min())
            if after > before + 1e-3:
                for i, r in moved:
                    coords_list[i] = r.coords
                    poses[i].variant = poses[i].add_variant(r.coords)
                improved = True
                break
    return improved


class Contacts:
    """
    Closest contacts of ligands with the core left after displacement and with each other,
    following the cell `box` of the core. `interior` holds the corners of the core without
    its outer `depth` Å, along the axes where it has a surface: no place for ligand atoms.
    """

    def __init__(self, core_tree, box: Box, cutoff: float, depth: float):
        self.core_tree = core_tree
        self.box = box
        self.cutoff = cutoff
        x = np.array(core_tree.data)
        lo, hi = x.min(axis=0) + depth, x.max(axis=0) - depth
        lo[box.axes], hi[box.axes] = -np.inf, np.inf
        self.interior = (lo, hi)
        self.reach = cutoff + 0.5   # contacts farther than this make no difference
        self.core_cells = make_cells(x, self.reach)

    def closest(self, coords_list: List[np.ndarray],
                ligands: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Closest contact with the core and the other ligands of every ligand, or only of
        `ligands`, from one search over all atoms. Contacts with ligands count up to
        CONTACT_REACH beyond the cutoff.

        Returns:
            (contact per ligand, inf where not asked for; both ligands of every pair closer
            than the cutoff)
        """
        owner = np.repeat(np.arange(len(coords_list)), [len(c) for c in coords_list])
        x = np.vstack(coords_list)
        own = np.arange(len(x)) if ligands is None else np.flatnonzero(np.isin(owner, ligands))
        tree, image_of = self.box.tree(x)
        found = cKDTree(x[own]).sparse_distance_matrix(tree, self.cutoff + CONTACT_REACH,
                                                       output_type="ndarray")
        a, b = owner[own[found["i"]]], owner[image_of[found["j"]]]
        other = a != b
        a, b, d = a[other], b[other], found["v"][other]
        dists = np.full(len(coords_list), np.inf)
        np.minimum.at(dists, owner[own], self.core_tree.query(x[own])[0])
        np.minimum.at(dists, a, d)
        close = d < self.cutoff
        return dists, a[close], b[close]

    def around(self, neighbors_coords: List[np.ndarray]) -> tuple:
        """
        Cells of `neighbors_coords` and their images one cell away along the periodic axes.
        """
        x = np.vstack(neighbors_coords) if neighbors_coords else np.zeros((0, 3))
        return make_cells((x + self.box.shifts[:, None]).reshape(-1, 3), self.reach)

    def neighbors(self, i: int, centers: np.ndarray, radii: np.ndarray) -> List[int]:
        """
        Ligands within reach of ligand `i`, over the spheres of bounding_spheres.
        """
        d = np.linalg.norm(self.box.mic(centers - centers[i]), axis=1)
        within = d <= radii[i] + radii + self.cutoff
        within[i] = False
        return np.flatnonzero(within).tolist()

    def conflicts(self, coords_list: List[np.ndarray]) -> Tuple[float, List[int]]:
        """
        Returns:
            (overall closest contact, ligands closer than the cutoff to the core or a ligand)
        """
        dists = self.closest(coords_list)[0]
        return float(dists.min()), np.flatnonzero(dists < self.cutoff).tolist()


def active_cluster(conflict_ligs: List[int], site_positions: np.ndarray, radius: float,
                   box: Box) -> List[int]:
    """
    The ligands in conflict and those whose sites lie within `radius` of theirs (minimum
    images along the periodic axes of `box`).
    """
    tree, image_of = box.tree(site_positions)
    hits = tree.query_ball_point(site_positions[conflict_ligs], r=radius)
    return np.unique(image_of[np.concatenate(hits)]).tolist()


def pack(
    poses: Sequence,
    contacts: Contacts,
    rng: random.Random,
    max_iters: int,
    coarse_step_deg: float,
    fine_step_deg: float,
    window_deg: float,
    active_radius: float,
    max_rmsd: float,
    refine_torsions: bool,
    stretch_tails: bool,
    n_workers: int,
    log: Callable = print,
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Pack the ligands of `poses` until no atoms of two ligands, or of a ligand and the core,
    are closer than the cutoff of `contacts`.

    The dangling chains of the ligands are first stretched (`stretch_tails`, in `n_workers`
    processes) and, if any were, relaxed together. Then, up to `max_iters` times, the
    ligands in conflict and those whose sites lie within `active_radius` of theirs turn one
    by one, in random order, to the spin (and fit variant) that clears their neighbors best
    (turn_pose: a scan every `coarse_step_deg`, then every `fine_step_deg` within
    `window_deg` of the best). With `refine_torsions`, a ligand that turning cannot clear
    is refined in torsion space, and ligands that none can clear alone are refined together
    (bend_clusters); anchors may end `max_rmsd` Å (RMS) from their sites, and a
    single-motif ligand keeps its tilt and turns only about its normal.

    Returns:
        The coordinates of every ligand.

    Raises:
        RuntimeError: when an iteration moves no ligand, or the cutoff is still not met
        after `max_iters`.
    """
    cutoff = contacts.cutoff
    # Where each ligand binds: its site, or the centroid of its sites
    site_positions = np.array([p.center for p in poses])

    # stand up tails first: coiled conformers tangle with their neighbors
    n_stretched = 0
    if stretch_tails:
        n_stretched = stretch_poses(poses, contacts.core_tree, cutoff, n_workers)
        if n_stretched:
            log(f"[Log] Stretched the tails of {n_stretched} of {len(poses)} ligands.")

    # starting poses; multi-motif ligands keep the spin of their fit
    thetas = np.array([2.0 * math.pi * rng.random() if p.random_start else 0.0
                       for p in poses], dtype=float)
    coords_list: List[np.ndarray] = [p.coords(t, p.variant) for p, t in zip(poses, thetas)]

    # relax the stretched shell together, short of the cutoff; single turns clear the rest
    if n_stretched and refine_torsions:
        bend_clusters(poses, coords_list, contacts, max_rmsd, stop_at=cutoff - 0.1)

    # Rotation optimization loop
    centers, radii = bounding_spheres(poses, coords_list)
    global_min, conflict_ligs = contacts.conflicts(coords_list)
    log(f"[Log] Initial global_min = {global_min:.3f} Å")

    for iteration in range(1, max_iters + 1):
        if global_min >= cutoff:
            log(f"[Log] Hard cutoff {cutoff:.3f} Å satisfied. Stopping optimization.")
            break

        active = active_cluster(conflict_ligs, site_positions, active_radius, contacts.box)

        improved = False
        rng.shuffle(active)

        for i in tqdm(active, desc=f"Optimizing ligand (iter {iteration})", disable=not verbose):
            neighbors_coords = [coords_list[j] for j in contacts.neighbors(i, centers, radii)]

            around = contacts.around(neighbors_coords)
            bends = poses[i].can_bend
            d_old = _closest(coords_list[i], contacts.core_cells, around, contacts.reach)
            best_variant, best_coords = turn_pose(
                poses[i], contacts, around, rng, coarse_step_deg, fine_step_deg, window_deg,
            )
            d_new = _closest(best_coords, contacts.core_cells, around, contacts.reach)

            # if turning fails, the ligand may bend, against every other ligand
            if refine_torsions and bends and d_new < cutoff:
                others = [c for j, c in enumerate(coords_list) if j != i]
                refined, d_ref = bend_pose(poses[i], best_coords, others, contacts, max_rmsd)
                if refined is not None and d_ref > d_new:
                    best_variant, best_coords, d_new = poses[i].add_variant(refined), refined, d_ref

            if d_new > d_old + 1e-3:
                coords_list[i] = best_coords
                poses[i].variant = best_variant
                improved = True
                if bends:
                    # a refined variant may reach farther: let the ligands still to move see it
                    centers[[i]], radii[[i]] = bounding_spheres([poses[i]], [best_coords])

        if not improved and refine_torsions:
            # nobody could move alone: let jammed ligands give way together
            improved = bend_clusters(poses, coords_list, contacts, max_rmsd)

        if not improved:
            raise RuntimeError("[Log] No improvement in active cluster. Stopping.")

        centers, radii = bounding_spheres(poses, coords_list)
        global_min, conflict_ligs = contacts.conflicts(coords_list)
        log(f"[Log] Iter {iteration}  global_min = {global_min:.3f} Å")

    if global_min < cutoff:
        raise RuntimeError("[Log] Maximum iterations reached without satisfying overlap cutoff.")
    return coords_list
