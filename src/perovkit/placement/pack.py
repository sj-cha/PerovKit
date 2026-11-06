from __future__ import annotations
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from ..refine import CONTACT_MARGINS, INTRA_SCALE, K_CONTACT, intra_pairs, refine, refine_joint
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
    spin axis (a single-motif ligand) keeps its anchor and tilt and turns only about it.

    Returns the refined coordinates whose closest contact is farthest, and that contact,
    stopping at the first margin that clears the cutoff; (None, -inf) when every attempt
    leaves the anchors farther than `max_rmsd` (RMS, Å) from their sites.
    """
    cutoff = contacts.cutoff
    trees = [contacts.core_tree]
    if neighbors_coords:
        trees.append(contacts.box.tree(np.vstack(neighbors_coords))[0])
    best, best_d = None, -float("inf")
    for margin in CONTACT_MARGINS:
        res = refine(pose.ligand.torsion_model(), coords, pose.ligand.binding_groups,
                     pose.targets, trees, contact_distance=cutoff + margin, spin_axis=pose.spin_axis,
                     stop_at=cutoff + margin / 2, stop_rmsd=max_rmsd)
        if res.anchor_rmsd > max_rmsd:
            continue
        d = min(float(tree.query(res.coords)[0].min()) for tree in trees)
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
        n += len(intra_pairs(model, x[None], INTRA_SCALE)[0])
        d = np.linalg.norm(x[:, None] - x[None], axis=-1)[np.triu(model.far)]
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


def tree_min_distance(core_tree, box: Box, neighbors_coords: List[np.ndarray]) -> Callable:
    """
    Closest contact with the core or any of `neighbors_coords`, as a function of the
    coordinates, from a tree built once: the rotation search evaluates it at every spin and
    variant of a long ligand that has nearly every other ligand as a neighbor.
    """
    tree = box.tree(np.vstack(neighbors_coords))[0] if neighbors_coords else None

    def min_distance(coords: np.ndarray) -> float:
        d_core = float(core_tree.query(coords)[0].min())
        if tree is None:
            return d_core
        return min(d_core, float(tree.query(coords, k=1)[0].min()))

    return min_distance


def turn_pose(
    pose,
    min_distance: Callable,
    rng: random.Random,
    cutoff: float,
    coarse_step_deg: float,
    fine_step_deg: float,
    window_deg: float,
) -> Tuple[int, np.ndarray]:
    """
    Search the spin (and the fit variant) that maximizes `min_distance`, the closest contact
    of the coordinates with the core and the neighboring ligands.

    Variants are scanned in order and the search stops at the first one whose coarse scan
    clears `cutoff`, so better-fitting conformers win whenever they are clash-free. The
    fine scan refines the best angle of the best variant.

    Returns:
        (variant, coords) of the best pose found.
    """
    best_theta = None
    best_variant = 0
    best_min_d = -float("inf")
    best_coords = None
    theta0 = 2.0 * math.pi * rng.random()

    # Coarse search
    n_coarse = max(int(round(360.0 / coarse_step_deg)), 1) if pose.can_spin else 1

    for v in range(pose.n_variants):
        for k in range(n_coarse):
            theta = theta0 + 2.0 * math.pi * (k / n_coarse)
            theta = theta % (2.0 * math.pi)

            coords = pose.coords(theta, v)
            min_d = min_distance(coords)

            if min_d > best_min_d:
                best_theta, best_variant, best_min_d, best_coords = theta, v, min_d, coords

        if best_min_d >= cutoff:
            return best_variant, best_coords

    if not pose.can_spin:
        return best_variant, best_coords

    # Fine search
    window_rad = math.radians(window_deg)
    fine_step_rad = math.radians(fine_step_deg)

    # the window follows best_theta as it improves
    n_fine = int(round(2.0 * window_rad / fine_step_rad)) + 1
    for k in range(n_fine):
        delta = -window_rad + k * (2.0 * window_rad / (n_fine - 1))
        theta = (best_theta + delta) % (2.0 * math.pi)

        coords = pose.coords(theta, best_variant)
        min_d = min_distance(coords)

        if min_d > best_min_d:
            best_theta, best_min_d, best_coords = theta, min_d, coords

    return best_variant, best_coords


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
    cutoff, box = contacts.cutoff, contacts.box
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
        trees = [contacts.core_tree, box.tree(np.vstack(others))[0] if others else None]
        for stiff, margin in [(k, m) for k in (1.0, 2.0, 4.0) for m in CONTACT_MARGINS]:
            results = refine_joint(
                [poses[i].ligand.torsion_model() for i in members],
                [coords_list[i] for i in members],
                [poses[i].ligand.binding_groups for i in members],
                [poses[i].targets for i in members],
                trees, cutoff + margin, box, k_contact=K_CONTACT * stiff,
                stop_at=cutoff + margin / 2 if stop_at is None else stop_at, stop_rmsd=max_rmsd,
                spin_axes=[poses[i].spin_axis for i in members],
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
    following the cell `box` of the core.
    """

    def __init__(self, core_tree, box: Box, cutoff: float):
        self.core_tree = core_tree
        self.box = box
        self.cutoff = cutoff

    def quick(self, coords: np.ndarray, neighbors_coords: Optional[List[np.ndarray]]) -> float:
        """
        Closest contact of `coords` with the core, and with `neighbors_coords` unless the
        core is already closer than the cutoff.
        """
        d_core = float(np.min(self.core_tree.query(coords)[0]))
        if d_core < self.cutoff or not neighbors_coords:
            return d_core

        neighbors = np.vstack(neighbors_coords)  # (N,3)
        if len(coords) * len(neighbors) > 20000:
            # wrapped into the box, periodic images one box away hold every minimum image
            tree, image_of = self.box.tree(self.box.wrap(coords))
            _, k = tree.query(self.box.wrap(neighbors))
            diff = self.box.mic(coords[image_of[k]] - neighbors)
        else:
            diff = self.box.mic(coords[:, None, :] - neighbors[None, :, :])  # (M,N,3)
        d2 = np.sum(diff * diff, axis=-1)
        return min(d_core, float(np.sqrt(d2.min())))

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
    (bend_clusters); anchors may end `max_rmsd` Å (RMS) from their sites, while a
    single-motif ligand keeps its anchor and tilt and turns only about its normal.

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

            # ligands that may bend need exact contacts, neighbors included even near the core
            bends = poses[i].can_bend
            if bends:
                min_distance = tree_min_distance(contacts.core_tree, contacts.box, neighbors_coords)
            else:
                min_distance = lambda x: contacts.quick(x, neighbors_coords)
            d_old = min_distance(coords_list[i])
            best_variant, best_coords = turn_pose(
                poses[i], min_distance, rng, cutoff, coarse_step_deg, fine_step_deg, window_deg,
            )
            d_new = min_distance(best_coords)

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
