"""
Ligand poses and their packing in Assembly placement, on nanocrystals and slabs.

A pose maps a spin angle (and, for multi-motif ligands, a choice among alternative
conformer fits) to ligand coordinates. Single-motif ligands spin about the normal of
their site; multi-motif ligands spin about the axis through their anchors when two
anchors leave it free, and are otherwise rigid. Packing (pack) turns the ligands that
clash, and bends multi-motif ones in torsion space, until every contact clears a cutoff.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import math
import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from .constellation import Candidate, ConstellationSampler, mean_normal, rotations_about_axis
from .core import BindingSite
from .ligand import Ligand, LigandSpec
from .refine import INTRA_SCALE, K_CONTACT, _intra_pairs, refine, refine_joint, stretch
from .utils.geometry import compute_bounding_spheres
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v


def place_about_normal(
    coords_loc: np.ndarray,
    plane: np.ndarray,
    site_pos: np.ndarray,
    anchor_offset: float,
    theta: float,
) -> np.ndarray:
    """
    Turn an oriented single-motif ligand (body along +z) onto a site normal, spin it by
    `theta` about that normal and put its anchor on the site, `anchor_offset` above it.
    """
    # Rotate to surface normal
    R_align = rotation_from_u_to_v(
        np.array([0.0, 0.0, 1.0], dtype=float),
        plane,
    )
    coords_aligned = coords_loc @ R_align.T

    # Rotate around surface normal
    R_rot = rotation_about_axis(plane, theta)
    coords_rot = coords_aligned @ R_rot.T

    anchor_offset = plane / np.linalg.norm(plane) * float(anchor_offset)
    anchor_pos = site_pos + anchor_offset

    # Translate to binding site position
    return coords_rot + anchor_pos


class SitePose:
    """
    Single-motif ligand on one site, free to spin about the site normal.
    """
    random_start = True
    spins = True
    n_variants = 1

    def __init__(self, ligand: Ligand, plane: np.ndarray, site_pos: np.ndarray):
        self.ligand = ligand
        self.plane = plane
        self.center = site_pos

    def coords(self, theta: float, variant: int = 0) -> np.ndarray:
        return place_about_normal(
            self.ligand.atoms.get_positions(),
            self.plane,
            self.center,
            getattr(self.ligand, "_anchor_offset", 0.0),
            theta,
        )


class FitPose:
    """
    Multi-motif ligand fitted to a constellation. Variants are coordinates of the
    candidate's alternative conformer fits, best first, followed by any refined ones;
    a stretched variant goes first. `origins` says where each came from ("fit",
    "refined", "unstuck", "stretched"); theta is the spin about the anchor axis relative
    to each.
    """
    random_start = False

    def __init__(self, ligand: Ligand, candidate: Candidate):
        self.ligand = ligand
        self.candidate = candidate
        lib = ligand.conformers()
        self.center = candidate.targets.mean(axis=0)
        self.axis = candidate.fits[0].axis
        self.spins = self.axis is not None
        self.variants = [f.coords(lib) for f in candidate.fits]
        self.origins = [f.origin for f in candidate.fits]
        self.variant = 0
        self.reach = 0.0
        for x in self.variants:
            self._extend_reach(x)

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    def _extend_reach(self, x: np.ndarray) -> None:
        # spinning about an axis through the pivot keeps every atom's distance to it
        self.reach = max(self.reach, float(np.linalg.norm(x - self.center, axis=1).max()))

    def coords(self, theta: float, variant: int = 0) -> np.ndarray:
        x = self.variants[variant]
        if not self.spins or theta == 0.0:
            return x.copy()
        R = rotations_about_axis(self.axis, np.array([theta]))[0]
        return (x - self.center) @ R.T + self.center

    def add_variant(self, coords: np.ndarray, origin: str = "refined", first: bool = False) -> int:
        k = 0 if first else len(self.variants)
        self.variants.insert(k, np.array(coords, dtype=float))
        self.origins.insert(k, origin)
        if first:
            self.variant += 1
        self._extend_reach(self.variants[k])
        return k


_WORKER_CORE = None


def _single_threaded_blas() -> None:
    """
    Run the OpenBLAS that scipy uses on one thread (Linux; silently nothing elsewhere).
    Every L-BFGS step calls into it, which wakes all its threads (64 here); in a pool of
    workers those threads starved each other, and adding workers made stretching slower.
    The copy bundled with numpy wheels is left alone: the refinement barely uses it, and
    resetting it in a forked worker cost more than it saved.
    """
    import ctypes
    try:
        paths = {line.split()[-1] for line in open("/proc/self/maps")
                 if "openblas" in line and "/numpy.libs/" not in line}
    except OSError:
        return
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for name in ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
                     "openblas_set_num_threads64_", "openblas_set_num_threads"):
            if hasattr(lib, name):
                getattr(lib, name)(1)


def _init_stretch_worker(core_points: np.ndarray) -> None:
    global _WORKER_CORE
    _single_threaded_blas()
    _WORKER_CORE = cKDTree(core_points)


def _stretch_job(job: tuple) -> Optional[np.ndarray]:
    model, coords, groups, targets, normal, contact = job
    res = stretch(model, coords, groups, targets, normal, [_WORKER_CORE], contact)
    return None if res is None else res.coords


def available_cpus() -> int:
    """CPUs this process may run on (a Slurm allocation or taskset included)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def stretch_poses(poses: Sequence, core_tree, cutoff: float, n_workers: int = 1) -> int:
    """
    Put a stretched copy of each multi-motif pose's best variant first: its dangling
    chains anti and raised along the mean normal of its sites, its anchors untouched.

    Stretching sees only the core, so the ligands are independent and, with `n_workers`
    above 1, go to that many processes (no more than the CPUs available); each builds the
    core tree once. The result does not depend on `n_workers`.

    Returns:
        How many poses now start stretched: they had chains to stretch and the result
        clears the core, or site selection had already stretched them.
    """
    fit_poses = [p for p in poses if isinstance(p, FitPose)]
    done = sum(p.origins[p.variant] == "stretched" for p in fit_poses)
    todo = [p for p in fit_poses
            if p.origins[p.variant] != "stretched" and p.ligand.torsion_model().tails.any()]
    jobs = [(p.ligand.torsion_model(), p.variants[0], p.ligand.binding_groups,
             p.candidate.targets, mean_normal(p.candidate.sites), cutoff + 0.2) for p in todo]
    n_workers = min(n_workers, available_cpus(), len(jobs))
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_stretch_worker,
                                 initargs=(core_tree.data,)) as pool:
            results = list(pool.map(_stretch_job, jobs,
                                    chunksize=max(1, len(jobs) // (4 * n_workers))))
    else:
        results = [None if (r := stretch(*job[:4], job[4], [core_tree], job[5])) is None else r.coords
                   for job in jobs]
    for pose, coords in zip(todo, results):
        if coords is not None and float(core_tree.query(coords)[0].min()) >= cutoff:
            pose.variant = pose.add_variant(coords, origin="stretched", first=True)
            done += 1
    return done


# Soft-sphere radii (cutoff + margin) tried in turn. A wide margin clears contacts with
# room to spare, but in a dense shell it also pushes on every contact that is already
# fine, and can leave a real one below the cutoff; the narrow one then acts on those alone.
MARGINS = (0.2, 0.05)
# Contact weights, as multiples of refine's default, that refine_clusters tries in turn when
# the margins fail: a ligand whose anchor springs hold it against the core sits where the
# two balance, which can be a hair below the cutoff (SB3-8 at 1.798 A, anchors 0.45 A off).
STIFFNESS = (1.0, 2.0, 4.0)


def refine_pose(
    pose: FitPose,
    coords: np.ndarray,
    neighbors_coords: List[np.ndarray],
    core_tree,
    tree_builder: Callable,
    cutoff: float,
    max_rmsd: float,
    min_distance: Callable,
) -> Optional[np.ndarray]:
    """
    Refine a multi-motif ligand in torsion space against the core and its neighbors.

    Returns the refined coordinates with the largest `min_distance(coords, neighbors)`,
    stopping at the first margin that clears `cutoff`, or None when every attempt leaves
    the anchors farther than `max_rmsd` (RMS, Å) from their sites.
    """
    trees = [core_tree]
    if neighbors_coords:
        trees.append(tree_builder(np.vstack(neighbors_coords))[0])
    best, best_d = None, -float("inf")
    for margin in MARGINS:
        res = refine(pose.ligand.torsion_model(), coords, pose.ligand.binding_groups,
                     pose.candidate.targets, trees, contact=cutoff + margin,
                     stop_at=cutoff + margin / 2, stop_rmsd=max_rmsd)
        if res.anchor_rmsd > max_rmsd:
            continue
        d = min_distance(res.coords, neighbors_coords)
        if d > best_d:
            best, best_d = res.coords, d
        if best_d >= cutoff:
            break
    return best


def intramolecular_contacts(poses: Sequence, coords_list: List[np.ndarray]) -> Tuple[float, int]:
    """
    Over the multi-motif ligands: the closest pair of atoms of one ligand four or more bonds
    apart (Å), and how many such pairs are closer than INTRA_SCALE x their summed radii.
    """
    closest, n = float("inf"), 0
    for pose, x in zip(poses, coords_list):
        if not isinstance(pose, FitPose):
            continue
        model = pose.ligand.torsion_model()
        n += len(_intra_pairs(model, x[None], INTRA_SCALE)[0])
        d = np.linalg.norm(x[:, None] - x[None], axis=-1)[np.triu(model.far)]
        closest = min(closest, float(d.min()) if d.size else closest)
    return closest, n


def bounding_spheres(poses: Sequence, coords_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Neighbor spheres of the ligands. A multi-motif ligand gets the sphere about its pivot
    that holds every spin and variant it may take, so that its neighbors stay known while
    it turns; a long ligand turning about its anchors sweeps far beyond its current sphere.
    """
    centers, radii = compute_bounding_spheres(coords_list)
    for i, pose in enumerate(poses):
        if isinstance(pose, FitPose):
            centers[i] = pose.center
            radii[i] = pose.reach
    return centers, radii


# Above this many atom pairs, _min_distance finds each neighbor atom's nearest atom in a tree
# and measures those pairs alone, with the same arithmetic: a long multi-motif ligand has
# nearly every other ligand within its sweep sphere, ~10^6 pairs on a large nanocrystal.
DENSE_PAIRS = 20000


def tree_min_distance(
    core_tree,
    tree_builder: Callable,
    cutoff: float,
    neighbors_coords: List[np.ndarray],
    early_core: bool = True,
) -> Callable:
    """
    The structures' _min_distance against one fixed set of neighbors, from a tree built
    once: a long multi-motif ligand has nearly every other ligand as a neighbor, and the
    rotation search evaluates it at every spin and variant. With `early_core`, like
    _min_distance, a core contact below `cutoff` is returned without looking at the
    ligands; without it, the result is the closest contact of either kind.
    """
    tree = tree_builder(np.vstack(neighbors_coords))[0] if neighbors_coords else None

    def min_distance(coords: np.ndarray, _neighbors=None) -> float:
        d_core = float(core_tree.query(coords)[0].min()) if core_tree is not None else float("inf")
        if (early_core and d_core < cutoff) or tree is None:
            return d_core
        return min(d_core, float(tree.query(coords, k=1)[0].min()))

    return min_distance


def optimize_pose(
    pose,
    neighbors_coords: List[np.ndarray],
    min_distance: Callable,
    rng: random.Random,
    cutoff: float,
    coarse_step_deg: float,
    fine_step_deg: float,
    window_deg: float,
) -> Tuple[float, int, np.ndarray]:
    """
    Search the spin (and the fit variant) that maximizes the minimum distance to the core
    and to neighboring ligands.

    Variants are scanned in order and the search stops at the first one whose coarse scan
    clears `cutoff`, so better-fitting conformers win whenever they are clash-free. The
    fine scan refines the best angle of the best variant.

    Returns:
        (theta, variant, coords) of the best pose found.
    """
    best_theta = None
    best_variant = 0
    best_min_d = -float("inf")
    best_coords = None
    theta0 = 2.0 * math.pi * rng.random()

    # Coarse search
    n_coarse = max(int(round(360.0 / coarse_step_deg)), 1) if pose.spins else 1

    for v in range(pose.n_variants):
        for k in range(n_coarse):
            theta = theta0 + 2.0 * math.pi * (k / n_coarse)
            theta = theta % (2.0 * math.pi)

            coords = pose.coords(theta, v)
            min_d = min_distance(coords, neighbors_coords)

            if min_d > best_min_d:
                best_theta, best_variant, best_min_d, best_coords = theta, v, min_d, coords

        if best_min_d >= cutoff:
            return best_theta, best_variant, best_coords

    if not pose.spins:
        return best_theta, best_variant, best_coords

    # Fine search
    window_rad = math.radians(window_deg)
    fine_step_rad = math.radians(fine_step_deg)

    # the window follows best_theta as it improves, as the original search did
    n_fine = int(round(2.0 * window_rad / fine_step_rad)) + 1
    for k in range(n_fine):
        delta = -window_rad + k * (2.0 * window_rad / (n_fine - 1))
        theta = (best_theta + delta) % (2.0 * math.pi)

        coords = pose.coords(theta, best_variant)
        min_d = min_distance(coords, neighbors_coords)

        if min_d > best_min_d:
            best_theta, best_min_d, best_coords = theta, min_d, coords

    return best_theta, best_variant, best_coords


def refine_clusters(
    poses: Sequence,
    coords_list: List[np.ndarray],
    core_tree,
    tree_builder: Callable,
    mic: Callable,
    min_distance: Callable,
    cutoff: float,
    max_rmsd: float,
    margin: float = 2.0,
    stop_at: Optional[float] = None,
) -> bool:
    """
    Refine jammed multi-motif ligands together, for when no ligand can clear its contacts
    alone.

    Ligands closer than `cutoff` form a contact graph; in every connected component with a
    contact below the cutoff (a single ligand touching the core included), the multi-motif
    members are refined jointly against the core and all other ligands, and pushed apart
    from each other wherever they come within `cutoff` + margin (see MARGINS, then with
    stiffer contacts, STIFFNESS). Members whose
    anchors end farther than `max_rmsd` from their sites keep their poses (one in a shell
    of 89 lecithins would otherwise throw the whole shell away); the others take theirs if
    the closest contact of the component improves. Each refinement stops once every contact
    is `stop_at` Å (by default halfway into the margin). Updates `coords_list` and the poses
    in place.

    Returns:
        Whether any component improved.
    """
    n = len(poses)
    centers, radii = compute_bounding_spheres(coords_list)
    close: Dict[Tuple[int, int], float] = {}
    for a in range(n):
        for b in range(a + 1, n):
            if np.linalg.norm(mic(centers[a] - centers[b])) > radii[a] + radii[b] + cutoff + margin:
                continue
            d = np.linalg.norm(mic(coords_list[a][:, None] - coords_list[b][None]), axis=-1).min()
            if d < cutoff + margin:
                close[(a, b)] = float(d)

    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (a, b), d in close.items():
        if d < cutoff:
            parent[root(a)] = root(b)
    components: Dict[int, List[int]] = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)

    def closest(coords, i):
        return min_distance(coords[i], [c for j, c in enumerate(coords) if j != i])

    improved = False
    for comp in components.values():
        members = [i for i in comp if isinstance(poses[i], FitPose)]
        if not members:
            continue
        before = min(closest(coords_list, i) for i in members)
        if before >= cutoff:
            continue  # nothing to clear: a lone ligand with no contact below the cutoff
        others = [c for j, c in enumerate(coords_list) if j not in members]
        trees = [core_tree, tree_builder(np.vstack(others))[0] if others else None]
        for stiff, margin in [(k, m) for k in STIFFNESS for m in MARGINS]:
            results = refine_joint(
                [poses[i].ligand.torsion_model() for i in members],
                [coords_list[i] for i in members],
                [poses[i].ligand.binding_groups for i in members],
                [poses[i].candidate.targets for i in members],
                trees, cutoff + margin, tree_builder, k_contact=K_CONTACT * stiff,
                stop_at=cutoff + margin / 2 if stop_at is None else stop_at, stop_rmsd=max_rmsd,
            )
            moved = [(i, r) for i, r in zip(members, results) if r.anchor_rmsd <= max_rmsd]
            if not moved:
                continue
            trial = list(coords_list)
            for i, r in moved:
                trial[i] = r.coords
            after = min(closest(trial, i) for i in members)
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
    following the core's periodicity: `tree_builder(points)` gives a tree of the points and
    their periodic images with the point each image comes from, `mic` and `wrap` take
    vectors and points to their minimum image and into the box, and
    `neighbor_map(centers, radii, cutoff)` lists for every ligand those whose spheres come
    within `cutoff` of its own.
    """

    def __init__(self, core_tree, tree_builder: Callable, mic: Callable, wrap: Callable,
                 neighbor_map: Callable, cutoff: float):
        self.core_tree = core_tree
        self.tree_builder = tree_builder
        self.mic = mic
        self.wrap = wrap
        self.neighbor_map = neighbor_map
        self.cutoff = cutoff

    def closest(self, coords: np.ndarray, neighbors_coords: Optional[List[np.ndarray]]) -> float:
        """
        Closest contact of `coords` with the core, and with `neighbors_coords` unless the
        core is already closer than the cutoff.
        """
        if self.core_tree is not None:
            d_core = float(np.min(self.core_tree.query(coords)[0]))
        else:
            d_core = float("inf")
        if d_core < self.cutoff or not neighbors_coords:
            return d_core

        neighbors = np.vstack(neighbors_coords)  # (N,3)
        if len(coords) * len(neighbors) > DENSE_PAIRS:
            # wrapped into the box, periodic images one box away hold every minimum image
            tree, image_of = self.tree_builder(self.wrap(coords))
            _, k = tree.query(self.wrap(neighbors))
            diff = self.mic(coords[image_of[k]] - neighbors)
        else:
            diff = self.mic(coords[:, None, :] - neighbors[None, :, :])  # (M,N,3)
        d2 = np.sum(diff * diff, axis=-1)
        return min(d_core, float(np.sqrt(d2.min())))

    def exact(self, coords: np.ndarray, neighbors_coords: List[np.ndarray]) -> float:
        """
        Closest contact of `coords` with the core or any of `neighbors_coords`.
        """
        return tree_min_distance(self.core_tree, self.tree_builder, self.cutoff,
                                 neighbors_coords, early_core=False)(coords)

    def neighbors(self, poses: Sequence, coords_list: List[np.ndarray]) -> Dict[int, List[int]]:
        """
        Ligands within reach of each ligand, over the spheres of bounding_spheres.
        """
        centers, radii = bounding_spheres(poses, coords_list)
        return self.neighbor_map(centers, radii, self.cutoff)

    def conflicts(
        self,
        coords_list: List[np.ndarray],
        neighbor_map: Dict[int, List[int]],
    ) -> Tuple[float, List[int], np.ndarray]:
        """
        Closest contact of every ligand with the core and its neighbors (closest).

        Returns:
            (overall closest contact, ligands closer than the cutoff, contact per ligand)
        """
        n_lig = len(coords_list)
        conflict_ligs = []
        dists = np.empty(n_lig, dtype=float)
        global_min = float("inf")
        for i in range(n_lig):
            idxs = neighbor_map.get(i, [])
            neighbors_coords = [coords_list[j] for j in idxs] if idxs else None
            d_i = self.closest(coords_list[i], neighbors_coords)
            dists[i] = d_i
            global_min = min(global_min, d_i)
            if d_i < self.cutoff:
                conflict_ligs.append(i)
        return global_min, conflict_ligs, dists


def active_cluster(conflict_ligs: List[int], site_positions: np.ndarray, radius: float) -> List[int]:
    """
    The ligands in conflict and every ligand whose site lies within `radius` of a site
    already in the cluster, grown until no more join.
    """
    if not conflict_ligs:
        return []
    site_tree = cKDTree(site_positions)
    active = set(conflict_ligs)
    queue = list(conflict_ligs)
    visited = set(conflict_ligs)
    while queue:
        i = queue.pop()
        for j in site_tree.query_ball_point(site_positions[i], r=radius):
            if j in visited:
                continue
            visited.add(j)
            active.add(j)
            queue.append(j)
    return sorted(list(active))


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

    The dangling chains of multi-motif ligands are first stretched (`stretch_tails`, in
    `n_workers` processes) and, if any were, relaxed together. Then, up to `max_iters`
    times, the ligands in conflict and those whose sites lie within `active_radius` of
    theirs turn one by one, in random order, to the spin (and fit variant) that clears their
    neighbors best (optimize_pose: a scan every `coarse_step_deg`, then every
    `fine_step_deg` within `window_deg` of the best). With `refine_torsions`, a multi-motif
    ligand that turning cannot clear is refined in torsion space, and ligands that none can
    clear alone are refined together (refine_clusters); anchors may end `max_rmsd` Å
    (RMS) from their sites.

    Returns:
        The coordinates of every ligand.

    Raises:
        RuntimeError: when an iteration moves no ligand, or the cutoff is still not met
        after `max_iters`.
    """
    cutoff = contacts.cutoff
    # Where each ligand binds: its site, or the centroid of its sites
    site_positions = np.array([p.center for p in poses])

    # stand up multi-motif tails first: coiled conformers tangle with their neighbors
    n_stretched = 0
    if stretch_tails:
        n_stretched = stretch_poses(poses, contacts.core_tree, cutoff, n_workers)
        if n_stretched:
            n_multi = sum(isinstance(p, FitPose) for p in poses)
            log(f"[Log] Stretched the tails of {n_stretched} of {n_multi} multi-motif ligands.")

    # starting poses; multi-motif ligands keep the spin of their fit
    thetas = np.array([2.0 * math.pi * rng.random() if p.random_start else 0.0
                       for p in poses], dtype=float)
    coords_list: List[np.ndarray] = [p.coords(t, getattr(p, "variant", 0))
                                     for p, t in zip(poses, thetas)]

    # relax the stretched shell together, short of the cutoff; single turns clear the rest
    if n_stretched and refine_torsions:
        refine_clusters(poses, coords_list, contacts.core_tree, contacts.tree_builder, contacts.mic,
                        contacts.exact, cutoff, max_rmsd, stop_at=cutoff - 0.1)

    # Rotation optimization loop
    neighbor_map = contacts.neighbors(poses, coords_list)
    global_min, conflict_ligs, dists = contacts.conflicts(coords_list, neighbor_map)
    log(f"[Log] Initial global_min = {global_min:.3f} Å")

    for iter in range(1, max_iters + 1):
        if global_min >= cutoff:
            log(f"[Log] Hard cutoff {cutoff:.3f} Å satisfied. Stopping optimization.")
            break

        active = active_cluster(conflict_ligs, site_positions, active_radius)

        improved = False
        rng.shuffle(active)

        for i in tqdm(active, desc=f"Optimizing ligand (iter {iter})", disable=not verbose):
            neighbor_idx = neighbor_map.get(i, [])
            neighbors_coords = [coords_list[j] for j in neighbor_idx]

            # multi-motif ligands need exact contacts, neighbors included even near the core
            multi = isinstance(poses[i], FitPose)
            min_distance = contacts.exact if multi else contacts.closest
            d_old = min_distance(coords_list[i], neighbors_coords) if multi else dists[i]

            search = contacts.closest
            if multi:
                search = tree_min_distance(contacts.core_tree, contacts.tree_builder, cutoff,
                                           neighbors_coords, early_core=False)
            best_theta, best_variant, best_coords = optimize_pose(
                poses[i], neighbors_coords, search, rng, cutoff,
                coarse_step_deg, fine_step_deg, window_deg,
            )
            d_new = min_distance(best_coords, neighbors_coords)

            # if turning fails, a multi-motif ligand may bend, against every other ligand
            if refine_torsions and multi and d_new < cutoff:
                others = [c for j, c in enumerate(coords_list) if j != i]
                refined = refine_pose(poses[i], best_coords, others, contacts.core_tree,
                                      contacts.tree_builder, cutoff, max_rmsd, min_distance)
                if refined is not None:
                    d_ref = min_distance(refined, others)
                    if d_ref > d_new:
                        best_theta, best_variant = 0.0, poses[i].add_variant(refined)
                        best_coords, d_new = refined, d_ref

            if d_new > d_old + 1e-3:
                coords_list[i] = best_coords
                thetas[i] = best_theta
                poses[i].variant = best_variant
                improved = True
                if multi:
                    # a refined variant may reach farther: let the ligands still to move see it
                    neighbor_map = contacts.neighbors(poses, coords_list)

        if not improved and refine_torsions:
            # nobody could move alone: let jammed multi-motif ligands give way together
            improved = refine_clusters(
                poses, coords_list, contacts.core_tree, contacts.tree_builder, contacts.mic,
                contacts.exact, cutoff, max_rmsd,
            )

        if not improved:
            raise RuntimeError("[Log] No improvement in active cluster. Stopping.")

        neighbor_map = contacts.neighbors(poses, coords_list)
        global_min, conflict_ligs, dists = contacts.conflicts(coords_list, neighbor_map)
        log(f"[Log] Iter {iter}  global_min = {global_min:.3f} Å")

    if global_min < cutoff:
        raise RuntimeError("[Log] Maximum iterations reached without satisfying overlap cutoff.")
    return coords_list


def motif_pools(
    spec: LigandSpec,
    binding_sites: List[BindingSite],
    symbol_map: Dict[str, str],
) -> Tuple[List[List[BindingSite]], Dict[str, List[BindingSite]]]:
    """
    Candidate sites of every motif, and the sites of every site type the motifs need.
    """
    by_type = {t: [s for s in binding_sites if s.symbol == symbol_map[t]]
               for t in {m.site for m in spec.ligand.binding_motifs}}
    return [by_type[m.site] for m in spec.ligand.binding_motifs], by_type


def select_constellations(
    spec: LigandSpec,
    pools: List[List[BindingSite]],
    by_type: Dict[str, List[BindingSite]],
    sampler: ConstellationSampler,
    log: Callable = print,
    refine: bool = True,
) -> List[Tuple[Ligand, Candidate]]:
    """
    Choose the constellations of one multi-motif LigandSpec and clone a ligand for each.

    With ``spec.binding_sites`` every entry lists one site atom index per motif, in motif
    order. Otherwise ``spec.coverage`` is a fraction of the most ligands the free sites
    can host (or, above 1, a count), spread so that every face holds about the same
    fraction of its sites (see ConstellationSampler.sample), and seeds are tried until it
    is met or the surface is exhausted; a shortfall is reported with the reasons seeds were
    rejected. The first motif's sites set the per-face balance, so it is best given the
    scarcer site type. With `refine`,
    constellations whose fits all clash with the core get a torsion-space refinement first.
    """
    lig = spec.ligand
    lib = lig.conformers()
    replace = not spec.adsorb
    offset = float(spec.anchor_offset)
    model = lig.torsion_model() if refine else None
    # the frame's van der Waals radii shadow the sites under it (ConstellationSampler.accept);
    # chains and rings that torsions turn can still get out of a neighbor's way
    tm = lig.torsion_model()
    radii = np.where(tm.frame, tm.radii, 0.0)
    groups = lig.binding_groups
    shadowed = sampler.n_shadowed
    chosen: List[Candidate] = []

    if spec.binding_sites is not None:
        for group in spec.binding_sites:
            group = list(group) if isinstance(group, (list, tuple)) else [group]
            if len(group) != lig.n_motifs:
                print(f"[Warning] Binding site group {group} for ligand {lig.name} needs one "
                      f"site index per motif ({lig.n_motifs}). Skipped.")
                continue
            sites = [next((s for s in pool if idx in s.atom_indices and not s.passivated), None)
                     for idx, pool in zip(group, pools)]
            if any(s is None for s in sites) or len({id(s) for s in sites}) < len(sites):
                print(f"[Warning] Binding site group {group} is not available for ligand "
                      f"{lig.name}. Skipped.")
                continue
            cand, reason = sampler.fit_sites(lib, sites, offset, replace, model, groups, lig.mol,
                                             radii)
            if cand is None:
                print(f"[Warning] Binding site group {group} rejected for ligand {lig.name}: {reason}.")
                continue
            sampler.accept(cand, lib, replace, radii)
            chosen.append(cand)
    else:
        # each site type hosts as many ligands as it has free sites per motif needing it
        need = Counter(m.site for m in lig.binding_motifs)
        capacity = min(sum(not s.passivated for s in by_type[t]) // k for t, k in need.items())
        if 0.0 < spec.coverage <= 1.0:
            n_target = int(math.ceil(spec.coverage * capacity))
        else:
            n_target = int(spec.coverage)
            if n_target > capacity:
                print(f"[Warning] Requested {n_target} ligands {lig.name}, but the free sites "
                      f"host at most {capacity}.")
                n_target = capacity
        chosen, failures = sampler.sample(lib, pools, n_target, offset, replace, model, groups,
                                          lig.mol, radii)
        if len(sampler.occupancy) > 1:
            occ = list(sampler.occupancy.values())
            log(f"[Log] {lig.name} occupancy per face: {min(occ):.2f}-{max(occ):.2f} "
                f"over {len(occ)} faces.")
        if len(chosen) < n_target:
            reasons = ", ".join(f"{r} ({n}x)" for r, n in failures.most_common())
            print(f"[Warning] Placed {len(chosen)} of {n_target} requested {lig.name} ligands; "
                  f"rejected seeds: {reasons or 'none left to try'}.")

    mode = "Adsorbed" if spec.adsorb else "Placed"
    log(f"[Log] {mode} total {len(chosen)} {lig.name} ligands with {lig.n_motifs} motifs "
        f"({'/'.join(m.site for m in lig.binding_motifs)} sites); {sampler.n_shadowed - shadowed} "
        f"free sites under them taken out of use.")

    out = []
    for cand in chosen:
        clone = lig.clone()
        clone.planes = [s.plane for s in cand.sites]
        clone._anchor_offset = offset
        out.append((clone, cand))
    return out
