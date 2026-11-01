"""
Ligand poses and the rotation search shared by NanoCrystal and Slab placement.

A pose maps a spin angle (and, for multi-motif ligands, a choice among alternative
conformer fits) to ligand coordinates. Single-motif ligands spin about the normal of
their site; multi-motif ligands spin about the axis through their anchors when two
anchors leave it free, and are otherwise rigid.
"""
from __future__ import annotations
import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constellation import Candidate, ConstellationSampler, rotations_about_axis
from .core import BindingSite
from .ligand import Ligand, LigandSpec
from .refine import refine, refine_joint, stretch
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
    "refined", "stretched"); theta is the spin about the anchor axis relative to each.
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


def stretch_pose(pose: FitPose, core_tree, cutoff: float) -> bool:
    """
    Put a stretched copy of the pose's best variant first: its dangling chains anti and
    raised along the mean normal of its sites, its anchors untouched. Returns whether the
    pose now starts stretched: it had chains to stretch and the result clears the core,
    or site selection already stretched it.
    """
    if pose.origins[pose.variant] == "stretched":
        return True
    normal = np.mean([np.asarray(s.plane, dtype=float) / np.linalg.norm(s.plane)
                      for s in pose.candidate.sites], axis=0)
    res = stretch(pose.ligand.torsion_model(), pose.variants[0], pose.ligand.binding_groups,
                  pose.candidate.targets, normal / np.linalg.norm(normal), [core_tree], cutoff + 0.2)
    if res is None or float(core_tree.query(res.coords)[0].min()) < cutoff:
        return False
    pose.variant = pose.add_variant(res.coords, origin="stretched", first=True)
    return True


# Soft-sphere radii (cutoff + margin) tried in turn. A wide margin clears contacts with
# room to spare, but in a dense shell it also pushes on every contact that is already
# fine, and can leave a real one below the cutoff; the narrow one then acts on those alone.
MARGINS = (0.2, 0.05)


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
) -> bool:
    """
    Refine jammed multi-motif ligands together, for when no ligand can clear its contacts
    alone.

    Ligands closer than `cutoff` form a contact graph; in every connected component with a
    contact below the cutoff (a single ligand touching the core included), the multi-motif
    members are refined jointly against the core and all other ligands, and pushed apart
    from each other wherever they come within `cutoff` + margin (see MARGINS). A result
    replaces the poses only if its anchors stay within `max_rmsd` and its closest contact
    improves. Updates `coords_list` and the poses in place.

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
        for margin in MARGINS:
            results = refine_joint(
                [poses[i].ligand.torsion_model() for i in members],
                [coords_list[i] for i in members],
                [poses[i].ligand.binding_groups for i in members],
                [poses[i].candidate.targets for i in members],
                trees, cutoff + margin, tree_builder,
                stop_at=cutoff + margin / 2, stop_rmsd=max_rmsd,
            )
            if any(r.anchor_rmsd > max_rmsd for r in results):
                continue
            trial = list(coords_list)
            for i, r in zip(members, results):
                trial[i] = r.coords
            after = min(closest(trial, i) for i in members)
            if after > before + 1e-3:
                for i, r in zip(members, results):
                    coords_list[i] = r.coords
                    poses[i].variant = poses[i].add_variant(r.coords)
                improved = True
                break
    return improved


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
    groups = lig.binding_groups
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
            cand, reason = sampler.fit_sites(lib, sites, offset, replace, model, groups)
            if cand is None:
                print(f"[Warning] Binding site group {group} rejected for ligand {lig.name}: {reason}.")
                continue
            sampler.accept(cand, lib, replace)
            chosen.append(cand)
    else:
        capacity = sampler.capacity([m.site for m in lig.binding_motifs], by_type)
        if 0.0 < spec.coverage <= 1.0:
            n_target = int(math.ceil(spec.coverage * capacity))
        else:
            n_target = int(spec.coverage)
            if n_target > capacity:
                print(f"[Warning] Requested {n_target} ligands {lig.name}, but the free sites "
                      f"host at most {capacity}.")
                n_target = capacity
        chosen, failures = sampler.sample(lib, pools, n_target, offset, replace, model, groups)
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
        f"({'/'.join(m.site for m in lig.binding_motifs)} sites).")

    out = []
    for cand in chosen:
        clone = lig.clone()
        clone.planes = [s.plane for s in cand.sites]
        clone._anchor_offset = offset
        out.append((clone, cand))
    return out
