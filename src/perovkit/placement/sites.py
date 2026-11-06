from __future__ import annotations
from collections import Counter
import math
from numbers import Real
import random
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import BindingSite, Core
from ..ligand import Ligand
from ..utils.sampling import farthest_point_sampling, per_face_farthest_point_sampling
from .fitting import Candidate

if TYPE_CHECKING:
    from .sampler import ConstellationSampler


def _n_target(coverage: float, capacity: int) -> int:
    """Ligands a coverage asks for: up to 1, a fraction of `capacity`; above, a number."""
    return int(math.ceil(coverage * capacity)) if 0.0 < coverage <= 1.0 else int(coverage)


def assign_faces(sites: Sequence[BindingSite]) -> List[tuple]:
    """
    The face each site counts toward when spreading ligands over a nanocrystal.

    A site on a face counts toward it. A site on an edge or corner lies on two or three
    faces at once: along each edge (or among the corners of one kind) the sites are dealt
    to those faces in turn, so that every face gets its share instead of all edges going
    to one axis. On a slab every site keeps its plane.
    """
    faces: List[Optional[tuple]] = [None] * len(sites)
    shared: Dict[tuple, List[int]] = {}
    for i, s in enumerate(sites):
        plane = tuple(int(v) for v in s.plane)
        if sum(v != 0 for v in plane) <= 1:
            faces[i] = plane
        else:
            shared.setdefault(plane, []).append(i)
    for plane, idx in shared.items():
        options = [tuple(v if a == ax else 0 for a, v in enumerate(plane))
                   for ax in range(3) if plane[ax] != 0]
        for k, i in enumerate(sorted(idx, key=lambda i: tuple(sites[i].position))):
            faces[i] = options[k % len(options)]
    return faces


def take_sites(
    core: Core,
    lig: Ligand,
    coverage: float | Sequence[int],
    sites: List[BindingSite],
    rng: random.Random,
    displaced: List[int],
    log: Callable,
) -> List[Tuple[Ligand, BindingSite]]:
    """
    Take free sites for a single-motif ligand on the sublattice of its motif: those
    `coverage` lists, or else as many as it asks for, spread by farthest-point sampling.
    Returns a copy of the ligand for each site; the surface atoms the ligands replace are
    added to `displaced`.
    """
    motif = lig.binding_motifs[0]
    sublattice = motif.site
    symbol = core.site_symbols[sublattice]
    available = [s for s in sites if s.symbol == symbol and s.available]

    if not isinstance(coverage, Real):
        chosen_sites = []
        for idx in coverage:
            site = next((s for s in available if idx in s.atom_indices), None)
            if site is None:
                print(f"[Warning] Requested binding site index {idx} not available for ligand {lig.name}.")
            elif site not in chosen_sites:
                chosen_sites.append(site)
    else:
        n_target = _n_target(coverage, len(available))
        if n_target > len(available):
            print(f"[Warning] Requested {n_target} sites for ligand, "
                  f"but only {len(available)} available.")
            n_target = len(available)

        coords = np.array([s.position for s in available])
        sc = core.supercell
        if core.is_nanocrystal and sc is not None and len(set(sc)) > 1:
            planes = [s.plane for s in available]
            chosen_indices = per_face_farthest_point_sampling(coords, planes, n_target, rng)
        else:
            chosen_indices = farthest_point_sampling(coords, n_target, rng)
        chosen_sites = [available[i] for i in chosen_indices]

    lig.torsion_model()   # shared by the clones
    taken = []
    for site in chosen_sites:
        clone = lig.clone()
        clone.planes = [site.plane]
        site.available = False
        taken.append((clone, site))
        # Replace mode removes the surface atom(s); adsorb mode keeps them.
        if not motif.adsorb:
            displaced.extend(site.atom_indices)

    mode = "Adsorbed" if motif.adsorb else "Placed"
    log(f"[Log] {mode} total {len(chosen_sites)} {sublattice}-site ligands.")
    return taken


def motif_pools(
    lig: Ligand,
    binding_sites: List[BindingSite],
    symbol_map: Dict[str, str],
) -> Tuple[List[List[BindingSite]], Dict[str, List[BindingSite]]]:
    """
    Candidate sites of every motif, and the sites of every sublattice the motifs need.
    """
    by_sublattice = {sublattice: [s for s in binding_sites if s.symbol == symbol_map[sublattice]]
                     for sublattice in {m.site for m in lig.binding_motifs}}
    return [by_sublattice[m.site] for m in lig.binding_motifs], by_sublattice


def select_constellations(
    lig: Ligand,
    coverage: float | Sequence[Sequence[int]],
    pools: List[List[BindingSite]],
    by_sublattice: Dict[str, List[BindingSite]],
    sampler: ConstellationSampler,
    log: Callable = print,
    refine_torsions: bool = True,
) -> List[Tuple[Ligand, Candidate]]:
    """
    Choose the constellations of one multi-motif ligand and clone it for each.

    A list `coverage` gives the constellations, each one site atom index per motif, in
    motif order. Otherwise `coverage` is a fraction of the most ligands the free sites
    can host (or, above 1, a count), spread so that every face holds about the same
    fraction of its sites (see ConstellationSampler.sample), and seeds are tried until it
    is met or the surface is exhausted; a shortfall is reported with the reasons seeds were
    rejected. The first motif's sites set the per-face balance, so it is best given the
    scarcer sublattice. With `refine_torsions`, constellations whose fits all clash with
    the core get a torsion-space refinement first.
    """
    shadowed = sampler.n_shadowed
    chosen: List[Candidate] = []

    if not isinstance(coverage, Real):
        for group in coverage:
            group = list(group) if isinstance(group, (list, tuple)) else [group]
            if len(group) != lig.n_motifs:
                print(f"[Warning] Binding site group {group} for ligand {lig.name} needs one "
                      f"site index per motif ({lig.n_motifs}). Skipped.")
                continue
            sites = [next((s for s in pool if idx in s.atom_indices and s.available), None)
                     for idx, pool in zip(group, pools)]
            if any(s is None for s in sites) or len({id(s) for s in sites}) < len(sites):
                print(f"[Warning] Binding site group {group} is not available for ligand "
                      f"{lig.name}. Skipped.")
                continue
            cand, reason = sampler.fit_sites(lig, sites, refine_torsions)
            if cand is None:
                print(f"[Warning] Binding site group {group} rejected for ligand {lig.name}: {reason}.")
                continue
            chosen.append(cand)
    else:
        # each sublattice hosts as many ligands as it has free sites per motif needing it
        need = Counter(m.site for m in lig.binding_motifs)
        capacity = min(sum(s.available for s in by_sublattice[sublattice]) // k
                       for sublattice, k in need.items())
        n_target = _n_target(coverage, capacity)
        if n_target > capacity:
            print(f"[Warning] Requested {n_target} ligands {lig.name}, but the free sites "
                  f"host at most {capacity}.")
            n_target = capacity
        chosen, failures = sampler.sample(lig, pools, n_target, refine_torsions)
        if len(sampler.occupancy) > 1:
            occ = list(sampler.occupancy.values())
            log(f"[Log] {lig.name} occupancy per face: {min(occ):.2f}-{max(occ):.2f} "
                f"over {len(occ)} faces.")
        if len(chosen) < n_target:
            reasons = ", ".join(f"{r} ({n}x)" for r, n in failures.most_common())
            print(f"[Warning] Placed {len(chosen)} of {n_target} requested {lig.name} ligands; "
                  f"rejected seeds: {reasons or 'none left to try'}.")

    mode = "Adsorbed" if all(m.adsorb for m in lig.binding_motifs) else "Placed"
    log(f"[Log] {mode} total {len(chosen)} {lig.name} ligands with {lig.n_motifs} motifs "
        f"({'/'.join(m.site for m in lig.binding_motifs)} sites); {sampler.n_shadowed - shadowed} "
        f"free sites under them taken out of use.")

    out = []
    for cand in chosen:
        clone = lig.clone()
        clone.planes = [s.plane for s in cand.sites]
        out.append((clone, cand))
    return out
