from __future__ import annotations
from collections import Counter
import math
import random
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import BindingSite, Core
from ..ligand import Ligand, LigandSpec
from ..utils.sampling import farthest_point_sampling, per_face_farthest_point_sampling
from .fitting import Candidate

if TYPE_CHECKING:
    from .sampler import ConstellationSampler


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
    spec: LigandSpec,
    sites: List[BindingSite],
    rng: random.Random,
    displaced: List[int],
    log: Callable,
) -> List[Tuple[Ligand, BindingSite]]:
    """
    Take free sites for a single-motif spec, those it names or else as many as its
    coverage asks for, spread by farthest-point sampling. The site type is `spec.site`,
    or else A for a cation and X otherwise. Returns a copy of the ligand for each site;
    the surface atoms the ligands replace are added to `displaced`.
    """
    lig = spec.ligand
    kind = spec.site if spec.site is not None else ("A" if lig.charge > 0 else "X")
    symbol = core.site_symbols[kind]
    available = [s for s in sites if s.symbol == symbol and not s.passivated]

    if spec.binding_sites is not None:
        chosen_sites = []
        for idx in spec.binding_sites:
            site = next((s for s in available if idx in s.atom_indices), None)
            if site is None:
                print(f"[Warning] Requested binding site index {idx} not available for ligand {lig.name}.")
            elif site not in chosen_sites:
                chosen_sites.append(site)
    else:
        if 0.0 < spec.coverage <= 1.0:
            n_target = int(math.ceil(spec.coverage * len(available)))
        else:
            n_target = int(spec.coverage)
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

    taken = []
    for site in chosen_sites:
        clone = lig.clone()
        clone.planes = [site.plane]
        clone._anchor_offset = float(spec.anchor_offset)
        site.passivated = True
        taken.append((clone, site))
        # Replace mode removes the surface atom(s); adsorb mode keeps them.
        if not spec.adsorb:
            displaced.extend(site.atom_indices)

    mode = "Adsorbed" if spec.adsorb else "Placed"
    log(f"[Log] {mode} total {len(chosen_sites)} {kind}-site ligands.")
    return taken


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
    scarcer site type. With `refine`, constellations whose fits all clash with the core get
    a torsion-space refinement first.
    """
    lig = spec.ligand
    lib = lig.conformers()
    replace = not spec.adsorb
    offset = float(spec.anchor_offset)
    tm = lig.torsion_model()
    model = tm if refine else None
    # the frame's van der Waals radii shadow the sites under it (ConstellationSampler.accept);
    # chains and rings that torsions turn can still get out of a neighbor's way
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
