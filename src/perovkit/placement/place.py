from __future__ import annotations
from copy import deepcopy
import math
from numbers import Real
import random
from typing import List, Sequence

import numpy as np

from ..assembly import Assembly
from ..core import Core
from ..ligand import Ligand
from ..refine import INTRA_SCALE
from .pack import Contacts, intramolecular_contacts, pack
from .poses import MultiMotifPose, SingleMotifPose
from .sampler import ConstellationSampler
from .sites import motif_pools, select_constellations, take_sites


def _in_order(ligands: Sequence[Ligand], coverage: Sequence) -> List[tuple]:
    """(ligand, coverage) pairs, multi-motif ligands first: they need site combinations."""
    def key(pair):
        lig = pair[0]
        return 0 if lig.n_motifs > 1 else 1, 0 if lig.charge > 0 else 1, -lig.volume
    return sorted(zip(ligands, coverage), key=key)


def place_ligands(
    core: Core,
    ligands: Ligand | Sequence[Ligand],
    coverage: float | Sequence,
    max_iters: int = 10,
    overlap_cutoff: float = 1.6,   # Å
    coarse_step_deg: int = 18,
    fine_step_deg: int = 2,
    window_deg: int = 12,
    active_radius_factor: float = math.sqrt(2)/2,
    random_seed: int = 42,
    verbose: bool = True,
    max_anchor_rmsd: float = 0.5,   # Å
    refine_torsions: bool = True,
    stretch_tails: bool = True,
    n_workers: int = 8,   # at most the CPUs available
    **sampler_options,
) -> Assembly:
    """
    Implements Core.place_ligands, which documents the arguments.
    """
    if isinstance(ligands, Ligand):   # one ligand given as it is, and so its coverage
        ligands, coverage = [ligands], [coverage]
    elif isinstance(coverage, Real):
        raise ValueError("with a list of ligands, give a list of coverages, one per ligand")
    if len(coverage) != len(ligands):
        raise ValueError(f"{len(ligands)} ligands but {len(coverage)} coverages; give one per ligand")
    for lig, cov in zip(ligands, coverage):
        if not lig.binding_motifs:
            raise ValueError(f"ligand {lig.name!r} has no BindingMotif to bind it to the surface")
        if isinstance(cov, Real) and cov < 0:
            raise ValueError(f"coverage must be at least 0; got {cov} for ligand {lig.name!r}")

    log = print if verbose else (lambda *a, **k: None)
    rng = random.Random(random_seed)
    box = core.box
    sites = deepcopy(core.binding_sites)
    core_positions = core.atoms.get_positions()
    requests = _in_order(ligands, coverage)

    displaced: List[int] = []
    poses: List[SingleMotifPose | MultiMotifPose] = []
    # made even without multi-motif ligands, so that a misspelled option fails at once
    sampler = ConstellationSampler(core_positions, box, rng, sites, overlap_cutoff=overlap_cutoff,
                                   max_rmsd=max_anchor_rmsd, **sampler_options)

    for lig, cov in requests:
        if lig.n_motifs > 1:
            pools, by_sublattice = motif_pools(lig, sites, core.site_symbols)
            for lig_cloned, cand in select_constellations(lig, cov, pools, by_sublattice, sampler, log,
                                                          refine_torsions=refine_torsions):
                poses.append(MultiMotifPose(lig_cloned, cand))
            continue

        # sites under a multi-motif ligand cannot take one more
        n_blocked = sampler.block_covered(sites, overlap_cutoff + 1.0)
        if n_blocked:
            log(f"[Log] Left out {n_blocked} free sites covered by multi-motif ligands.")

        for lig_cloned, site in take_sites(core, lig, cov, sites, rng, displaced, log):
            poses.append(SingleMotifPose(lig_cloned, np.array(site.plane, dtype=float),
                                  np.array(site.position, dtype=float)))

    if not poses:
        raise RuntimeError("No ligands to place.")

    displaced.extend(sampler.displaced)   # the site atoms multi-motif ligands replace
    core_points, _ = box.images(np.delete(core_positions, displaced, axis=0))
    contacts = Contacts(core_points, box, overlap_cutoff, core.a / 2)   # down to the first BX2 layer
    coords_list = pack(
        poses, contacts, rng, max_iters=max_iters, coarse_step_deg=coarse_step_deg,
        fine_step_deg=fine_step_deg, window_deg=window_deg,
        active_radius=active_radius_factor * core.a + 1e-2, max_rmsd=max_anchor_rmsd,
        refine_torsions=refine_torsions, stretch_tails=stretch_tails, n_workers=n_workers,
        log=log, verbose=verbose,
    )
    assembly = _assemble(core, requests, [p.ligand for p in poses], coords_list, displaced)

    def deviations(chosen: List) -> np.ndarray:
        return np.concatenate([np.linalg.norm(box.mic(p.ligand.anchor_positions - p.targets), axis=1)
                               for p in chosen])

    single_motif_poses = [p for p in poses if isinstance(p, SingleMotifPose)]
    if single_motif_poses:
        dev = deviations(single_motif_poses)
        log(f"[Log] Single-motif anchors sit {dev.mean():.2f} Å (max {dev.max():.2f} Å) from their sites.")

    multi_motif_poses = [p for p in poses if isinstance(p, MultiMotifPose)]
    if multi_motif_poses:
        dev = deviations(multi_motif_poses)
        bent = sum(p.sources[p.variant] != "fit" for p in multi_motif_poses)
        min_distance, n_close = intramolecular_contacts(poses, coords_list)
        log(f"[Log] Multi-motif anchors sit {dev.mean():.2f} Å (max {dev.max():.2f} Å) from their sites; "
            f"{bent} of {len(multi_motif_poses)} ligands refined in torsion space; closest intramolecular "
            f"contact {min_distance:.2f} Å ({n_close} pairs within {INTRA_SCALE:g}x their vdW radii).")

    return assembly


def _assemble(
    core: Core,
    requests: List[tuple],
    ligands: List[Ligand],
    coords_list: List[np.ndarray],
    displaced: List[int],
) -> Assembly:
    """
    Give the ligands their coordinates and ids, and put them on a copy of `core` without
    the `displaced` atoms, keeping the core's octahedra and lattice coordinates.
    """
    max_z = 0
    for i, (lig, coords) in enumerate(zip(ligands, coords_list)):
        lig.atoms.set_positions(coords)
        lig.id = i
        max_z = max(max_z, coords[:, 2].max())

    keep = np.ones(len(core.atoms), dtype=bool)
    keep[displaced] = False
    stripped_atoms = core.atoms[keep]
    if core.is_slab:
        # Periodic along z too, across the vacuum: an MLIP then sees what VASP sees
        stripped_atoms.set_pbc(True)
        stripped_atoms.cell[2, 2] = max_z + core.vacuum

    stripped = Core(
        A=core.A[:1] if core.has_organic_A else core.A,
        B=core.B,
        X=core.X,
        atoms=stripped_atoms,
        a=core.a,
        supercell=core.supercell,
        vacuum=core.vacuum,
        build_surface=False,
    )
    # the core's own octahedra, renumbered: strain or tilt can defeat a distance search
    new = np.cumsum(keep) - 1
    stripped.octahedra = {int(new[b]): {"X": [int(new[x]) for x in octa["X"] if keep[x]], "Ligand": []}
                          for b, octa in core.octahedra.items() if keep[b]}
    stripped.B_ijk = {int(new[b]): ijk for b, ijk in core.B_ijk.items() if keep[b]}
    _assign_to_octahedra(stripped, ligands)

    # as plain Python numbers, which JSON takes; None for a ligand given its sites
    coverage = {lig.name: np.asarray(cov).item() if isinstance(cov, Real) else None
                for lig, cov in requests}
    return Assembly(core=stripped, ligands=list(ligands), ligand_coverage=coverage)


def _assign_to_octahedra(core: Core, ligands: List[Ligand]) -> None:
    octahedra = core.octahedra
    if not octahedra:
        return

    b_idx = np.flatnonzero(np.array(core.atoms.get_chemical_symbols()) == core.B)

    anchor_positions = []
    ligand_ids = []

    for lig in ligands:
        # every motif bound to an X or B site belongs to the octahedron there
        for motif, pos in zip(lig.binding_motifs, lig.anchor_positions):
            if motif.site in ("X", "B"):
                anchor_positions.append(pos)
                ligand_ids.append(int(lig.id))

    if anchor_positions:
        # the nearest B, minimum images along the periodic axes
        tree, image_of = core.box.tree(core.atoms.positions[b_idx])
        _, nearest = tree.query(np.vstack(anchor_positions))

        for j, k in enumerate(nearest):
            b_abs = int(b_idx[image_of[k]])
            if ligand_ids[j] not in octahedra[b_abs]["Ligand"]:
                octahedra[b_abs]["Ligand"].append(ligand_ids[j])
