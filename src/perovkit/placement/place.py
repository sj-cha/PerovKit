from __future__ import annotations
from copy import deepcopy
from numbers import Real
import random
from typing import List, Optional, Sequence

import numpy as np

from ..assembly import Assembly
from ..config import PlacementConfig
from ..core import Core
from ..ligand import Ligand
from .pack import Contacts, intramolecular_contacts, pack
from .polymerize import bond_units
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
    config: Optional[PlacementConfig] = None,
    random_seed: int = 42,
    verbose: bool = True,
    n_workers: int = 8,   # at most the CPUs available
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

    config = PlacementConfig() if config is None else config
    log = print if verbose else (lambda *a, **k: None)
    rng = random.Random(random_seed)
    box = core.box
    sites = deepcopy(core.binding_sites)
    core_positions = core.atoms.get_positions()
    requests = _in_order(ligands, coverage)

    displaced: List[int] = []
    poses: List[SingleMotifPose | MultiMotifPose] = []
    # single-motif ligands too ask it which sites the multi-motif ones cover
    sampler = ConstellationSampler(core_positions, box, rng, sites, config)

    for lig, cov in requests:
        if lig.n_motifs > 1:
            pools, by_sublattice = motif_pools(lig, sites, core.site_symbols)
            for lig_cloned, cand in select_constellations(lig, cov, pools, by_sublattice, sampler, log,
                                                          refine_torsions=config.refine_torsions):
                poses.append(MultiMotifPose(lig_cloned, cand))
            continue

        # sites under a multi-motif ligand cannot take one more
        n_blocked = sampler.block_covered(sites, config.overlap_cutoff + config.cover_margin)
        if n_blocked:
            log(f"[Log] Left out {n_blocked} free sites covered by multi-motif ligands.")

        for lig_cloned, site in take_sites(core, lig, cov, sites, rng, displaced, log):
            poses.append(SingleMotifPose(lig_cloned, np.array(site.plane, dtype=float),
                                  np.array(site.position, dtype=float)))

    if not poses:
        raise RuntimeError("No ligands to place.")

    displaced.extend(sampler.displaced)   # the site atoms multi-motif ligands replace
    core_points, _ = box.images(np.delete(core_positions, displaced, axis=0))
    contacts = Contacts(core_points, box, config.overlap_cutoff, config.interior_depth * core.a,
                        config.contact_reach)
    coords_list = pack(poses, contacts, rng, config, config.active_radius_factor * core.a + 1e-2, n_workers,
                       log=log, verbose=verbose)
    reactive = [l for l, p in enumerate(poses) if p.ligand.reactive_groups]
    bonds = []
    if config.polymerize and reactive:
        bonds = bond_units(poses, coords_list, contacts, config, log)
    bonded = {l: set() for l in reactive}
    for l, a, m, b in bonds:
        bonded[l].add(a)
        bonded[m].add(b)
    for l in reactive:
        lig = poses[l].ligand
        lig.atoms.set_positions(coords_list[l])
        lig._settle_reactive(bonded[l])
        coords_list[l] = lig.atoms.get_positions()
    assembly = _assemble(core, requests, [p.ligand for p in poses], coords_list, displaced, bonds, config)

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
        min_distance, n_close = intramolecular_contacts(poses, coords_list, config.intra_scale)
        log(f"[Log] Multi-motif anchors sit {dev.mean():.2f} Å (max {dev.max():.2f} Å) from their sites; "
            f"{bent} of {len(multi_motif_poses)} ligands refined in torsion space; closest intramolecular "
            f"contact {min_distance:.2f} Å ({n_close} pairs within {config.intra_scale:g}x their vdW radii).")

    return assembly


def _assemble(
    core: Core,
    requests: List[tuple],
    ligands: List[Ligand],
    coords_list: List[np.ndarray],
    displaced: List[int],
    bonds: List[tuple],
    config: PlacementConfig,
) -> Assembly:
    """
    Give the ligands their coordinates and ids, and put them on a copy of `core` without
    the `displaced` atoms, keeping the core's octahedra and lattice coordinates; `bonds`
    join ligands, which take the ids of their places in `ligands`. The assembly records
    `config`.
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
    return Assembly(core=stripped, ligands=list(ligands), ligand_coverage=coverage, ligand_bonds=list(bonds),
                    config=config)


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
