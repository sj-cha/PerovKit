from __future__ import annotations
from copy import deepcopy
import math
import random
from typing import List, Optional, Sequence

from ase import Atoms
import numpy as np
from scipy.spatial import cKDTree

from ..assembly import Assembly
from ..core import Core
from ..ligand import Ligand, LigandSpec
from ..refine import INTRA_SCALE
from .pack import Contacts, intramolecular_contacts, pack
from .poses import MultiMotifPose, SingleMotifPose, place_about_normal
from .sampler import ConstellationSampler
from .sites import motif_pools, select_constellations, take_sites


def _in_order(ligand_specs: Sequence[LigandSpec]) -> List[LigandSpec]:
    # Multi-motif ligands need site combinations, so they choose first
    return sorted(ligand_specs, key=lambda spec: (
        0 if spec.ligand.n_motifs > 1 else 1,
        0 if spec.ligand.charge > 0 else 1,
        -spec.ligand.volume,
    ))


def place_ligands(
    core: Core,
    ligand_specs: Sequence[LigandSpec],
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
    n_workers: int = 32,   # at most the CPUs available
    **sampler_options,
) -> Assembly:
    """
    Implements Core.place_ligands, which documents the options.
    """
    log = print if verbose else (lambda *a, **k: None)
    rng = random.Random(random_seed)
    box = core.box
    sites = deepcopy(core.binding_sites)
    core_positions = core.atoms.get_positions()
    specs = _in_order(ligand_specs)

    displaced: List[int] = []
    poses: List[SingleMotifPose | MultiMotifPose] = []
    sampler: Optional[ConstellationSampler] = None

    for spec in specs:
        if spec.ligand.n_motifs > 1:
            if sampler is None:
                sampler = ConstellationSampler(
                    core_positions, box, rng,
                    displaced=displaced, overlap_cutoff=overlap_cutoff,
                    max_rmsd=max_anchor_rmsd, sites=sites, **sampler_options,
                )
            pools, by_type = motif_pools(spec, sites, core.site_symbols)
            for lig_cloned, cand in select_constellations(spec, pools, by_type, sampler, log,
                                                          refine=refine_torsions):
                poses.append(MultiMotifPose(lig_cloned, cand))
                if not spec.adsorb:
                    displaced.extend(cand.atom_indices)
            continue

        if sampler is not None:
            # sites under a multi-motif ligand cannot take one more
            n_blocked = sampler.block_covered(sites, overlap_cutoff + 1.0)
            if n_blocked:
                log(f"[Log] Left out {n_blocked} free sites covered by multi-motif ligands.")

        for lig_cloned, site in take_sites(core, spec, sites, rng, displaced, log):
            poses.append(SingleMotifPose(lig_cloned, np.array(site.plane, dtype=float),
                                  np.array(site.position, dtype=float)))

    assert poses, "No ligands to place."

    core_tree, _ = box.tree(np.delete(core_positions, displaced, axis=0))
    contacts = Contacts(core_tree, box, overlap_cutoff)
    coords_list = pack(
        poses, contacts, rng, max_iters=max_iters, coarse_step_deg=coarse_step_deg,
        fine_step_deg=fine_step_deg, window_deg=window_deg,
        active_radius=active_radius_factor * core.a + 1e-2, max_rmsd=max_anchor_rmsd,
        refine_torsions=refine_torsions, stretch_tails=stretch_tails, n_workers=n_workers,
        log=log, verbose=verbose,
    )
    assembly = _assemble(core, specs, [p.ligand for p in poses], coords_list, displaced)

    multi_motif_poses = [p for p in poses if isinstance(p, MultiMotifPose)]
    if multi_motif_poses:
        dev = np.concatenate([
            np.linalg.norm(box.mic(p.ligand.anchor_positions - p.candidate.targets), axis=1)
            for p in multi_motif_poses
        ])
        bent = sum(p.origins[p.variant] != "fit" for p in multi_motif_poses)
        closest, n_close = intramolecular_contacts(poses, coords_list)
        log(f"[Log] Multi-motif anchors sit {dev.mean():.2f} Å (max {dev.max():.2f} Å) from their sites; "
            f"{bent} of {len(multi_motif_poses)} ligands refined in torsion space; closest intramolecular "
            f"contact {closest:.2f} Å ({n_close} pairs within {INTRA_SCALE:g}x their vdW radii).")

    return assembly


def place_ligands_manually(
    core: Core,
    ligand_specs: Sequence[LigandSpec],
    cat_x_move: float = 0.0,
    cat_y_move: float = 0.0,
    cat_z_move: float = 0.0,
    cat_z_rotate: float = 0.0,
    an_x_move: float = 0.0,
    an_y_move: float = 0.0,
    an_z_move: float = 0.0,
    an_z_rotate: float = 0.0,
    ligand_ids: Optional[Sequence[int]] = None,
    random_seed: int = 42,
    verbose: bool = True,
) -> Assembly:
    """
    Place every ligand at its site without the rotation search, cations shifted by
    cat_x/y/z_move (Å) and turned by cat_z_rotate (degrees), anions by the an_* values.
    With `ligand_ids`, only those ligands (ids count from 0 in placement order) are
    shifted and turned; the rest sit at their sites unmoved.
    """
    specs = _in_order(ligand_specs)
    if any(spec.ligand.n_motifs > 1 for spec in specs):
        raise NotImplementedError("Manual placement supports single-motif ligands only.")

    log = print if verbose else (lambda *a, **k: None)
    rng = random.Random(random_seed)
    sites = deepcopy(core.binding_sites)
    displaced: List[int] = []
    placed = [pair for spec in specs
              for pair in take_sites(core, spec, sites, rng, displaced, log)]
    assert placed, "No ligands to place."

    moved = set(range(len(placed))) if ligand_ids is None else {int(i) for i in ligand_ids}
    unknown = sorted(moved - set(range(len(placed))))
    if unknown:
        print(f"[Warning] No ligands with ids {unknown}; ids run from 0 to {len(placed) - 1}.")

    cation = np.array([cat_x_move, cat_y_move, cat_z_move], dtype=float), math.radians(cat_z_rotate)
    anion = np.array([an_x_move, an_y_move, an_z_move], dtype=float), math.radians(an_z_rotate)
    unmoved = np.zeros(3), 0.0

    coords_list = []
    for i, (lig, site) in enumerate(placed):
        shift, theta = (cation if lig.charge > 0 else anion) if i in moved else unmoved
        coords = place_about_normal(lig.atoms.get_positions(), np.array(site.plane, dtype=float),
                                    np.array(site.position, dtype=float), lig._anchor_offset, theta)
        coords_list.append(coords + shift)

    return _assemble(core, specs, [lig for lig, _ in placed], coords_list, displaced)


def _assemble(
    core: Core,
    specs: List[LigandSpec],
    ligands: List[Ligand],
    coords_list: List[np.ndarray],
    displaced: List[int],
) -> Assembly:
    """
    Give the ligands their coordinates and ids, and put them on a copy of `core` without
    the `displaced` atoms.
    """
    max_z = 0
    for i, (lig, coords) in enumerate(zip(ligands, coords_list)):
        lig.atoms.set_positions(coords)
        lig.id = i
        max_z = max(max_z, coords[:, 2].max())

    keep = np.ones(len(core.atoms), dtype=bool)
    keep[displaced] = False
    stripped_atoms = Atoms(
        numbers=core.atoms.numbers[keep],
        positions=core.atoms.positions[keep],
        pbc=False,
    )
    if core.is_slab:
        # Periodic along z too, across the vacuum: an MLIP then sees what VASP sees
        stripped_atoms.set_pbc(True)
        stripped_atoms.set_cell(core.atoms.get_cell())
        stripped_atoms.cell[2, 2] = max_z + core.vacuum

    # An organic A-site loses a molecule with each surface cation replaced: re-derive the
    # surviving cations from the preserved a_site_id grouping
    A_value = core.A
    if isinstance(core.A, list):
        stripped_atoms.set_array("a_site_id", core.atoms.get_array("a_site_id")[keep])
        A_value = Core._build_A_ligands(stripped_atoms, core.A[0])

    stripped = Core(
        A=A_value,
        B=core.B,
        X=core.X,
        atoms=stripped_atoms,
        a=core.a,
        supercell=core.supercell,
        vacuum=core.vacuum,
        build_surface=False,
    )
    stripped._build_octahedra()
    _assign_to_octahedra(stripped, ligands)

    return Assembly(core=stripped, ligands=list(ligands),
                    ligand_coverage={spec.ligand.name: spec.coverage for spec in specs})


def _assign_to_octahedra(core: Core, ligands: List[Ligand]) -> None:
    octahedra = core.octahedra
    if not octahedra:
        return

    at = core.atoms
    syms = np.array(at.get_chemical_symbols())
    b_idx = np.where(syms == core.B)[0]

    core_pos, boxsize = core._box_positions()
    B_pos = core_pos[b_idx]

    anchor_positions = []
    ligand_ids = []

    for lig in ligands:
        if lig.n_motifs > 1:
            # every motif bound to an X or B site belongs to the octahedron there
            for motif, pos in zip(lig.binding_motifs, lig.anchor_positions):
                if motif.site in ("X", "B"):
                    anchor_positions.append(pos)
                    ligand_ids.append(int(lig.id))
        elif lig.charge <= 0 and lig.anchor_pos is not None:
            anchor_positions.append(np.asarray(lig.anchor_pos, dtype=float))
            ligand_ids.append(int(lig.id))

    if anchor_positions:
        anchor_positions = np.vstack(anchor_positions)

        b_tree = cKDTree(B_pos, boxsize=boxsize)
        _, b_loc = b_tree.query(anchor_positions, k=1)

        for j, b_loc_j in enumerate(b_loc):
            b_abs = int(b_idx[int(b_loc_j)])
            if ligand_ids[j] not in octahedra[b_abs]["Ligand"]:
                octahedra[b_abs]["Ligand"].append(ligand_ids[j])
