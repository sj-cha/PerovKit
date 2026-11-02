from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import List, Dict, Optional, Tuple, Sequence
import json
from pathlib import Path

import random
import math
import numpy as np
from tqdm.auto import tqdm

from ase import Atoms
from ase.io import read, write
from ase.io.vasp import write_vasp

from scipy.spatial import cKDTree 

from .core import Core, BindingSite
from .ligand import Ligand, LigandSpec, BindingMotif
from .constellation import ConstellationSampler
from .refine import INTRA_SCALE
from .placement import (SitePose, FitPose, bounding_spheres, optimize_pose,
                        place_about_normal, motif_pools, refine_clusters, refine_pose,
                        select_constellations, stretch_poses, tree_min_distance,
                        intramolecular_contacts, DENSE_PAIRS)
from .utils.geometry import farthest_point_sampling, compute_bounding_spheres, build_neighbor_map


@dataclass
class Slab:
    core: Core
    ligand_specs: List[LigandSpec]
    ligands: List[Ligand] = field(default_factory=list)
    ligand_coverage: Dict[str, float] = field(default_factory=dict)

    @property
    def octahedra(self):
        return self.core.octahedra

    @property
    def B_ijk(self):
        return self.core.B_ijk

    def __post_init__(self):
        self.core = deepcopy(self.core)

        if getattr(self.core, "is_nanocrystal", False):
            raise ValueError(
                "Slab got a nanocrystal-mode Core. Use NanoCrystal class instead."
            )

        if getattr(self.core, "binding_sites", None):
            self.binding_sites = deepcopy(self.core.binding_sites)
        else:
            self.binding_sites = []
        # Multi-motif ligands need site combinations, so they choose first
        self.ligand_specs.sort(
            key=lambda spec: (
                0 if spec.ligand.n_motifs > 1 else 1,
                0 if spec.ligand.charge > 0 else 1,
                -spec.ligand.volume,
            )
        )
        self._atoms: Optional[Atoms] = None
        self._setup_pbc()


    def _symbol_map(self) -> Dict[str, str]:
        return {
            "A": self.core.A_label,
            "B": self.core.B,
            "X": self.core.X,
        }


    def _select_site_pool(self, spec: LigandSpec) -> Tuple[List[BindingSite], str]:
        """
        Resolve which binding sites a LigandSpec targets.

        If `spec.site` ("A"/"B"/"X") is given it is used directly. Otherwise the
        target is inferred from the ligand charge (positive -> A-site, otherwise
        X-site) for backward compatibility. Returns (pool, label).
        """
        symbol_map = self._symbol_map()
        if spec.site is not None:
            target = symbol_map[spec.site]
            label = f"{spec.site}-site"
        elif spec.ligand.charge > 0:
            target, label = self.core.A_label, "A-site"
        else:
            target, label = self.core.X, "X-site"

        pool = [s for s in self.binding_sites if s.symbol == target]
        return pool, label


    def place_ligands(
        self,
        max_iters: int = 10,
        overlap_cutoff: float = 1.8,   # Å
        coarse_step_deg: int = 18,
        fine_step_deg: int = 2,
        window_deg: int = 12,
        active_radius_factor: float = math.sqrt(2)/2,
        random_seed: int = 42,
        verbose: bool = True,
        max_anchor_rmsd: float = 0.5,   # Å
        refine_torsions: bool = True,
        stretch_tails: bool = True,
        n_workers: int = 1,
    ) -> None:

        assert not self.ligands, "Ligands have already been placed."

        log = print if verbose else (lambda *a, **k: None)

        self._rng = random.Random(random_seed)
        self.overlap_cutoff = overlap_cutoff
        self.coarse_step_deg = coarse_step_deg
        self.fine_step_deg = fine_step_deg
        self.window_deg = window_deg
        self.active_radius_factor = active_radius_factor
        
        self.ligands = []
        displaced_indices = []
        for s in self.binding_sites:
            s.passivated = False

        core_positions = self.core.atoms.get_positions()

        ligands: List[Ligand] = []
        poses: List[SitePose | FitPose] = []
        sampler: Optional[ConstellationSampler] = None

        for spec in self.ligand_specs:
            lig = spec.ligand

            if lig.n_motifs > 1:
                if sampler is None:
                    sampler = ConstellationSampler(
                        core_positions, *self._sampler_geometry(), self._rng,
                        displaced=displaced_indices, overlap_cutoff=overlap_cutoff,
                        max_rmsd=max_anchor_rmsd,
                    )
                pools, by_type = motif_pools(spec, self.binding_sites, self._symbol_map())
                for lig_cloned, cand in select_constellations(spec, pools, by_type, sampler, log,
                                                              refine=refine_torsions):
                    ligands.append(lig_cloned)
                    poses.append(FitPose(lig_cloned, cand))
                    if not spec.adsorb:
                        displaced_indices.extend(cand.atom_indices)
                self.ligand_coverage[spec.ligand.name] = spec.coverage
                continue

            if sampler is not None:
                # sites under a multi-motif ligand cannot take one more
                n_blocked = sampler.block_covered(self.binding_sites, overlap_cutoff + 1.0)
                if n_blocked:
                    log(f"[Log] Left out {n_blocked} free sites covered by multi-motif ligands.")

            pool, ligand_type = self._select_site_pool(spec)
            available = [s for s in pool if not s.passivated]
            
            if spec.binding_sites is not None:
                if len(spec.binding_sites) == 0:
                    print(f"[Warning] Empty list provided for ligand {lig.name}. Skipping placement.")
                    continue
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
                chosen_indices = farthest_point_sampling(coords, n_target, self._rng)
                chosen_sites = [available[i] for i in chosen_indices]

            for site in chosen_sites:
                lig_cloned = lig.clone()
                lig_cloned.plane = site.plane
                lig_cloned._anchor_offset = float(spec.anchor_offset)
                site.passivated = True

                ligands.append(lig_cloned)
                poses.append(SitePose(lig_cloned, np.array(site.plane, dtype=float),
                                      np.array(site.position, dtype=float)))
                # Replace mode removes the surface atom(s); adsorb mode keeps them.
                if not spec.adsorb:
                    displaced_indices.extend(site.atom_indices)

            mode = "Adsorbed" if spec.adsorb else "Placed"
            log(f"[Log] {mode} total {len(chosen_sites)} {ligand_type} ligands.")
            self.ligand_coverage[spec.ligand.name] = spec.coverage

        n_lig = len(ligands)
        assert n_lig > 0, "No ligands to place."

        # Where each ligand binds: its site, or the centroid of its sites
        site_positions = np.array([p.center for p in poses])

        # Core atoms coordinates except displaced ones
        n_core = core_positions.shape[0]
        core_mask = np.ones(n_core, dtype=bool)

        for idx in displaced_indices:
            if 0 <= idx < n_core:
                core_mask[idx] = False

        core_coords = core_positions[core_mask]

        if core_coords.size > 0:
            core_tree, _ = self._build_pbc_tree(core_coords)
        else:
            core_tree = None
        self._core_tree = core_tree

        # Straighten the dangling chains of multi-motif ligands and raise them off the
        # surface before packing: a coiled library conformer tangles with its neighbors
        n_stretched = 0
        if stretch_tails:
            n_stretched = stretch_poses(poses, self._core_tree, self.overlap_cutoff, n_workers)
            if n_stretched:
                n_multi = sum(isinstance(p, FitPose) for p in poses)
                log(f"[Log] Stretched the tails of {n_stretched} of {n_multi} multi-motif ligands.")

        # Initialize rotation angles and current coordinates; multi-motif ligands start
        # from the spin their constellation fit chose
        thetas = np.array([2.0 * math.pi * self._rng.random() if p.random_start else 0.0
                           for p in poses], dtype=float)
        ligand_coords_list: List[np.ndarray] = [p.coords(t, getattr(p, "variant", 0))
                                                for p, t in zip(poses, thetas)]

        # Stretched chains leave few, shallow contacts: relax them all together while the
        # shell is still upright, before single ligands start turning
        if n_stretched and refine_torsions:
            # stop short of the cutoff: the last contacts are cheaper to clear one ligand at a time
            refine_clusters(poses, ligand_coords_list, self._core_tree, *self._sampler_geometry(),
                            self._exact_min_distance, self.overlap_cutoff, max_anchor_rmsd,
                            stop_at=self.overlap_cutoff - 0.1)

        # Rotation optimization loop
        centers, radii = bounding_spheres(poses, ligand_coords_list)
        neighbor_map = self._build_neighbor_map(centers, radii, self.overlap_cutoff)

        global_min, conflict_ligs, dists = self._get_conflict_ligands(
            ligand_coords_list,
            neighbor_map,
        )
        log(f"[Log] Initial global_min = {global_min:.3f} Å")

        for iter in range(1, max_iters + 1):
            if global_min >= self.overlap_cutoff:
                log(
                    f"[Log] Hard cutoff {self.overlap_cutoff:.3f} Å satisfied. Stopping optimization."
                )
                break

            active_cluster = self._build_active_cluster(conflict_ligs, site_positions)

            improved = False
            self._rng.shuffle(active_cluster)

            for i in tqdm(
                active_cluster,
                desc=f"Optimizing ligand (iter {iter})",
                disable=not verbose,
            ):
                neighbor_idx = neighbor_map.get(i, [])
                neighbors_coords = [ligand_coords_list[j] for j in neighbor_idx]

                # A multi-motif ligand is judged by its closest contact of either kind:
                # _min_distance stops at a core contact, which would let a long ligand
                # trade a sliver of core distance for running its tail through a neighbor
                multi = isinstance(poses[i], FitPose)
                min_distance = self._exact_min_distance if multi else self._min_distance
                d_old = min_distance(ligand_coords_list[i], neighbors_coords) if multi else dists[i]

                best_theta, best_variant, best_coords = self._optimize_rotation(
                    poses[i],
                    neighbors_coords,
                )
                d_new = min_distance(best_coords, neighbors_coords)

                # Multi-motif ligands that turning cannot clear may still bend out of the way.
                # Bending can reach past the neighbor spheres, so every other ligand counts.
                if refine_torsions and multi and d_new < self.overlap_cutoff:
                    others = [c for j, c in enumerate(ligand_coords_list) if j != i]
                    refined = refine_pose(poses[i], best_coords, others, self._core_tree,
                                          self._sampler_geometry()[0], self.overlap_cutoff,
                                          max_anchor_rmsd, min_distance)
                    if refined is not None:
                        d_ref = min_distance(refined, others)
                        if d_ref > d_new:
                            best_theta, best_variant = 0.0, poses[i].add_variant(refined)
                            best_coords, d_new = refined, d_ref

                if d_new > d_old + 1e-3:
                    ligand_coords_list[i] = best_coords
                    thetas[i] = best_theta
                    poses[i].variant = best_variant
                    improved = True
                    if multi:
                        # a refined variant may reach farther: let the ligands still to move see it
                        centers, radii = bounding_spheres(poses, ligand_coords_list)
                        neighbor_map = self._build_neighbor_map(centers, radii, self.overlap_cutoff)

            if not improved and refine_torsions:
                # nobody could move alone: let jammed multi-motif ligands give way together
                improved = refine_clusters(
                    poses, ligand_coords_list, self._core_tree, *self._sampler_geometry(),
                    self._exact_min_distance, self.overlap_cutoff, max_anchor_rmsd,
                )

            if not improved:
                raise RuntimeError("[Log] No improvement in active cluster. Stopping.")

            centers, radii = bounding_spheres(poses, ligand_coords_list)
            neighbor_map = self._build_neighbor_map(centers, radii, self.overlap_cutoff)
            global_min, conflict_ligs, dists = self._get_conflict_ligands(
                ligand_coords_list,
                neighbor_map,
            )
            log(f"[Log] Iter {iter}  global_min = {global_min:.3f} Å")

        if global_min < self.overlap_cutoff:
            raise RuntimeError("[Log] Maximum iterations reached without satisfying overlap cutoff.")

        # Apply final coordinates to Ligand
        max_z = 0
        for lig, coords in zip(ligands, ligand_coords_list):
            lig.atoms.set_positions(coords)
            lig.id = len(self.ligands)
            self.ligands.append(lig)

            if coords[:, 2].max() > max_z:
                max_z = coords[:, 2].max()

        fit_poses = [p for p in poses if isinstance(p, FitPose)]
        if fit_poses:
            dev = np.concatenate([
                np.linalg.norm(self._mic(p.ligand.anchor_positions - p.candidate.targets), axis=1)
                for p in fit_poses
            ])
            bent = sum(p.origins[p.variant] != "fit" for p in fit_poses)
            closest, n_close = intramolecular_contacts(poses, ligand_coords_list)
            log(f"[Log] Multi-motif anchors sit {dev.mean():.2f} Å (max {dev.max():.2f} Å) from their sites; "
                f"{bent} of {len(fit_poses)} ligands refined in torsion space; closest intramolecular "
                f"contact {closest:.2f} Å ({n_close} pairs within {INTRA_SCALE:g}x their vdW radii).")

        # Update core by removing displaced atoms           
        core_symbols = self.core.atoms.get_chemical_symbols()
        stripped_symbols = [s for s, keep in zip(core_symbols, core_mask) if keep]
        stripped_positions = core_positions[core_mask]

        old = self.core
        stripped_atoms = Atoms(
            symbols=stripped_symbols,
            positions=stripped_positions,
            pbc=[True, True, True],
            cell=old.atoms.get_cell(),
        )

        stripped_atoms.cell[2, 2] = max_z + old.vacuum

        # Rebuild the A-site. For an organic A-site, replacing a surface cation
        # removes its molecule, so re-derive the surviving cations from the
        # preserved a_site_id grouping; otherwise keep the symbol.
        A_value = old.A
        if "a_site_id" in old.atoms.arrays:
            stripped_atoms.set_array(
                "a_site_id", old.atoms.get_array("a_site_id")[core_mask]
            )
            if isinstance(old.A, list) and old.A:
                A_value = Core._build_A_ligands(stripped_atoms, old.A[0])

        self.core = Core(
            A=A_value,
            B=old.B,
            X=old.X,
            atoms=stripped_atoms,
            a=old.a,
            supercell=old.supercell,
            vacuum=old.vacuum,
            build_surface=False,
        )
        self.core._build_octahedra()
        self.core._build_B_ijk()

        self._assign_ligands_to_octahedra()
        self._build_index_map()
        self._build_and_link_atoms()


    def place_ligands_manually(
        self,
        cat_x_move: float = 0.0,
        cat_y_move: float = 0.0,
        cat_z_move: float = 0.0,
        cat_z_rotate: float = 0.0,
        an_x_move: float = 0.0,
        an_y_move: float = 0.0,
        an_z_move: float = 0.0,
        an_z_rotate: float = 0.0,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        assert not self.ligands, "Ligands have already been placed."
        if any(spec.ligand.n_motifs > 1 for spec in self.ligand_specs):
            raise NotImplementedError("Manual placement supports single-motif ligands only.")

        self._rng = random.Random(random_seed)
        
        log = print if verbose else (lambda *a, **k: None)

        self.ligands = []
        displaced_indices = []
        for s in self.binding_sites:
            s.passivated = False

        core_positions = self.core.atoms.get_positions()

        ligands: List[Ligand] = []
        sites: List[BindingSite] = []

        for spec in self.ligand_specs:
            lig = spec.ligand

            pool, ligand_type = self._select_site_pool(spec)
            available = [s for s in pool if not s.passivated]
            
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
                chosen_indices = farthest_point_sampling(coords, n_target, self._rng)
                chosen_sites = [available[i] for i in chosen_indices]

            for site in chosen_sites:
                lig_cloned = lig.clone()
                lig_cloned.plane = site.plane
                lig_cloned._anchor_offset = float(spec.anchor_offset)
                site.passivated = True

                ligands.append(lig_cloned)
                sites.append(site)
                # Replace mode removes the surface atom(s); adsorb mode keeps them.
                if not spec.adsorb:
                    displaced_indices.extend(site.atom_indices)

            mode = "Adsorbed" if spec.adsorb else "Placed"
            log(f"[Log] {mode} total {len(chosen_sites)} {ligand_type} ligands.")
            self.ligand_coverage[spec.ligand.name] = spec.coverage

        n_lig = len(ligands)
        assert n_lig > 0, "No ligands to place."

        # Binding site positions and planes
        site_positions = np.array([s.position for s in sites])
        site_planes = np.array([s.plane for s in sites], dtype=float)

        # Core atoms coordinates except displaced ones
        n_core = core_positions.shape[0]
        core_mask = np.ones(n_core, dtype=bool)

        for idx in displaced_indices:
            if 0 <= idx < n_core:
                core_mask[idx] = False

        ligand_coords_list: List[np.ndarray] = []
        for i in range(n_lig):
            if ligands[i].charge > 0:
                x_move = cat_x_move
                y_move = cat_y_move
                z_move = cat_z_move
                z_rotate = math.radians(cat_z_rotate)
            else:
                x_move = an_x_move
                y_move = an_y_move
                z_move = an_z_move
                z_rotate = math.radians(an_z_rotate)
            coords_i = self._place_one_ligand(
                ligands[i],
                site_planes[i],
                site_positions[i],
                z_rotate,
                x_move=x_move,
                y_move=y_move,
                z_move=z_move
            )
            ligand_coords_list.append(coords_i)

        # Apply final coordinates to Ligand
        max_z = 0
        for lig, coords in zip(ligands, ligand_coords_list):
            lig.atoms.set_positions(coords)
            lig.id = len(self.ligands)
            self.ligands.append(lig)

            if coords[:, 2].max() > max_z:
                max_z = coords[:, 2].max()

        # Update core by removing displaced atoms           
        core_symbols = self.core.atoms.get_chemical_symbols()
        stripped_symbols = [s for s, keep in zip(core_symbols, core_mask) if keep]
        stripped_positions = core_positions[core_mask]

        old = self.core
        stripped_atoms = Atoms(
            symbols=stripped_symbols,
            positions=stripped_positions,
            pbc=[True, True, True],
            cell=old.atoms.get_cell(),
        )

        stripped_atoms.cell[2, 2] = max_z + old.vacuum

        # Rebuild the A-site. For an organic A-site, replacing a surface cation
        # removes its molecule, so re-derive the surviving cations from the
        # preserved a_site_id grouping; otherwise keep the symbol.
        A_value = old.A
        if "a_site_id" in old.atoms.arrays:
            stripped_atoms.set_array(
                "a_site_id", old.atoms.get_array("a_site_id")[core_mask]
            )
            if isinstance(old.A, list) and old.A:
                A_value = Core._build_A_ligands(stripped_atoms, old.A[0])

        self.core = Core(
            A=A_value,
            B=old.B,
            X=old.X,
            atoms=stripped_atoms,
            a=old.a,
            supercell=old.supercell,
            vacuum=old.vacuum,
            build_surface=False,
        )
        self.core._build_octahedra()
        self.core._build_B_ijk()

        self._assign_ligands_to_octahedra()
        self._build_index_map()
        self._build_and_link_atoms()


    def remove_ligand(self, id: int, re_index: bool = True) -> None:
        idx = next(
            (i for i, lig in enumerate(self.ligands) if lig.id == id),
            None,
        )
        if idx is None:
            raise ValueError(f"No ligand with id={id} found.")

        self.ligands.pop(idx)

        # Re-index remaining ligands
        if re_index:
            for i, lig in enumerate(self.ligands):
                lig.id = i

            # Remove ligand references from octahedra and remap ids. Entries are rebuilt rather
            # than edited in place: frames loaded together by from_ase share one octahedra dict.
            if self.octahedra is not None:
                octahedra = {}
                for b_abs, octa in self.octahedra.items():
                    old_ids = octa["Ligand"]
                    new_ids = []
                    for lid in old_ids:
                        if lid == id:
                            continue
                        # Shift ids that were above the removed one
                        new_ids.append(lid - 1 if lid > id else lid)
                    octahedra[b_abs] = {**octa, "Ligand": new_ids}
                self.core.octahedra = octahedra

        self._build_index_map()
        self._build_and_link_atoms()


    def apply_tilt(
        self,
        glazer: str,
        angles: Tuple[float, float, float],
        *,
        order: str = "xyz",
        move_ligands: bool = True,
    ):
        from .tilt import apply_tilt 
        apply_tilt(
            structure=self,
            glazer=glazer,
            angles=angles,
            order=order,
            move_ligands=move_ligands,
        )

    
    def apply_strain(
        self,
        strain: Sequence[float],          # (ex, ey, ez)
        strain_ligands: bool = True,
    ):
        from .strain import apply_strain
        apply_strain(
            structure=self,
            strain=strain,
            strain_ligands=strain_ligands
        )


    def check_overlaps(self, cutoff: float = None, verbose: bool = True) -> float:
        log = print if verbose else (lambda *a, **k: None)

        if cutoff is None:
            cutoff = self.overlap_cutoff

        core_positions = self.core.atoms.get_positions()

        entities = []
        entity_labels = []

        if core_positions.size > 0:
            entities.append(core_positions)
            entity_labels.append("core")

        for i, lig in enumerate(self.ligands):
            entities.append(lig.atoms.get_positions())
            entity_labels.append(f"ligand_{i}")

        print(f"[DEBUG] #entities (core + ligands) = {len(entities)}")

        global_min = np.inf
        global_pair = None

        n = 0

        for i in range(len(entities)):
            tree_i, _ = self._build_pbc_tree(entities[i])
            for j in range(i + 1, len(entities)):
                dists_j, _ = tree_i.query(entities[j], k=1)
                d_min = float(np.min(dists_j))

                if d_min < global_min:
                    global_min = d_min
                    global_pair = (i, j)

                if d_min < cutoff:
                    log(
                        f"[OVERLAP] {entity_labels[i]} vs {entity_labels[j]}: "
                        f"min distance = {d_min:.3f} Å < {cutoff:.3f} Å"
                    )
                    n += 1

        log(
            f"[Log] global min inter-entity distance = {global_min:.3f} Å "
            f"between {entity_labels[global_pair[0]]} and {entity_labels[global_pair[1]]}"
            f" (total overlaps: {n})"
        )

        return global_min

    def _setup_pbc(self) -> None:
        cell = self.core.atoms.get_cell()
        self._box = np.asarray(cell.lengths(), dtype=float)  # (Lx, Ly, Lz)
        self._pbc = np.asarray(self.core.atoms.pbc, dtype=bool)

    def _mic(self, delta: np.ndarray) -> np.ndarray:
        d = np.array(delta, dtype=float, copy=True)
        for ax in range(3):
            if self._pbc[ax]:
                L = self._box[ax]
                d[..., ax] -= L * np.round(d[..., ax] / L)
        return d

    def _wrap(self, points: np.ndarray) -> np.ndarray:
        x = np.array(points, dtype=float, copy=True)
        for ax in range(3):
            if self._pbc[ax]:
                x[..., ax] -= self._box[ax] * np.floor(x[..., ax] / self._box[ax])
        return x

    def _build_pbc_tree(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n == 0:
            return None, np.array([], dtype=int)

        if not np.any(self._pbc):
            return cKDTree(pts), np.arange(n, dtype=int)

        shifts = []
        for ix in ([-1, 0, 1] if self._pbc[0] else [0]):
            for iy in ([-1, 0, 1] if self._pbc[1] else [0]):
                for iz in ([-1, 0, 1] if self._pbc[2] else [0]):
                    shifts.append(np.array([ix, iy, iz], dtype=float) * self._box)

        all_pts = []
        all_map = []
        base = np.arange(n, dtype=int)
        for s in shifts:
            all_pts.append(pts + s)
            all_map.append(base)

        all_pts = np.vstack(all_pts)
        all_map = np.concatenate(all_map)

        return cKDTree(all_pts), all_map

    def _build_neighbor_map(self, centers: np.ndarray, radii: np.ndarray, cutoff: float):
        if not np.any(self._pbc):
            return build_neighbor_map(centers, radii, cutoff)

        tree, idx_map = self._build_pbc_tree(centers)
        rmax = float(np.max(radii)) if len(radii) else 0.0

        neighbor_map: Dict[int, List[int]] = {i: [] for i in range(len(centers))}
        for i in range(len(centers)):
            r_search = float(radii[i] + rmax + cutoff)
            hits = tree.query_ball_point(centers[i], r=r_search)

            cand = set(int(idx_map[h]) for h in hits)
            cand.discard(i)

            # exact filter with MIC on centers
            out = []
            for j in cand:
                dc = centers[i] - centers[j]
                dc = self._mic(dc)
                dist = float(np.linalg.norm(dc))
                if dist < float(radii[i] + radii[j] + cutoff):
                    out.append(j)

            neighbor_map[i] = out

        return neighbor_map

    def _min_distance(
        self,
        coords_i: np.ndarray,
        neighbors_coords: Optional[List[np.ndarray]],
    ) -> float:
        
        # Distance to core
        core_tree = self._core_tree
        if core_tree is not None:
            dists, _ = core_tree.query(coords_i)
            d_core = float(np.min(dists))
        else:
            d_core = float("inf")

        if d_core < self.overlap_cutoff:
            return d_core
        
        if not neighbors_coords:
            return d_core
        
        # Distance to neighboring ligands
        neighbors = np.vstack(neighbors_coords)  # (N,3)
        if len(coords_i) * len(neighbors) > DENSE_PAIRS:
            # wrapped into the box, periodic images one box away hold every minimum image
            tree, image_of = self._build_pbc_tree(self._wrap(coords_i))
            _, k = tree.query(self._wrap(neighbors))
            diff = self._mic(coords_i[image_of[k]] - neighbors)
        else:
            diff = coords_i[:, None, :] - neighbors[None, :, :]  # (M,N,3)
            diff = self._mic(diff)
        d2 = np.sum(diff * diff, axis=-1)
        d_lig = float(np.sqrt(d2.min()))

        return min(d_core, d_lig)


    def _get_conflict_ligands(
        self,
        ligand_coords_list: List[np.ndarray],
        neighbor_map: Dict[int, List[int]],
    ) -> Tuple[float, List[int], np.ndarray]:

        n_lig = len(ligand_coords_list)
        conflict_ligs = []
        dists = np.empty(n_lig, dtype=float)
        global_min = float("inf")

        for i in range(n_lig):
            coords_i = ligand_coords_list[i]
            idxs = neighbor_map.get(i, [])

            neighbors_coords = [ligand_coords_list[j] for j in idxs] if idxs else None

            d_i = self._min_distance(
                coords_i,
                neighbors_coords=neighbors_coords,
            )
            dists[i] = d_i
            global_min = min(global_min, d_i)

            if d_i < self.overlap_cutoff:
                conflict_ligs.append(i)

        return global_min, conflict_ligs, dists

    def _build_active_cluster(
        self,
        conflict_ligs: List[int],
        site_positions: np.ndarray,
    ) -> List[int]:

        radius = self.active_radius_factor * self.core.a + 1e-2

        if not conflict_ligs:
            return []

        site_tree = cKDTree(site_positions)

        active = set(conflict_ligs)  # start from conflict ligands

        # BFS-style cluster expansion
        queue = list(conflict_ligs)
        visited = set(conflict_ligs)

        while queue:
            i = queue.pop()
            pos_i = site_positions[i]

            neighbor_js = site_tree.query_ball_point(pos_i, r=radius)
            for j in neighbor_js:
                if j in visited:
                    continue
                visited.add(j)
                active.add(j)
                queue.append(j)

        return sorted(list(active))

    def _place_one_ligand(
        self,
        ligand: Ligand,
        plane: np.ndarray,
        site_pos: np.ndarray,
        theta: float,
        x_move: float = None,
        y_move: float = None,
        z_move: float = None
    ) -> np.ndarray:

        coords_final = place_about_normal(
            ligand.atoms.get_positions(),
            plane,
            site_pos,
            getattr(ligand, "_anchor_offset", 0.0),
            theta,
        )

        if x_move is not None:
            coords_final[:, 0] += x_move
        if y_move is not None:
            coords_final[:, 1] += y_move
        if z_move is not None:
            coords_final[:, 2] += z_move

        return coords_final


    def _sampler_geometry(self):
        """
        Periodic tree builder and minimum-image convention for the constellation sampler.
        """
        return self._build_pbc_tree, self._mic


    def _exact_min_distance(self, coords: np.ndarray, neighbors_coords: List[np.ndarray]) -> float:
        """
        Closest contact of `coords` with the core or any of `neighbors_coords`.
        """
        return tree_min_distance(self._core_tree, self._sampler_geometry()[0], self.overlap_cutoff,
                                 neighbors_coords, early_core=False)(coords)


    def _optimize_rotation(
        self,
        pose: SitePose | FitPose,
        neighbors_coords: List[np.ndarray],
    ) -> Tuple[float, int, np.ndarray]:
        min_distance = self._min_distance
        if isinstance(pose, FitPose):
            min_distance = tree_min_distance(self._core_tree, self._sampler_geometry()[0],
                                             self.overlap_cutoff, neighbors_coords, early_core=False)
        return optimize_pose(
            pose,
            neighbors_coords,
            min_distance,
            self._rng,
            self.overlap_cutoff,
            self.coarse_step_deg,
            self.fine_step_deg,
            self.window_deg,
        )


    def _assign_ligands_to_octahedra(self) -> None:
        if not self.octahedra:
            return

        # Fresh entries rather than clearing in place: frames loaded together by from_ase share one octahedra dict.
        octahedra = {b_abs: {**octa, "Ligand": []} for b_abs, octa in self.octahedra.items()}
        self.core.octahedra = octahedra

        at = self.core.atoms
        syms = np.array(at.get_chemical_symbols())
        b_idx = np.where(syms == self.core.B)[0]

        cell = at.get_cell()
        Lx, Ly, Lz = cell.lengths()

        scaled = at.get_scaled_positions(wrap=True)
        pos = scaled @ cell.array
        B_pos = pos[b_idx]

        anchor_positions = []
        ligand_ids = []

        for lig in self.ligands:
            if lig.n_motifs > 1:
                # every motif bound to an X or B site belongs to the octahedron there
                for motif, pos in zip(lig.binding_motifs, lig.anchor_positions):
                    if motif.site in ("X", "B"):
                        anchor_positions.append(pos)
                        ligand_ids.append(int(lig.id))
            elif lig.charge <= 0 and getattr(lig, "anchor_pos", None) is not None:
                anchor_positions.append(np.asarray(lig.anchor_pos, dtype=float))
                ligand_ids.append(int(lig.id))

        if anchor_positions:
            anchor_positions = np.vstack(anchor_positions)

            b_tree = cKDTree(B_pos, boxsize=(Lx, Ly, Lz))
            _, b_loc = b_tree.query(anchor_positions, k=1)

            for j, b_loc_j in enumerate(b_loc):
                b_abs = int(b_idx[int(b_loc_j)])
                if ligand_ids[j] not in octahedra[b_abs]["Ligand"]:
                    octahedra[b_abs]["Ligand"].append(ligand_ids[j])


    def _build_index_map(self) -> None:

        n_core = len(self.core.atoms)

        cursor = n_core
        for lig in self.ligands:
            n = len(lig.atoms)
            idx = np.arange(cursor, cursor + n, dtype=int)
            lig.indices = idx
            cursor += n


    def to(
        self, 
        fmt='vasp', 
        filename: Optional[str] = None, 
        write_json: bool = True,
        **kwargs
    ) -> None:      
        at = self.atoms
        formula = at.get_chemical_formula()

        if filename is None:
            filename = f"{formula}.{fmt}"
        
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)  
        if fmt == 'vasp':
            ase_kwargs = {**kwargs, "sort": True}
            write_vasp(str(path), self.atoms, **ase_kwargs)
            if write_json:
                self.to_json(str(path) + ".json", sort = True)
        else:
            ase_kwargs = {**kwargs, "format": fmt, "comment": formula}
            write(str(path), self.atoms, **ase_kwargs)
            if write_json:
                self.to_json(str(path) + ".json", sort = False)


    def to_json(self, filename: str, sort: bool = False) -> None:

        self._build_index_map()

        core_meta = {
            "A": self.core.a_site_metadata(),
            "B": self.core.B,
            "X": self.core.X,
            "a": self.core.a,
            "supercell": list(self.core.supercell),
            "cell": self.core.atoms.get_cell().array.tolist(),
            "pbc": [bool(x) for x in self.core.atoms.pbc],
            "vacuum": float(self.core.vacuum),
            "n_core_atoms": len(self.core.atoms),
        }

        ligand_types_meta = []
        type_key_to_id: Dict[tuple, int] = {}
        type_counts: Dict[int, int] = {}

        for lig in self.ligands:
            key = lig._type_key()

            if key not in type_key_to_id:
                type_id = len(ligand_types_meta)
                type_key_to_id[key] = type_id
                type_counts[type_id] = 0

                ligand_types_meta.append(
                    {
                        "id": type_id,
                        **lig._type_metadata(),
                        "coverage": self.ligand_coverage.get(lig.name),
                        "n_atoms": len(lig.atoms),
                        "volume": float(lig.volume),
                    }
                )

            type_id = type_key_to_id[key]
            type_counts[type_id] += 1

        for meta in ligand_types_meta:
            meta["n_instances"] = type_counts[meta["id"]]

        core_indices_meta = {"octahedra": self.octahedra,
                             "B_ijk": self.B_ijk
                             }

        ligands_meta = []
        for i, lig in enumerate(self.ligands):
            key = lig._type_key()
            spec_id = type_key_to_id[key]

            ligands_meta.append(
                {   
                    "ligand_id": i,
                    "spec_id": spec_id,
                    **lig._instance_metadata(),
                }
            )

        n_total_atoms = core_meta["n_core_atoms"] + sum(
            t["n_atoms"] * t["n_instances"] for t in ligand_types_meta
        )

        topo = {
            "schema_version": 1,
            "type": "slab",
            "n_total_atoms": n_total_atoms,
            "core": core_meta,
            "ligand_types": ligand_types_meta,
            "core_indices": core_indices_meta,
            "ligands": ligands_meta
        }

        if sort:
            topo["sort_idx"] = np.argsort(self.atoms.symbols).tolist()

        with open(filename, "w") as f:
            json.dump(topo, f, indent=2)


    @classmethod
    def from_file(cls, filename: str, json_filename: str) -> Slab:

        atoms = read(filename)

        with open(json_filename) as f:
            topo = json.load(f)

        schema_version = topo.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {schema_version!r}")

        file_type = topo.get("type")
        if file_type is not None and file_type != "slab":
            raise ValueError(
                f"JSON type is '{file_type}', expected 'slab'. "
                f"Use NanoCrystal.from_file() for nanocrystal data."
            )

        sort_idx = topo.get("sort_idx", None)
        if sort_idx is not None:
            atoms = atoms[np.argsort(sort_idx)]

        n_total_atoms_json = topo["n_total_atoms"]
        if n_total_atoms_json != len(atoms):
            raise ValueError(
                f"Atom count mismatch: JSON={n_total_atoms_json}, XYZ={len(atoms)}."
            )

        core_meta = topo["core"]
        n_core_atoms = core_meta["n_core_atoms"]

        if n_core_atoms > len(atoms):
            raise ValueError(
                f"n_core_atoms={n_core_atoms}, but XYZ has only {len(atoms)} atoms."
            )

        core_atoms = atoms[:n_core_atoms]
        core_atoms.set_cell(np.array(core_meta["cell"], dtype=float))
        core_atoms.set_pbc(np.array(core_meta["pbc"], dtype=bool))

        core_indices_meta = topo["core_indices"]

        octa_raw = core_indices_meta.get("octahedra", None)
        X = str(core_meta["X"])
        if octa_raw is not None:
            octahedra = {
                int(b): {"X": v["X"], "Ligand": v["Ligand"]}
                for b, v in octa_raw.items()
            }
        else:
            octahedra = None

        B_ijk_raw = core_indices_meta.get("B_ijk")
        B_ijk = {int(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in B_ijk_raw.items()}

        core = Core(
            A=Core.a_site_from_json(core_meta["A"], core_atoms),
            B=core_meta["B"],
            X=core_meta["X"],
            atoms=core_atoms,
            a=core_meta["a"],
            supercell=tuple(core_meta["supercell"]),
            vacuum=core_meta["vacuum"],
            build_surface=False,
        )


        ligand_types_meta = topo["ligand_types"]
        type_id_to_meta: Dict[int, dict] = {
            t["id"]: t for t in ligand_types_meta
        }

        ligands_meta = topo["ligands"]
        ligands: List[Ligand] = []
        cursor = n_core_atoms

        for inst_meta in ligands_meta:
            spec_id = inst_meta["spec_id"]
            tmeta = type_id_to_meta[spec_id]

            n_atoms = tmeta["n_atoms"]

            if cursor + n_atoms > len(atoms):
                raise ValueError(
                    f"Ligand slice out of range: need up to {cursor + n_atoms}, "
                    f"but XYZ has only {len(atoms)} atoms."
                )

            lig_atoms = atoms[cursor : cursor + n_atoms]
            cursor += n_atoms

            binding_atoms = list(map(int, tmeta.get("binding_atoms_indices", [])))

            lig = Ligand._from_data(
                atoms=lig_atoms,
                mol=None,
                smiles=tmeta["smiles"],
                charge=tmeta["charge"],
                binding_motif=BindingMotif._from_metadata(tmeta),
                name=tmeta["name"],
                id=inst_meta["ligand_id"],
                planes=Ligand._planes_from_metadata(inst_meta),
                volume=tmeta["volume"],
                binding_atoms=binding_atoms,
            )

            ligands.append(lig)

        slab = cls(
            core=core,
            ligand_specs=[],
        )
        slab.ligands = ligands
        slab.ligand_coverage = {t["name"]: t["coverage"] for t in ligand_types_meta}
        slab.core.octahedra = octahedra
        slab.core.B_ijk = B_ijk
        slab._build_index_map()
        slab._build_and_link_atoms()

        return slab

    @classmethod
    def from_ase(cls,
                 atoms: Atoms | List[Atoms],
                 json_filename: str
                 ) -> Slab | List[Slab]:

        with open(json_filename) as f:
            topo = json.load(f)

        schema_version = topo.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {schema_version!r}")

        file_type = topo.get("type")
        if file_type is not None and file_type != "slab":
            raise ValueError(
                f"JSON type is '{file_type}', expected 'slab'. "
                f"Use NanoCrystal.from_file() for nanocrystal data."
            )

        single_input = isinstance(atoms, Atoms)
        atoms_list: List[Atoms] = [atoms] if single_input else list(atoms)

        sort_idx = topo.get("sort_idx", None)
        n_total_atoms_json = topo["n_total_atoms"]

        core_meta = topo["core"]
        n_core_atoms = core_meta["n_core_atoms"]
        core_cell = np.array(core_meta["cell"], dtype=float)
        core_pbc = np.array(core_meta["pbc"], dtype=bool)

        core_indices_meta = topo["core_indices"]

        octa_raw = core_indices_meta.get("octahedra", None)
        if octa_raw is not None:
            octahedra = {
                int(b): {"X": v["X"], "Ligand": v["Ligand"]}
                for b, v in octa_raw.items()
            }
        else:
            octahedra = None

        B_ijk_raw = core_indices_meta.get("B_ijk")
        B_ijk = {int(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in B_ijk_raw.items()}

        ligand_types_meta = topo["ligand_types"]
        type_id_to_meta: Dict[int, dict] = {
            t["id"]: t for t in ligand_types_meta
        }
        ligands_meta = topo["ligands"]

        # Everything that is the same for every frame is resolved once, outside the loop.
        order = np.argsort(sort_idx) if sort_idx is not None else None
        motifs = {t["id"]: BindingMotif._from_metadata(t) for t in ligand_types_meta}

        if n_core_atoms > n_total_atoms_json:
            raise ValueError(
                f"n_core_atoms={n_core_atoms}, but JSON has only {n_total_atoms_json} atoms."
            )

        lig_slices: List[slice] = []
        cursor = n_core_atoms
        for inst_meta in ligands_meta:
            n_atoms = type_id_to_meta[inst_meta["spec_id"]]["n_atoms"]
            if cursor + n_atoms > n_total_atoms_json:
                raise ValueError(
                    f"Ligand slice out of range: need up to {cursor + n_atoms}, "
                    f"but JSON has only {n_total_atoms_json} atoms."
                )
            lig_slices.append(slice(cursor, cursor + n_atoms))
            cursor += n_atoms

        slabs: List[Slab] = []
        for atoms_i in atoms_list:
            if n_total_atoms_json != len(atoms_i):
                raise ValueError(
                    f"Atom count mismatch: JSON={n_total_atoms_json}, XYZ={len(atoms_i)}."
                )

            # Split plain arrays instead of slicing atoms_i: every ASE slice copies all
            # arrays and re-indexes the constraints, which dominated the per-frame cost.
            numbers = atoms_i.numbers
            positions = atoms_i.positions
            if order is not None:
                numbers = numbers[order]
                positions = positions[order]

            core_atoms = Atoms(
                numbers=numbers[:n_core_atoms],
                positions=positions[:n_core_atoms],
                pbc=core_pbc,
                cell=core_cell,
            )

            core = Core(
                A=Core.a_site_from_json(core_meta["A"], core_atoms),
                B=core_meta["B"],
                X=core_meta["X"],
                atoms=core_atoms,
                a=core_meta["a"],
                supercell=tuple(core_meta["supercell"]),
                vacuum=core_meta["vacuum"],
                build_surface=False,
            )

            ligands: List[Ligand] = []

            for inst_meta, sl in zip(ligands_meta, lig_slices):
                spec_id = inst_meta["spec_id"]
                tmeta = type_id_to_meta[spec_id]

                lig_atoms = Atoms(
                    numbers=numbers[sl],
                    positions=positions[sl],
                    pbc=atoms_i.pbc,
                    cell=atoms_i.cell,
                )

                binding_atoms = list(map(int, tmeta.get("binding_atoms_indices", [])))

                lig = Ligand._from_data(
                    atoms=lig_atoms,
                    mol=None,
                    smiles=tmeta["smiles"],
                    charge=tmeta["charge"],
                    binding_motif=motifs[spec_id],
                    name=tmeta["name"],
                    id=inst_meta["ligand_id"],
                    planes=Ligand._planes_from_metadata(inst_meta),
                    volume=tmeta["volume"],
                    binding_atoms=binding_atoms,
                )

                ligands.append(lig)

            slab = cls._from_data(core=core)
            slab.ligands = ligands
            slab.ligand_coverage = {t["name"]: t["coverage"] for t in ligand_types_meta}
            slab.core.octahedra = octahedra
            slab.core.B_ijk = B_ijk
            slab._build_index_map()
            slab._build_and_link_atoms()

            slabs.append(slab)

        return slabs[0] if single_input else slabs


    @classmethod
    def _from_data(cls, core: Core) -> Slab:
        """
        Construct a Slab around `core` bypassing __post_init__, which deep-copies it.
        Used when deserializing from the metadata, where the core is already private.
        """
        slab = object.__new__(cls)
        slab.core = core
        slab.ligand_specs = []
        slab.ligands = []
        slab.ligand_coverage = {}
        slab.binding_sites = []
        slab._atoms = None
        slab._setup_pbc()
        return slab


    def _build_and_link_atoms(self) -> None:
        core_atoms = self.core.atoms
        all_numbers = [core_atoms.numbers]
        all_positions = [core_atoms.positions]

        for lig in self.ligands:
            all_numbers.append(lig.atoms.numbers)
            all_positions.append(lig.atoms.positions)

        self._atoms = Atoms(
            numbers=np.concatenate(all_numbers),
            positions=np.vstack(all_positions),
            pbc=core_atoms.pbc,
            cell=core_atoms.get_cell(),
        )

        # Link position arrays: make core and ligand position arrays
        # be numpy views into self._atoms's position buffer.
        shared = self._atoms.arrays['positions']
        n_core = len(core_atoms)
        self.core.atoms.arrays['positions'] = shared[:n_core]

        cursor = n_core
        for lig in self.ligands:
            n = len(lig.atoms)
            lig.atoms.arrays['positions'] = shared[cursor:cursor + n]
            cursor += n

    @property
    def atoms(self) -> Atoms:
        if self._atoms is not None:
            return self._atoms

        # Before ligands are placed, assemble on the fly (core only).
        core_atoms = self.core.atoms
        return Atoms(
            symbols=core_atoms.get_chemical_symbols(),
            positions=core_atoms.get_positions(),
            pbc=core_atoms.pbc,
            cell=core_atoms.get_cell(),
        )