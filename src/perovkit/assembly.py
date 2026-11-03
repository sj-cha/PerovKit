from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import List, Dict, Optional, Tuple, Sequence
import json
from pathlib import Path

import random
import math
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.io.vasp import write_vasp

from scipy.spatial import cKDTree 

from .core import Core, BindingSite
from .ligand import Ligand, LigandSpec, BindingMotif
from .constellation import ConstellationSampler
from .refine import INTRA_SCALE
from .placement import (SitePose, FitPose, Contacts, pack, place_about_normal, motif_pools,
                        select_constellations, intramolecular_contacts)
from .utils.geometry import farthest_point_sampling, per_face_farthest_point_sampling, build_neighbor_map


@dataclass
class Assembly:
    """
    A Core with ligands on its surface: a nanocrystal, or a slab when the Core is periodic
    (Core.build_core(..., pbc=(True, True, False))). KD-tree searches follow the core's pbc:
    minimum-image distances along its periodic axes. A slab keeps its cell.
    Made by Core.place_ligands and Core.place_ligands_manually, or read back with
    from_file and from_ase.
    """
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


    def _place_ligands(
        self,
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
    ) -> None:

        assert not self.ligands, "Ligands have already been placed."

        log = print if verbose else (lambda *a, **k: None)

        self._rng = random.Random(random_seed)
        self.overlap_cutoff = overlap_cutoff
        
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
                        max_rmsd=max_anchor_rmsd, sites=self.binding_sites, **sampler_options,
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
                sc = self.core.supercell
                if self.core.is_nanocrystal and sc is not None and len(set(sc)) > 1:
                    planes = [s.plane for s in available]
                    chosen_indices = per_face_farthest_point_sampling(coords, planes, n_target, self._rng)
                else:
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

        contacts = Contacts(core_tree, *self._sampler_geometry(), self._wrap,
                            self._build_neighbor_map, overlap_cutoff)
        ligand_coords_list = pack(
            poses, contacts, self._rng, max_iters=max_iters, coarse_step_deg=coarse_step_deg,
            fine_step_deg=fine_step_deg, window_deg=window_deg,
            active_radius=active_radius_factor * self.core.a + 1e-2, max_rmsd=max_anchor_rmsd,
            refine_torsions=refine_torsions, stretch_tails=stretch_tails, n_workers=n_workers,
            log=log, verbose=verbose,
        )

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
            pbc=False,
        )
        if old.is_slab:
            stripped_atoms.set_pbc(True)
            stripped_atoms.set_cell(old.atoms.get_cell())
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


    def _place_ligands_manually(
        self,
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
    ):
        """
        Place every ligand at its site without the rotation search, cations shifted by
        cat_x/y/z_move (Å) and turned by cat_z_rotate (degrees), anions by the an_* values.
        With `ligand_ids`, only those ligands (ids count from 0 in placement order) are
        shifted and turned; the rest sit at their sites unmoved.
        """
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

        moved = set(range(n_lig)) if ligand_ids is None else {int(i) for i in ligand_ids}
        if moved - set(range(n_lig)):
            print(f"[Warning] No ligands with ids {sorted(moved - set(range(n_lig)))}; "
                  f"ids run from 0 to {n_lig - 1}.")

        ligand_coords_list: List[np.ndarray] = []
        for i in range(n_lig):
            if i not in moved:
                x_move = y_move = z_move = z_rotate = 0.0
            elif ligands[i].charge > 0:
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
            pbc=False,
        )
        if old.is_slab:
            stripped_atoms.set_pbc(True)
            stripped_atoms.set_cell(old.atoms.get_cell())
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
        Tree builder and minimum-image convention for the constellation sampler: periodic
        for a slab, plain (identity) for a nanocrystal.
        """
        if self.core.is_nanocrystal:
            return (lambda pts: (cKDTree(pts), np.arange(len(pts)))), (lambda d: d)
        return self._build_pbc_tree, self._mic


    def _assign_ligands_to_octahedra(self) -> None:
        if not self.octahedra:
            return

        # Fresh entries rather than clearing in place: frames loaded together by from_ase share one octahedra dict.
        octahedra = {b_abs: {**octa, "Ligand": []} for b_abs, octa in self.octahedra.items()}
        self.core.octahedra = octahedra

        at = self.core.atoms
        syms = np.array(at.get_chemical_symbols())
        b_idx = np.where(syms == self.core.B)[0]

        if self.core.is_slab:
            cell = at.get_cell()
            boxsize = tuple(cell.lengths())
            pos = at.get_scaled_positions(wrap=True) @ cell.array
        else:
            boxsize = None
            pos = at.get_positions()
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

            b_tree = cKDTree(B_pos, boxsize=boxsize)
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
        fmt: Optional[str] = None, 
        filename: Optional[str] = None, 
        write_json: bool = True,
        vacuum: float = 15.0,
        **kwargs
    ) -> None:      
        """
        Write the structure and, with `write_json`, its metadata to `filename`.json.
        `fmt` defaults to "vasp" for a slab and "xyz" for a nanocrystal; a nanocrystal
        written as vasp is boxed with `vacuum` Å around it.
        """
        at = self.atoms
        formula = at.get_chemical_formula()
        if fmt is None:
            fmt = "vasp" if self.core.is_slab else "xyz"

        if filename is None:
            filename = f"{formula}.{fmt}"
        
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)  
        if fmt == 'vasp':
            ase_kwargs = {**kwargs, "sort": True}
            vasp_atoms = at
            if self.core.is_nanocrystal:
                pos = at.get_positions()
                center = pos.mean(axis=0)
                extent = pos.max(axis=0) - pos.min(axis=0)
                cell_diag = extent + vacuum

                vasp_atoms = at.copy()
                vasp_atoms.set_cell(np.diag(cell_diag))
                vasp_atoms.positions += (cell_diag / 2 - center)
                vasp_atoms.pbc = True
            write_vasp(str(path), vasp_atoms, **ase_kwargs)
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
        }
        if self.core.is_slab:
            core_meta["cell"] = self.core.atoms.get_cell().array.tolist()
            core_meta["pbc"] = [bool(x) for x in self.core.atoms.pbc]
            core_meta["vacuum"] = float(self.core.vacuum)
        core_meta["n_core_atoms"] = len(self.core.atoms)

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
            "type": "slab" if self.core.is_slab else "nanocrystal",
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
    def from_file(cls, filename: str, json_filename: str) -> Assembly:

        atoms = read(filename)

        with open(json_filename) as f:
            topo = json.load(f)

        schema_version = topo.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {schema_version!r}")

        file_type = topo.get("type")
        if file_type not in (None, "nanocrystal", "slab"):
            raise ValueError(
                f"JSON type is '{file_type}', expected 'nanocrystal' or 'slab'."
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
        if "cell" in core_meta:
            core_atoms.set_cell(np.array(core_meta["cell"], dtype=float))
            core_atoms.set_pbc(np.array(core_meta["pbc"], dtype=bool))
        else:
            core_atoms.pbc = False

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
            vacuum=core_meta.get("vacuum"),
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

        assembly = cls(
            core=core,
            ligand_specs=[],
        )
        assembly.ligands = ligands
        assembly.ligand_coverage = {t["name"]: t["coverage"] for t in ligand_types_meta}
        assembly.core.octahedra = octahedra
        assembly.core.B_ijk = B_ijk
        assembly._build_index_map()
        assembly._build_and_link_atoms()

        return assembly

    @classmethod
    def from_ase(cls,
                 atoms: Atoms | List[Atoms],
                 json_filename: str
                 ) -> Assembly | List[Assembly]:

        with open(json_filename) as f:
            topo = json.load(f)

        schema_version = topo.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {schema_version!r}")

        file_type = topo.get("type")
        if file_type not in (None, "nanocrystal", "slab"):
            raise ValueError(
                f"JSON type is '{file_type}', expected 'nanocrystal' or 'slab'."
            )

        single_input = isinstance(atoms, Atoms)
        atoms_list: List[Atoms] = [atoms] if single_input else list(atoms)

        sort_idx = topo.get("sort_idx", None)
        n_total_atoms_json = topo["n_total_atoms"]

        core_meta = topo["core"]
        n_core_atoms = core_meta["n_core_atoms"]
        periodic = "cell" in core_meta
        if periodic:
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

        assemblies: List[Assembly] = []
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
                pbc=core_pbc if periodic else False,
                cell=core_cell if periodic else atoms_i.cell,
            )

            core = Core(
                A=Core.a_site_from_json(core_meta["A"], core_atoms),
                B=core_meta["B"],
                X=core_meta["X"],
                atoms=core_atoms,
                a=core_meta["a"],
                supercell=tuple(core_meta["supercell"]),
                vacuum=core_meta.get("vacuum"),
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

            assembly = cls._from_data(core=core)
            assembly.ligands = ligands
            assembly.ligand_coverage = {t["name"]: t["coverage"] for t in ligand_types_meta}
            assembly.core.octahedra = octahedra
            assembly.core.B_ijk = B_ijk
            assembly._build_index_map()
            assembly._build_and_link_atoms()

            assemblies.append(assembly)

        return assemblies[0] if single_input else assemblies


    @classmethod
    def _from_data(cls, core: Core) -> Assembly:
        """
        Construct an Assembly around `core` bypassing __post_init__, which deep-copies it.
        Used when deserializing from the metadata, where the core is already private.
        """
        assembly = object.__new__(cls)
        assembly.core = core
        assembly.ligand_specs = []
        assembly.ligands = []
        assembly.ligand_coverage = {}
        assembly.binding_sites = []
        assembly._atoms = None
        assembly._setup_pbc()
        return assembly


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