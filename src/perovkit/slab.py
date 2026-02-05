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
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v
from .utils.geometry import farthest_point_sampling, compute_bounding_spheres, build_neighbor_map


@dataclass
class Slab:
    core: Core
    ligand_specs: List[LigandSpec]
    random_seed: int = 42
    ligands: List[Ligand] = field(default_factory=list)
    ligand_coverage: Dict[str, float] = field(default_factory=dict)
    octahedra: Optional[Dict[int, dict]] = None

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
        self._rng = random.Random(self.random_seed)
        self.ligand_specs.sort(
            key=lambda spec: (
                0 if spec.ligand.charge > 0 else 1, 
                -spec.ligand.volume,                
            )
        )
        self._setup_pbc()


    def place_ligands(
        self,
        max_iters: int = 10,
        overlap_cutoff: float = 2.0,   # Å
        coarse_step_deg: int = 18,
        fine_step_deg: int = 2,
        window_deg: int = 12,
        active_radius_factor: float = math.sqrt(2)/2
    ) -> None:
        
        assert not self.ligands, "Ligands have already been placed."

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

        A_sites = [s for s in self.binding_sites if s.symbol == self.core.A]
        X_sites = [s for s in self.binding_sites if s.symbol == self.core.X]

        ligands: List[Ligand] = []
        sites: List[BindingSite] = []

        for spec in self.ligand_specs:
            lig = spec.ligand
            charge = lig.charge

            if charge > 0:
                pool = A_sites
                ligand_type = "A-site"
            else: 
                pool = X_sites
                ligand_type = "X-site"

            available = [s for s in pool if not s.passivated]
            
            if spec.binding_sites is not None:
                chosen_sites = []
                for idx in spec.binding_sites:
                    site = next((s for s in available if s.index == idx), None)
                    if site is not None:
                        chosen_sites.append(site)
                    else:
                        print(f"[Warning] Requested binding site index {idx} not available for ligand {lig.name}.")
            else:
                if 0.0 < spec.coverage <= 1.0:
                    n_target = int(math.ceil(spec.coverage * len(available)))
                else:
                    n_target = int(spec.coverage)
                    if n_target > len(available):
                        print(f"[Warning] Requested {n_target} sites for ligand, "
                            f"but only {len(available)} available.")
                        n_target = len(available)

                coords = np.array([core_positions[s.index] for s in available])
                chosen_indices = farthest_point_sampling(coords, n_target, self._rng)
                chosen_sites = [available[i] for i in chosen_indices]

            for site in chosen_sites:
                lig_cloned = lig.clone()
                lig_cloned.plane = site.plane
                lig_cloned._anchor_offset = float(spec.anchor_offset)
                site.passivated = True

                ligands.append(lig_cloned)
                sites.append(site)
                displaced_indices.append(site.index)

            print(f"[Log] Placed total {len(chosen_sites)} {ligand_type} ligands.")
            self.ligand_coverage[spec.ligand.name] = spec.coverage

        n_lig = len(ligands)
        assert n_lig > 0, "No ligands to place."

        # Binding site positions and planes
        site_positions = np.array([core_positions[s.index] for s in sites])
        site_planes = np.array([s.plane for s in sites], dtype=float)

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

        # Initialize rotation angles around surface normal and current coordinates
        thetas = np.array([2.0 * math.pi * self._rng.random() for _ in range(n_lig)], dtype=float)
        ligand_coords_list: List[np.ndarray] = []
        for i in range(n_lig):
            coords_i = self._place_one_ligand(
                ligands[i],
                site_planes[i],
                site_positions[i],
                thetas[i],
            )
            ligand_coords_list.append(coords_i)

        # Rotation optimization loop
        centers, radii = compute_bounding_spheres(ligand_coords_list)
        neighbor_map = self._build_neighbor_map(centers, radii, self.overlap_cutoff)

        global_min, conflict_ligs, dists = self._get_conflict_ligands(
            ligand_coords_list,
            neighbor_map,
        )
        print(f"[Log] Initial global_min = {global_min:.3f} Å")

        for iter in range(1, max_iters + 1):
            if global_min >= self.overlap_cutoff:
                print(
                    f"[Log] Hard cutoff {self.overlap_cutoff:.3f} Å satisfied. Stopping optimization."
                )
                break

            active_cluster = self._build_active_cluster(conflict_ligs, site_positions)

            improved = False
            self._rng.shuffle(active_cluster)

            for i in tqdm(
                active_cluster,
                desc=f"Optimizing ligand (iter {iter})",
            ):
                neighbor_idx = neighbor_map.get(i, [])
                neighbors_coords = [ligand_coords_list[j] for j in neighbor_idx]

                d_old = dists[i]

                best_theta, best_coords = self._optimize_rotation(
                    i,
                    ligands,
                    neighbors_coords,
                    site_planes,
                    site_positions,
                )
                d_new = self._min_distance(best_coords, neighbors_coords)

                if d_new > d_old + 1e-3:
                    ligand_coords_list[i] = best_coords
                    thetas[i] = best_theta
                    improved = True

            if not improved:
                raise RuntimeError("[Log] No improvement in active cluster. Stopping.")

            centers, radii = compute_bounding_spheres(ligand_coords_list)
            neighbor_map = self._build_neighbor_map(centers, radii, self.overlap_cutoff)
            global_min, conflict_ligs, dists = self._get_conflict_ligands(
                ligand_coords_list,
                neighbor_map,
            )
            print(f"[Log] Iter {iter}  global_min = {global_min:.3f} Å")

        if global_min < self.overlap_cutoff:
            raise RuntimeError("[Error] Maximum iterations reached without satisfying overlap cutoff.")

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
            pbc=old.atoms.pbc,
            cell=old.atoms.get_cell(),
        )

        stripped_atoms.cell[2, 2] = max_z + old.vacuum
        
        self.core = Core(
            A=old.A, 
            B=old.B, 
            X=old.X,
            atoms=stripped_atoms,
            a=old.a,
            supercell=old.supercell,
            vacuum=old.vacuum,
            build_surface=False,
        )

        self._build_octahedra()
        self._build_B_ijk()
        self._build_index_map()


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
    ):
        assert not self.ligands, "Ligands have already been placed."
        
        self.ligands = []
        displaced_indices = []
        for s in self.binding_sites:
            s.passivated = False

        core_positions = self.core.atoms.get_positions()

        A_sites = [s for s in self.binding_sites if s.symbol == self.core.A]
        X_sites = [s for s in self.binding_sites if s.symbol == self.core.X]

        ligands: List[Ligand] = []
        sites: List[BindingSite] = []

        for spec in self.ligand_specs:
            lig = spec.ligand
            charge = lig.charge

            if charge > 0:
                pool = A_sites
                ligand_type = "A-site"
            else: 
                pool = X_sites
                ligand_type = "X-site"

            available = [s for s in pool if not s.passivated]
            
            if spec.binding_sites is not None:
                chosen_sites = []
                for idx in spec.binding_sites:
                    site = next((s for s in available if s.index == idx), None)
                    if site is not None:
                        chosen_sites.append(site)
                    else:
                        print(f"[Warning] Requested binding site index {idx} not available for ligand {lig.name}.")
            else:
                if 0.0 < spec.coverage <= 1.0:
                    n_target = int(math.ceil(spec.coverage * len(available)))
                else:
                    n_target = int(spec.coverage)
                    if n_target > len(available):
                        print(f"[Warning] Requested {n_target} sites for ligand, "
                            f"but only {len(available)} available.")
                        n_target = len(available)

                coords = np.array([core_positions[s.index] for s in available])
                chosen_indices = farthest_point_sampling(coords, n_target, self._rng)
                chosen_sites = [available[i] for i in chosen_indices]

            for site in chosen_sites:
                lig_cloned = lig.clone()
                lig_cloned.plane = site.plane
                lig_cloned._anchor_offset = float(spec.anchor_offset)
                site.passivated = True

                ligands.append(lig_cloned)
                sites.append(site)
                displaced_indices.append(site.index)

            print(f"[Log] Placed total {len(chosen_sites)} {ligand_type} ligands.")
            self.ligand_coverage[spec.ligand.name] = spec.coverage

        n_lig = len(ligands)
        assert n_lig > 0, "No ligands to place."

        # Binding site positions and planes
        site_positions = np.array([core_positions[s.index] for s in sites])
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
            pbc=old.atoms.pbc,
            cell=old.atoms.get_cell(),
        )

        stripped_atoms.cell[2, 2] = max_z + old.vacuum
        
        self.core = Core(
            A=old.A, 
            B=old.B, 
            X=old.X,
            atoms=stripped_atoms,
            a=old.a,
            supercell=old.supercell,
            vacuum=old.vacuum,
            build_surface=False,
        )

        self._build_octahedra()
        self._build_B_ijk()
        self._build_index_map()


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


    def check_overlaps(self, cutoff: float = None) -> float:
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
            for j in range(i + 1, len(entities)):
                A = entities[i]
                B = entities[j]
                diff = A[:, None, :] - B[None, :, :]
                diff = self._mic(diff)
                d2 = np.sum(diff * diff, axis=-1)
                min_idx = np.argmin(d2)
                d_min = float(np.sqrt(d2.flat[min_idx]))

                if d_min < global_min:
                    global_min = d_min
                    global_pair = (i, j)

                if d_min < cutoff:
                    print(
                        f"[OVERLAP] {entity_labels[i]} vs {entity_labels[j]}: "
                        f"min distance = {d_min:.3f} Å < {cutoff:.3f} Å"
                    )
                    n += 1

        print(
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

        coords_loc = ligand.atoms.get_positions()

        # Rotate to surface normal
        R_align = rotation_from_u_to_v(
            np.array([0.0, 0.0, 1.0], dtype=float),
            plane,
        )
        coords_aligned = coords_loc @ R_align.T

        # Rotate around surface normal 
        R_rot = rotation_about_axis(plane, theta)
        coords_rot = coords_aligned @ R_rot.T

        anchor_offset = plane / np.linalg.norm(plane) * float(getattr(ligand, "_anchor_offset", 0.0))
        anchor_pos = site_pos + anchor_offset

        if getattr(ligand, "anchor_pos", None) is None:
            ligand.anchor_pos = anchor_pos

        # Translate to binding site position
        coords_final = coords_rot + anchor_pos
        
        if x_move is not None:
            coords_final[:, 0] += x_move
        if y_move is not None:
            coords_final[:, 1] += y_move
        if z_move is not None:
            coords_final[:, 2] += z_move

        return coords_final


    def _optimize_rotation(
        self,
        i: int,
        ligands: List[Ligand],
        neighbors_coords: List[np.ndarray],
        site_planes: np.ndarray, 
        site_positions: np.ndarray,
    ) -> Tuple[float, np.ndarray]:

        cutoff = self.overlap_cutoff
        ligand_i = ligands[i]

        # Search parameters
        coarse_step_deg = self.coarse_step_deg
        fine_step_deg = self.fine_step_deg
        window_deg = self.window_deg

        # Initialization
        best_theta = None
        best_min_d = -float("inf")
        best_coords_i = None
        theta0 = 2.0 * math.pi * self._rng.random()
        
        # Coarse search
        n_coarse = max(int(round(360.0 / coarse_step_deg)), 1)

        for k in range(n_coarse):
            theta = theta0 + 2.0 * math.pi * (k / n_coarse)
            theta = theta % (2.0 * math.pi)

            coords_i = self._place_one_ligand(
                ligands[i],
                site_planes[i],
                site_positions[i],
                theta,
            )
            min_d = self._min_distance(coords_i, neighbors_coords)

            better = False
            if best_theta is None:
                better = True
            else:
                # Case 1: both new and best are below cutoff
                if best_min_d < cutoff and min_d > best_min_d:
                    better = True
                # Case 2: best is below cutoff, new candidate is above cutoff
                elif best_min_d < cutoff and min_d >= cutoff:
                    better = True
                # Case 3: both new and best are above cutoff, prefer larger min_d
                elif best_min_d >= cutoff and min_d >= cutoff and min_d > best_min_d:
                    better = True

            if better:
                best_theta = theta
                best_min_d = min_d
                best_coords_i = coords_i

        if best_min_d >= cutoff:
            return best_theta, best_coords_i
        
        # Fine search 
        window_rad = math.radians(window_deg)
        fine_step_rad = math.radians(fine_step_deg)

        n_fine = int(round(2.0 * window_rad / fine_step_rad)) + 1
        for k in range(n_fine):
            delta = -window_rad + k * (2.0 * window_rad / (n_fine - 1))
            theta = (best_theta + delta) % (2.0 * math.pi)

            coords_i = self._place_one_ligand(ligand_i, site_planes[i], site_positions[i], theta)
            min_d = self._min_distance(coords_i, neighbors_coords)

            better = False
            if best_min_d < cutoff and min_d > best_min_d:
                better = True
            elif best_min_d < cutoff and min_d >= cutoff:
                better = True
            elif best_min_d >= cutoff and min_d >= cutoff and min_d > best_min_d:
                better = True

            if better:
                best_theta = theta
                best_min_d = min_d
                best_coords_i = coords_i

        return best_theta, best_coords_i


    def _build_octahedra(self) -> None:
        
        # Core
        at = self.core.atoms
        syms = np.array(at.get_chemical_symbols())
        pos = at.get_positions()

        b_idx = np.where(syms == self.core.B)[0]
        x_idx = np.where(syms == self.core.X)[0]

        cell = at.get_cell()
        Lx, Ly, Lz = cell.lengths()

        scaled = at.get_scaled_positions(wrap=True)
        pos = scaled @ cell.array

        B_pos = pos[b_idx]
        X_pos = pos[x_idx]

        x_tree = cKDTree(X_pos, boxsize=(Lx, Ly, Lz))
        r_cut = float(self.core.a) + 1e-2
        neigh_lists = x_tree.query_ball_point(B_pos, r_cut)

        octahedra: Dict[int, Dict[str, List[int]]] = {}

        for b_loc, x_local_list in enumerate(neigh_lists):
            b_abs = int(b_idx[b_loc])  
            x_abs_list = [int(x_idx[j]) for j in x_local_list]  

            octahedra[b_abs] = {
                "X": x_abs_list,  
                "Ligand": []
            }

        anchor_positions = []
        ligand_ids = []

        for lig in self.ligands:
            if lig.charge <= 0 and getattr(lig, "anchor_pos", None) is not None:
                anchor_positions.append(np.asarray(lig.anchor_pos, dtype=float))
                ligand_ids.append(int(lig.id))

        if anchor_positions:
            anchor_positions = np.vstack(anchor_positions)

            b_tree = cKDTree(B_pos, boxsize=(Lx, Ly, Lz))
            _, b_loc = b_tree.query(anchor_positions, k=1)

            for j, b_loc_j in enumerate(b_loc):
                b_abs = int(b_idx[int(b_loc_j)])
                octahedra[b_abs]["Ligand"].append(ligand_ids[j])

        self.octahedra = octahedra


    def _build_B_ijk(self) -> None:
        if not self.octahedra:
            self.B_ijk = {}
            return

        b_keys = np.array(sorted(self.octahedra.keys()), dtype=int)
        pos = np.asarray(self.atoms.positions, dtype=float)
        b_pos = pos[b_keys]

        origin = b_pos.min(axis=0, keepdims=True)
        ijk_arr = np.rint((b_pos - origin) / float(self.core.a)).astype(int)

        self.B_ijk = {
            int(b): (int(ijk_arr[i, 0]), int(ijk_arr[i, 1]), int(ijk_arr[i, 2]))
            for i, b in enumerate(b_keys)
        }


    def _build_index_map(self) -> None:

        n_core = len(self.core.atoms)
        self.core.indices = np.arange(n_core, dtype=int)

        cursor = n_core
        for lig in self.ligands:
            n = len(lig.atoms)
            idx = np.arange(cursor, cursor + n, dtype=int)
            lig.indices = idx
            cursor += n


    def to(self, fmt='vasp', filename: Optional[str] = None) -> None:
        assert fmt == 'vasp', "Only 'vasp' format is supported currently."
        assert len(self.ligands) > 0, "No ligands placed. Cannot export Slab."
        
        at = self.atoms
        formula = at.get_chemical_formula()

        if filename is None:
            filename = f"{formula}.{fmt}"
        
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)  
        
        write_vasp(str(path), self.atoms, sort=True, direct= True)
        self.to_json(str(path) + ".json")


    def to_json(self, json_path: str) -> None:

        self._build_index_map()

        core_meta = {
            "A": self.core.A,
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
            key = (
                lig.name,
                lig.smiles,
                lig.charge,
                tuple(lig.binding_motif.atoms),
            )

            if key not in type_key_to_id:
                type_id = len(ligand_types_meta)
                type_key_to_id[key] = type_id
                type_counts[type_id] = 0

                ligand_types_meta.append(
                    {
                        "id": type_id,
                        "name": lig.name,
                        "smiles": lig.smiles,
                        "charge": lig.charge,
                        "binding_motif_atoms": list(lig.binding_motif.atoms),
                        "binding_atoms_indices": list(getattr(lig, "binding_atoms", [])),
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
            key = (
                lig.name,
                lig.smiles,
                lig.charge,
                tuple(lig.binding_motif.atoms),
            )
            spec_id = type_key_to_id[key]

            ligands_meta.append(
                {   
                    "ligand_id": i,
                    "spec_id": spec_id,
                    "plane": list(lig.plane),
                }
            )

        n_total_atoms = core_meta["n_core_atoms"] + sum(
            t["n_atoms"] * t["n_instances"] for t in ligand_types_meta
        )

        topo = {
            "schema_version": 1,
            "n_total_atoms": n_total_atoms,
            "core": core_meta,
            "ligand_types": ligand_types_meta,
            "core_indices": core_indices_meta,
            "ligands": ligands_meta,
            "vasp_sort_idx": np.argsort(self.atoms.symbols).tolist()
        }

        with open(json_path, "w") as f:
            json.dump(topo, f, indent=2)


    @classmethod
    def from_file(cls, file_path: str, json_path: str) -> Slab:

        atoms = read(file_path)

        with open(json_path) as f:
            topo = json.load(f)

        vasp_sort_idx = topo.get("vasp_sort_idx", None)
        atoms = atoms[np.argsort(vasp_sort_idx)]

        schema_version = topo.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {schema_version!r}")

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
            A=core_meta["A"], 
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

            lig = object.__new__(Ligand)
            lig.atoms = lig_atoms
            lig.mol = None                      
            lig.smiles = tmeta["smiles"]
            lig.charge = tmeta["charge"]
            lig.binding_motif = BindingMotif(tmeta["binding_motif_atoms"])
            lig.name = tmeta["name"]
            lig.plane = inst_meta["plane"]
            lig.volume = tmeta["volume"]
            lig._neighbor_cutoff = 2.0
            lig.binding_atoms = list(map(int, tmeta.get("binding_atoms_indices", [])))
            
            pos = np.asarray(lig_atoms.positions, dtype=float)
            lig.anchor_pos = pos[np.asarray(lig.binding_atoms, dtype=int)].mean(axis=0)

            ligands.append(lig)

        slab = cls(
            core=core,
            ligand_specs=[],  
            random_seed=0,    
        )
        slab.ligands = ligands
        slab.ligand_coverage = {t["name"]: t["coverage"] for t in ligand_types_meta}
        slab.octahedra = octahedra
        slab.B_ijk = B_ijk
        slab._build_index_map()

        return slab


    @property
    def atoms(self) -> Atoms:
        core_atoms = self.core.atoms
        core_symbols = list(core_atoms.get_chemical_symbols())
        core_positions = core_atoms.get_positions()

        all_symbols = list(core_symbols)
        all_positions = [core_positions]

        for lig in self.ligands:
            lig_atoms = lig.atoms
            all_symbols.extend(lig_atoms.get_chemical_symbols())
            all_positions.append(lig_atoms.get_positions())

        all_positions = np.vstack(all_positions)

        return Atoms(
            symbols=all_symbols,
            positions=all_positions,
            pbc=core_atoms.pbc,
            cell=core_atoms.get_cell(),
        )