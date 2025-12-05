from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import random
import math
import numpy as np
from ase import Atoms
from ase.io import write

from scipy.spatial import cKDTree 

from core import Core, BindingSite
from ligand import Ligand, LigandSpec
from utils.rotation import rotation_about_axis, rotation_from_u_to_v

@dataclass
class NanoCrystal:
    core: Core
    ligand_specs: List[LigandSpec]
    overlap_cutoff: float = 2.0   # Å
    random_seed: Optional[int] = None
    binding_sites: List[BindingSite] = field(default_factory=list)
    ligands: List[Ligand] = field(default_factory=list)
    displaced_indices: List[int] = field(default_factory=list)  # core atom indices removed

    def __post_init__(self):
        if not self.binding_sites:
            self.binding_sites = self.core.binding_sites
        self._rng = random.Random(self.random_seed)
        self.ligand_specs.sort(
            key=lambda spec: spec.ligand.volume,
            reverse=True,
        )

    def place_ligands(
        self,
        max_iters: int = 10
    ) -> None:
        
        assert not self.ligands, "Ligands have already been placed."

        self.ligands = []
        self.displaced_indices = []
        self.binding_sites = self.core.binding_sites
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

            # Coverage
            if 0.0 < spec.coverage <= 1.0:
                n_target = int(math.ceil(spec.coverage * len(available)))
            else:
                n_target = int(spec.coverage)
                if n_target > len(available):
                    print(f"[Warning] Requested {n_target} sites for ligand, "
                          f"but only {len(available)} available.")
                    n_target = len(available)

            chosen_sites = self._select_sites_uniform(core_positions, available, n_target)

            for site in chosen_sites:
                lig_copy = self._clone_ligand(lig)
                site.passivated = True

                ligands.append(lig_copy)
                sites.append(site)
                self.displaced_indices.append(site.index)

            print(f"[Log] Placed total {len(chosen_sites)} {ligand_type} ligands.")

        n_lig = len(ligands)
        assert n_lig > 0, "No ligands to place."

        # Core atoms coordinates except displaced ones
        n_core = core_positions.shape[0]

        surface_indices = np.concatenate(list(self.core.surface_atoms.values()))

        disp_set = set(self.displaced_indices)

        valid_surface_indices: List[int] = []
        for i in surface_indices:
            if 0 <= i < n_core and i not in disp_set:
                valid_surface_indices.append(i)

        if valid_surface_indices:
            idx_arr = np.array(valid_surface_indices, dtype=int)
            surface_coords = core_positions[idx_arr]
            core_tree = cKDTree(surface_coords)
        else:
            surface_coords = np.empty((0, 3), dtype=float)
            core_tree = None

        self._core_tree = core_tree

        site_positions = np.array([core_positions[s.index] for s in sites])

        site_normals = np.zeros((n_lig, 3), dtype=float)
        for i, s in enumerate(sites):
            n = np.asarray(s.plane, dtype=float)
            n_hat = n / np.linalg.norm(n)
            site_normals[i] = n_hat

        # Initialize rotation angles around surface normal and current coordinates
        thetas = np.array([2.0 * math.pi * self._rng.random() for _ in range(n_lig)], dtype=float)
        ligand_coords_list: List[np.ndarray] = []
        for i in range(n_lig):
            coords_i = self._place_one_ligand(
                ligands[i],
                site_normals[i],
                site_positions[i],
                thetas[i],
            )
            ligand_coords_list.append(coords_i)
        
        # Rotaation optimization loop
        for iter in range(max_iters):
            centers, radii = self._compute_bounding_spheres(ligand_coords_list)
            neighbor_map = self._build_neighbor_map(centers, radii, self.overlap_cutoff)

            neighbor_coords_map = self._build_neighbor_coords_map(
                ligand_coords_list,
                neighbor_map,
            )

            global_min, conflict_ligs, dists = self._get_conflict_ligands(ligand_coords_list, 
                                                                          neighbor_coords_map,
                                                                          self.overlap_cutoff
                                                                          )
            print(f"[Log] Iter {iter}  global_min = {global_min:.3f} Å")

            if global_min >= self.overlap_cutoff:
                print("[Log] Hard cutoff satisfied. Done.")
                break

            if not conflict_ligs:
                print("[Log] No conflict ligands but global_min < cutoff. stopping.")
                break

            active_set = self._build_active_cluster(conflict_ligs, sites, site_positions)

            improved = False
            active_order = active_set[:]
            self._rng.shuffle(active_order)

            for i in active_order:
                neighbor_idx = neighbor_map.get(i, [])
                other_ligs = [ligand_coords_list[j] for j in neighbor_idx]

                d_old = dists[i]

                best_theta, best_coords = self._optimize_rotation(
                    i, 
                    ligands, 
                    other_ligs,
                    site_normals, 
                    site_positions
                )
                d_new = self._min_distance(best_coords, other_ligs)

                if d_new > d_old + 1e-3:
                    ligand_coords_list[i] = best_coords
                    thetas[i] = best_theta
                    improved = True

            if not improved:
                print("[Log] No improvement in active cluster. Stopping.")
                break

        # Apply final coordinates to Ligand
        self.ligands = []
        for lig, coords in zip(ligands, ligand_coords_list):
            lig.atoms.set_positions(coords)
            self.ligands.append(lig)
    
    def check_overlaps(self, cutoff: float = None):
        if cutoff is None:
            cutoff = self.overlap_cutoff

        core_symbols = self.core.atoms.get_chemical_symbols()
        core_positions = self.core.atoms.get_positions()
        core_mask = np.ones(len(core_symbols), dtype=bool)
        for idx in self.displaced_indices:
            if 0 <= idx < len(core_mask):
                core_mask[idx] = False
        core_coords = core_positions[core_mask]

        entities = []
        entity_labels = []

        if core_coords.size > 0:
            entities.append(core_coords)
            entity_labels.append("core")

        for i, lig in enumerate(self.ligands):
            entities.append(lig.atoms.get_positions())
            entity_labels.append(f"ligand_{i}")

        global_min = np.inf
        global_pair = None

        n = 0

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                A = entities[i]
                B = entities[j]
                diff = A[:, None, :] - B[None, :, :]
                d2 = np.sum(diff * diff, axis=-1)
                min_idx = np.argmin(d2)
                d_min = float(np.sqrt(d2.flat[min_idx]))

                if d_min < global_min:
                    global_min = d_min
                    global_pair = (i, j)

                if d_min < cutoff:
                    print(
                        f"[Log] {entity_labels[i]} vs {entity_labels[j]}: "
                        f"min distance = {d_min:.3f} Å < {cutoff:.3f} Å"
                    )
                    n += 1

        print(
            f"[Log] global min inter-entity distance = {global_min:.3f} Å "
            f"between {entity_labels[global_pair[0]]} and {entity_labels[global_pair[1]]}"
            f" (total overlaps: {n})"
        )

    def _compute_bounding_spheres(
        self,
        coords_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_lig = len(coords_list)
        centers = np.zeros((n_lig, 3), dtype=float)
        radii = np.zeros(n_lig, dtype=float)

        for i, coords in enumerate(coords_list):
            c = coords.mean(axis=0)
            centers[i] = c
            radii[i] = np.linalg.norm(coords - c, axis=1).max()

        return centers, radii

    def _build_neighbor_map(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        cutoff: float,
    ) -> Dict[int, List[int]]:

        n = len(centers)
        neighbor_map: Dict[int, List[int]] = {i: [] for i in range(n)}

        if n == 0:
            return neighbor_map

        max_r = float(radii.max()) if n > 0 else 0.0

        tree = cKDTree(centers)

        for i in range(n):
            search_radius = radii[i] + max_r + cutoff

            candidate_js = tree.query_ball_point(centers[i], r=search_radius)

            for j in candidate_js:
                if j == i:
                    continue

                center_dist = np.linalg.norm(centers[j] - centers[i])
                if center_dist <= radii[i] + radii[j] + cutoff:
                    neighbor_map[i].append(j)
                    neighbor_map[j].append(i)

        for i in range(n):
            if neighbor_map[i]:
                neighbor_map[i] = sorted(set(neighbor_map[i]))

        return neighbor_map

    def _build_neighbor_coords_map(
        self,
        ligand_coords_list: List[np.ndarray],
        neighbor_map: Dict[int, List[int]],
    ) -> Dict[int, Optional[np.ndarray]]:
        
        n_lig = len(ligand_coords_list)
        neighbor_coords_map: Dict[int, Optional[np.ndarray]] = {}

        for i in range(n_lig):
            idxs = neighbor_map.get(i, [])
            if idxs:
                neighbor_coords_map[i] = np.vstack(
                    [ligand_coords_list[j] for j in idxs]
                )
            else:
                neighbor_coords_map[i] = None

        return neighbor_coords_map

    def _get_conflict_ligands(
        self,
        ligand_coords_list: List[np.ndarray],
        neighbor_coords_map: Dict[int, Optional[np.ndarray]],
        cutoff: float,
    ) -> Tuple[float, List[int], np.ndarray]:

        n_lig = len(ligand_coords_list)
        conflict_ligs: List[int] = []
        dists = np.empty(n_lig, dtype=float)
        global_min = float("inf")

        for i in range(n_lig):
            coords_i = ligand_coords_list[i]
            other_flat = neighbor_coords_map[i]

            d_i = self._min_distance(coords_i, other_ligands_coords=None, other_ligands_flat=other_flat)
            dists[i] = d_i
            global_min = min(global_min, d_i)

            if d_i < cutoff:
                conflict_ligs.append(i)

        return global_min, conflict_ligs, dists

    def _build_active_cluster(
        self,
        conflict_ligs: List[int],
        sites: List[BindingSite],
        site_positions: np.ndarray,
    ) -> List[int]:

        radius = self.core.a + 1e-3

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
        n_hat: np.ndarray,
        site_pos: np.ndarray,
        theta: float,
    ) -> np.ndarray:

        coords_loc = ligand.atoms.get_positions()

        # Rotate to surface normal
        R_align = rotation_from_u_to_v(
            np.array([0.0, 0.0, 1.0], dtype=float),
            n_hat,
        )
        coords_aligned = coords_loc @ R_align.T

        # Rotate around surface normal 
        R_rot = rotation_about_axis(n_hat, theta)
        coords_rot = coords_aligned @ R_rot.T

        # Translate to binding site position
        coords_final = coords_rot + site_pos

        return coords_final

    def _min_distance_core(
        self,
        coords_i: np.ndarray,
    ) -> float:
        
        core_tree = self._core_tree
        dists, _ = core_tree.query(coords_i)
        return float(np.min(dists))

    def _min_distance(
        self,
        coords_i: np.ndarray,
        other_ligands_coords: Optional[List[np.ndarray]] = None,
        other_ligands_flat: Optional[np.ndarray] = None,
    ) -> float:

        d_core = self._min_distance_core(coords_i)

        if d_core < self.overlap_cutoff:
            return d_core

        if other_ligands_flat is not None:
            others = other_ligands_flat
        elif other_ligands_coords:
            others = np.vstack(other_ligands_coords)
        else:
            return d_core

        diff = coords_i[None, :, :] - others[:, None, :]
        d2 = np.sum(diff * diff, axis=-1)
        d_lig = float(np.sqrt(d2.min()))

        return min(d_core, d_lig)

    def _optimize_rotation(
        self,
        i: int,
        ligands: List[Ligand],
        other_ligands_coords: List[np.ndarray],
        site_normals: np.ndarray, 
        site_positions: np.ndarray,
    ) -> Tuple[float, np.ndarray]:

        r_cut = self.overlap_cutoff

        ligand_i = ligands[i]

        fine_step_deg = 2
        coarse_step_deg = 12

        n_coarse = max(int(round(360.0 / coarse_step_deg)), 1)

        best_theta = None
        best_min_d = -float("inf")
        best_coords_i = None

        theta0 = 2.0 * math.pi * self._rng.random()

        # Coarse search
        for k in range(n_coarse):
            theta = theta0 + 2.0 * math.pi * (k / n_coarse)
            theta = theta % (2.0 * math.pi)

            coords_i = self._place_one_ligand(
                ligands[i],
                site_normals[i],
                site_positions[i],
                theta,
            )
            min_d = self._min_distance(coords_i, other_ligands_coords)

            better = False
            if best_theta is None:
                better = True
            else:
                # Case 1: both new and best are below cutoff
                if best_min_d < r_cut and min_d > best_min_d:
                    better = True
                # Case 2: best is below cutoff, new candidate is above cutoff
                elif best_min_d < r_cut and min_d >= r_cut:
                    better = True
                # Case 3: both new and best are above cutoff, prefer larger min_d
                elif best_min_d >= r_cut and min_d >= r_cut and min_d > best_min_d:
                    better = True

            if better:
                best_theta = theta
                best_min_d = min_d
                best_coords_i = coords_i

        if best_theta is None:
            best_theta = 0.0
            best_coords_i = self._place_one_ligand(ligand_i, site_normals[i], site_positions[i], best_theta)
            return best_theta, best_coords_i

        # Fine search 
        window_deg = 10
        window_rad = math.radians(window_deg)
        fine_step_rad = math.radians(fine_step_deg)

        if window_rad > 0 and fine_step_rad > 0:
            n_fine = int(round(2.0 * window_rad / fine_step_rad)) + 1
        else:
            n_fine = 1

        for k in range(n_fine):
            if n_fine > 1:
                delta = -window_rad + k * (2.0 * window_rad / (n_fine - 1))
            else:
                delta = 0.0

            theta = (best_theta + delta) % (2.0 * math.pi)

            coords_i = self._place_one_ligand(ligand_i, site_normals[i], site_positions[i], theta)
            min_d = self._min_distance(coords_i, other_ligands_coords)

            better = False
            if best_min_d < r_cut and min_d > best_min_d:
                better = True
            elif best_min_d < r_cut and min_d >= r_cut:
                better = True
            elif best_min_d >= r_cut and min_d >= r_cut and min_d > best_min_d:
                better = True

            if better:
                best_theta = theta
                best_min_d = min_d
                best_coords_i = coords_i

        return best_theta, best_coords_i

    def _clone_ligand(self, ligand: Ligand) -> Ligand:
        lig_cloned = object.__new__(Ligand)
        lig_cloned.__dict__ = ligand.__dict__.copy()
        lig_cloned.atoms = ligand.atoms.copy()

        return lig_cloned

    def _select_sites_uniform(
        self,
        core_positions: np.ndarray,
        sites: List[BindingSite],
        n_target: int,
    ) -> List[BindingSite]:
        
        if n_target >= len(sites):
            return list(sites)

        coords = np.array([core_positions[s.index] for s in sites])

        center = coords.mean(axis=0)
        d2 = np.sum((coords - center) ** 2, axis=1)
        first_idx = int(np.argmax(d2))

        selected_indices = [first_idx]

        d_min = np.linalg.norm(coords - coords[first_idx], axis=1)
        d_min[first_idx] = 0.0

        while len(selected_indices) < n_target:
            next_idx = int(np.argmax(d_min))
            selected_indices.append(next_idx)

            new_d = np.linalg.norm(coords - coords[next_idx], axis=1)
            d_min = np.minimum(d_min, new_d)
            d_min[selected_indices] = 0.0  # already chosen

        return [sites[i] for i in selected_indices]

    def to(self, fmt: str = "xyz", filename: str = None):
        at = self.atoms
        formula = at.get_chemical_formula()
        write(filename, at, format=fmt, comment=formula)

    @property
    def atoms(self) -> Atoms:
        """
        Combined Atoms object of (core - displaced A/X) + all placed ligands.
        """
        core_symbols = self.core.atoms.get_chemical_symbols()
        core_positions = self.core.atoms.get_positions()

        core_mask = np.ones(len(core_symbols), dtype=bool)
        for idx in self.displaced_indices:
            if 0 <= idx < len(core_mask):
                core_mask[idx] = False

        base_symbols = [s for s, m in zip(core_symbols, core_mask) if m]
        base_positions = core_positions[core_mask]

        lig_symbols: List[str] = []
        lig_positions_list: List[np.ndarray] = []

        for lig in self.ligands:
            lig_symbols.extend(lig.atoms.get_chemical_symbols())
            lig_positions_list.append(lig.atoms.get_positions())

        if lig_positions_list:
            lig_positions = np.vstack(lig_positions_list)
            all_symbols = base_symbols + lig_symbols
            all_positions = np.vstack([base_positions, lig_positions])
        else:
            all_symbols = base_symbols
            all_positions = base_positions

        return Atoms(symbols=all_symbols, positions=all_positions, pbc=False)


if __name__ == "__main__":
    import time 

    start_time = time.time()
    random_seed = 42

    core = Core.build_core(
        A="Cs",
        B="Pb",
        X="Br",
        a=5.95,
        n_cells=3,
        charge_neutral=True,
        random_seed=random_seed,
    )

    from ligand import BindingMotif

    # cat_lig = Ligand.from_smiles(
    #     smiles="CCCCCCCC/C=C\CCCCCCCC[NH3+]",
    #     binding_motif=BindingMotif(["N"]),
    #     random_seed=random_seed,
    # )

    #    e.g. Anionic ligand (displaces X sites)
    an_lig = Ligand.from_smiles(
        smiles="CCCCCCCC/C=C\CCCCCCCC(=O)[O-]",   
        binding_motif=BindingMotif(["O", "O"]),
        random_seed=random_seed,
    )

    cat_lig = Ligand.from_smiles(
        smiles="C[NH3+]",
        binding_motif=BindingMotif(["N"]),
        random_seed=random_seed,
    )

    # an_lig = Ligand.from_xyz(
    #     "../../ligands/3_OP.xyz",
    #     charge = -1,
    #     binding_motif=BindingMotif(["O", "O"]),
    # )

    specs = [
        LigandSpec(ligand=cat_lig, coverage=0.3),  
        LigandSpec(ligand=an_lig, coverage=0.3),  
    ]

    nc = NanoCrystal(core=core, ligand_specs=specs, random_seed=random_seed)
    nc.place_ligands()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    nc.to(filename=f"CsPbBr3_{core.n_cells}x{core.n_cells}x{core.n_cells}_NC.xyz")

    nc._debug_overlaps()