from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Sequence
from collections import defaultdict

import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.vasp import write_vasp
from scipy.spatial import cKDTree
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

import random

Plane = Tuple[int, int, int]


@dataclass
class BindingSite:
    index: int
    symbol: str
    plane: Plane
    passivated: bool = False


@dataclass
class Core:
    A: str
    B: str
    X: str
    atoms: Atoms
    a: float

    n_cells: Optional[int] = None                    # used for core mode
    supercell: Optional[Tuple[int, int, int]] = None # used for slab mode (nx, ny, nz)
    vacuum: Optional[float] = None                   # used for slab mode

    indices: Optional[np.ndarray] = None
    octahedra: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    B_ijk: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    build_surface: bool = True
    surface_atoms: Dict[str, np.ndarray] = field(init=False)
    plane_atoms: Dict[Plane, Dict[str, List[int]]] = field(default_factory=dict)
    binding_sites: List[BindingSite] = field(default_factory=list)

    def __post_init__(self):
        if self.supercell is not None:
            if len(self.supercell) != 3:
                raise ValueError(f"supercell must be length-3 (nx, ny, nz); got {self.supercell}")
            self.supercell = tuple(int(x) for x in self.supercell)

        if self.build_surface:
            self.surface_atoms = self._get_surface_atoms()
            self.binding_sites = self._build_binding_sites()
            self._build_octahedra()
            self._build_B_ijk()
        else:
            self.surface_atoms = {}
            self.binding_sites = []
            self.octahedra = {}
            self.B_ijk = {}

    @property
    def is_slab(self) -> bool:
        return self.supercell is not None

    @property
    def is_nanocrystal(self) -> bool:
        return not self.is_slab


    @classmethod
    def build_nanocrystal(
        cls,
        A: str,
        B: str,
        X: str,
        a: float,
        n_cells: int,
        charge_neutral: bool = True,
        random_seed: Optional[int] = None,
        tol: float = 1e-5,
    ) -> Core:

        species = [A, B, X, X, X]
        motif_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # A
                [0.5, 0.5, 0.5],  # B
                [0.5, 0.5, 0.0],  # X
                [0.5, 0.0, 0.5],  # X
                [0.0, 0.5, 0.5],  # X
            ],
            dtype=float,
        ) * float(a)

        N = int(n_cells) + 1
        all_symbols: List[str] = []
        all_positions: List[np.ndarray] = []

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    shift = np.array([i, j, k], dtype=float) * float(a)
                    for s, c in zip(species, motif_coords):
                        all_symbols.append(s)
                        all_positions.append(c + shift)

        all_positions = np.array(all_positions, dtype=float)

        atoms = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            pbc=False,
        )

        # Ensure AX termination 
        max_coord = float(a) * (N - 1)
        pos = atoms.positions
        mask = np.ones(len(atoms), dtype=bool)
        filt = np.any(pos > (max_coord + tol), axis=1)
        mask[filt] = False
        atoms = atoms[mask]

        core = cls(A=A, B=B, X=X, atoms=atoms, a=float(a), n_cells=int(n_cells))

        # Remove surface A atoms to ensure charge neutrality 
        if core.n_cells is not None and core.n_cells > 1 and charge_neutral:
            symbols = core.atoms.get_chemical_symbols()
            n_A = sum(s == A for s in symbols)
            n_B = sum(s == B for s in symbols)
            n_X = sum(s == X for s in symbols)

            net_charge = n_A * 1 + n_B * 2 - n_X * 1

            plane_indices = core.plane_atoms
            planes = list(plane_indices.keys())

            corners = [p for p in planes if np.all(p) == True]  
            corner_atoms = [x for c in corners for x in plane_indices[c].get(A, [])]
            rest = [p for p in planes if p not in corners]
            rest_atoms = [x for r in rest for x in plane_indices[r].get(A, [])]

            if core.n_cells == 2:
                corner_atoms = corner_atoms[:7]

            to_remove = set(corner_atoms)

            n_remove = int(net_charge - len(corner_atoms))
            if n_remove > 0:
                rng = random.Random(random_seed)
                extra_indices = rng.sample(sorted(rest_atoms), k=n_remove)
                to_remove.update(extra_indices)

            mask = np.ones(len(core.atoms), dtype=bool)
            mask[list(to_remove)] = False
            new_atoms = core.atoms[mask]

            core = cls(A=A, B=B, X=X, atoms=new_atoms, a=float(a), n_cells=int(n_cells))
            core._build_binding_sites()

            symbols = core.atoms.get_chemical_symbols()
            n_A = sum(s == A for s in symbols)
            n_B = sum(s == B for s in symbols)
            n_X = sum(s == X for s in symbols)
            assert n_A * 1 + n_B * 2 - n_X * 1 == 0, "Core is not charge neutral!"

        return core

    @classmethod
    def build_slab(
        cls,
        A: str,
        B: str,
        X: str,
        a: float,
        supercell: Sequence[int],  # (nx, ny, nz)
        vacuum: float = 15.0,
        tol: float = 1e-5,
    ) -> Core:

        if len(supercell) != 3:
            raise ValueError(f"supercell must be length-3 (nx, ny, nz); got {supercell}")
        nx, ny, nz = map(int, supercell)
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(f"supercell entries must be positive; got {supercell}")

        symbols = [A, B, X, X, X]
        scaled = np.array(
            [
                [0.0, 0.0, 0.0],  # A
                [0.5, 0.5, 0.5],  # B
                [0.5, 0.5, 0.0],  # X
                [0.5, 0.0, 0.5],  # X
                [0.0, 0.5, 0.5],  # X
            ],
            dtype=float,
        )

        bulk = Atoms(
            symbols=symbols,
            scaled_positions=scaled,
            cell=np.eye(3) * float(a),
            pbc=True
        )

        atoms = bulk.repeat((nx, ny, nz + 1))

        z_cut = float(a) * float(nz)
        pos = atoms.get_positions()
        keep = pos[:, 2] <= (z_cut + tol)
        atoms = atoms[keep]

        atoms.set_cell([nx * float(a), ny * float(a), nz * float(a) + float(vacuum)])
        atoms.set_pbc((True, True, False))

        slab = cls(
            A=A,
            B=B,
            X=X,
            atoms=atoms,
            a=float(a),
            supercell=(nx, ny, nz),
            vacuum=float(vacuum),
        )
        return slab
    

    def perturb(self, bound: List[float], seed: Optional[int] = None) -> None:
        if seed is not None:
            rng = np.random.default_rng(seed)
            rand_uniform = rng.uniform
        else:
            rand_uniform = np.random.uniform

        lo, hi = float(bound[0]), float(bound[1])
        if lo < 0 or hi <= 0 or hi < lo:
            raise ValueError(f"Bound must satisfy 0 <= low <= high, got {bound}")

        symbols = self.atoms.get_chemical_symbols()
        radii = np.array([CovalentRadius.radius[s] for s in symbols], dtype=float)
        mags = rand_uniform(radii * lo, radii * hi)

        dirs = rand_uniform(-1.0, 1.0, size=(len(self.atoms), 3))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs /= norms

        pos = self.atoms.get_positions()
        self.atoms.set_positions(pos + dirs * mags[:, None])

    def apply_tilt(self, glazer: str, angles: Tuple[float, float, float], *, order: str = "xyz"):
        from .tilt import apply_tilt
        if self.is_slab:
            raise NotImplementedError("Tilt is not implemented for slab structures.")
        apply_tilt(structure=self, glazer=glazer, angles=angles, order=order)

    def apply_strain(self, strain: Sequence[float]):
        from .strain import apply_strain
        apply_strain(structure=self, strain=strain)

        if self.is_slab:
            cell = self.atoms.get_cell().copy()
            cell[0] *= (1 + float(strain[0]))
            cell[1] *= (1 + float(strain[1]))
            cell[2] *= (1 + float(strain[2]))
            self.atoms.set_cell(cell, scale_atoms=False)


    def to(self, fmt, filename: Optional[str] = None) -> None:
        if filename is None:
            if self.is_slab:
                nx, ny, nz = self.supercell or (0, 0, 0)
                filename = f"{self.A}{self.B}{self.X}3_{nx}x{ny}x{nz}.{fmt}"
            else:
                nc = self.n_cells if self.n_cells is not None else "NA"
                filename = f"{self.A}{self.B}{self.X}3_{nc}.{fmt}"

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)  
        
        if fmt == "vasp":
            write_vasp(str(path), self.atoms, sort=True)
            return

        else:
            formula = self.atoms.get_chemical_formula()
            write(str(path), self.atoms, format=fmt, comment=formula)


    def _get_surface_atoms(self, tol: float = 1e-2) -> Dict[str, np.ndarray]:
        surface_indices: Dict[str, np.ndarray] = {}

        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        symbols = np.array(self.atoms.get_chemical_symbols())

        for element in [self.A, self.X]:
            elem_global = np.where(symbols == element)[0]
            elem_pos = positions[elem_global]

            if elem_pos.size == 0:
                surface_indices[element] = np.array([], dtype=int)
                continue

            if self.is_slab:
                z_max = elem_pos[:, 2].max()
                surface_flags = np.isclose(elem_pos[:, 2], z_max, atol=tol)
            else:
                mins = elem_pos.min(axis=0)
                maxs = elem_pos.max(axis=0)
                on_min = np.isclose(elem_pos, mins, atol=tol)
                on_max = np.isclose(elem_pos, maxs, atol=tol)
                surface_flags = np.any(on_min | on_max, axis=1)

            surface_indices[element] = elem_global[surface_flags].astype(int)

        return surface_indices

    def _build_binding_sites(self) -> List[BindingSite]:
        surface = self._get_surface_atoms()
        positions = np.array([a.position for a in self.atoms], dtype=float)
        symbols = np.array([a.symbol for a in self.atoms])

        tol = 1e-3
        plane_indices = defaultdict(lambda: defaultdict(list))

        if self.is_slab:
            for elem, idxs in surface.items():
                for i in idxs:
                    plane_indices[(0, 0, 1)][elem].append(int(i))            
        else:
            for elem, idxs in surface.items():
                elem_global = np.where(symbols == elem)[0]
                elem_pos = positions[elem_global]
                if elem_pos.size == 0:
                    continue

                mins, maxs = elem_pos.min(0), elem_pos.max(0)

                for i in idxs:
                    p = positions[int(i)]
                    is_max = np.isclose(p, maxs, atol=tol)
                    is_min = np.isclose(p, mins, atol=tol)

                    v = tuple(int(x) for x in (is_max.astype(int) - is_min.astype(int)))
                    if np.count_nonzero(v) == 0:
                        continue

                    plane_indices[v][elem].append(int(i))

        self.plane_atoms = {
            hkl: {elem: idxs for elem, idxs in elems.items()}
            for hkl, elems in plane_indices.items()
        }

        idx_to_site: Dict[int, BindingSite] = {}
        for plane, elem_map in self.plane_atoms.items():
            for elem, indices in elem_map.items():
                for idx in indices:
                    idx = int(idx)
                    if idx in idx_to_site:
                        continue
                    idx_to_site[idx] = BindingSite(index=idx, symbol=elem, plane=plane, passivated=False)

        return list(idx_to_site.values())

    def _build_octahedra(self) -> None:
        at = self.atoms
        syms = np.array(at.get_chemical_symbols())

        b_idx = np.where(syms == self.B)[0]
        x_idx = np.where(syms == self.X)[0]

        if len(b_idx) == 0 or len(x_idx) == 0:
            self.octahedra = {}
            return

        if self.is_slab:
            # periodic neighbor search (cKDTree boxsize needs coords in [0, L))
            cell = at.get_cell()
            Lx, Ly, Lz = cell.lengths()

            scaled = at.get_scaled_positions(wrap=True)
            pos = scaled @ cell.array

            B_pos = pos[b_idx]
            X_pos = pos[x_idx]

            tree = cKDTree(X_pos, boxsize=(Lx, Ly, Lz))
            r_cut = float(self.a) + 1e-2
            neigh_lists = tree.query_ball_point(B_pos, r_cut)
        else:
            pos = at.get_positions()
            B_pos = pos[b_idx]
            X_pos = pos[x_idx]

            tree = cKDTree(X_pos)
            r_cut = float(self.a) + 1e-2
            neigh_lists = tree.query_ball_point(B_pos, r_cut)

        octahedra: Dict[int, Dict[str, List[int]]] = {}
        for b_loc, x_local_list in enumerate(neigh_lists):
            b_abs = int(b_idx[b_loc])
            x_abs_list = [int(x_idx[j]) for j in x_local_list]
            octahedra[b_abs] = {"X": x_abs_list, "Ligand": []}

        self.octahedra = octahedra

    def _build_B_ijk(self) -> None:
        if not self.octahedra:
            self.B_ijk = {}
            return

        b_keys = np.array(sorted(self.octahedra.keys()), dtype=int)
        pos = np.asarray(self.atoms.positions, dtype=float)
        b_pos = pos[b_keys]

        origin = b_pos.min(axis=0, keepdims=True)
        ijk_arr = np.rint((b_pos - origin) / float(self.a)).astype(int)

        self.B_ijk = {
            int(b): (int(ijk_arr[i, 0]), int(ijk_arr[i, 1]), int(ijk_arr[i, 2]))
            for i, b in enumerate(b_keys)
        }