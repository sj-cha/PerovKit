from __future__ import annotations
from dataclasses import dataclass, field
from turtle import st
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
class Slab:
    A: str
    B: str
    X: str
    atoms: Atoms
    a: float
    supercell: Sequence[int] # (nx, ny, nz)                 
    indices: Optional[np.ndarray] = None
    octahedra: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    B_ijk: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    build_surface: bool = True
    surface_atoms: Dict[str, np.ndarray] = field(init=False)
    plane_atoms: Dict[Plane, Dict[str, List[int]]] = field(default_factory=dict)
    binding_sites: List[BindingSite] = field(default_factory=list)

    def __post_init__(self):
        if len(self.supercell) != 3:
            raise ValueError(f"supercell must be length-3 (nx, ny, nz); got {self.supercell}")
        self.supercell = tuple(int(x) for x in self.supercell)

        self.surface_atoms = self._get_surface_atoms() if self.build_surface else {}
        self.binding_sites = self._build_binding_sites() if self.build_surface else []
        self._build_octahedra()
        self._build_B_ijk()


    @classmethod
    def build(
        cls,
        A: str,
        B: str,
        X: str,
        a: float,
        supercell: Sequence[int], # (nx, ny, nz)
        tol: float = 1e-5,
    ) -> Slab:

        if len(supercell) != 3:
            raise ValueError(f"supercell must be length-3 (sx, sy, sz); got {supercell}")
        sx, sy, sz = map(int, supercell)
        if sx <= 0 or sy <= 0 or sz <= 0:
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
            pbc=True,
        )

        atoms = bulk.repeat((sx, sy, sz + 1))

        z_cut = float(a) * float(sz) 
        pos = atoms.get_positions()
        keep = pos[:, 2] <= (z_cut + tol)
        atoms = atoms[keep]

        atoms.set_cell([sx * a, sy * a, sz * a])

        slab = cls(A=A, B=B, X=X, atoms=atoms, a=float(a), supercell=(sx, sy, sz))

        return slab


    def perturb(self, 
                bound: List[float], 
                seed: Optional[int] = None
    ) -> None:
        
        if seed is not None:
            rng = np.random.default_rng(seed)
            rand_uniform = rng.uniform
        else:
            rand_uniform = np.random.uniform

        lo, hi = float(bound[0]), float(bound[1])
        if lo < 0 or hi <= 0 or hi < lo:
            raise ValueError(f"Bound must be a valid fraction, satisfying 0 <= low <= high, got {bound}")

        symbols = self.atoms.get_chemical_symbols()
        radii = np.array([CovalentRadius.radius[s] for s in symbols], dtype=float)
        mags = rand_uniform(radii * lo, radii * hi)

        dirs = rand_uniform(-1.0, 1.0, size=(len(self.atoms), 3))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs /= norms

        pos = self.atoms.get_positions()
        pos = pos + dirs * mags[:, None]
        self.atoms.set_positions(pos)


    def apply_tilt(
        self,
        glazer: str,
        angles: Tuple[float, float, float],
        *,
        order: str = "xyz",
    ):
        from .tilt import apply_tilt
        apply_tilt(structure=self, glazer=glazer, angles=angles, order=order)


    def apply_strain(self, strain: Sequence[float]):
        from .strain import apply_strain
        apply_strain(structure=self, strain=strain)


    def to(self, fmt: str = "vasp", filename: str = None, vacuum = 15) -> None:
        if fmt == 'vasp':
            self.atoms.cell += np.array([[0,0,0],[0,0,0],[0,0,vacuum]])
            write_vasp("POSCAR", self.atoms, sort=True)

        if fmt == 'xyz':
            if filename is None:
                nx, ny, nz = self.supercell
                filename = f"{self.A}{self.B}{self.X}3_{nx}x{ny}x{nz}.{fmt}"
            formula = self.atoms.get_chemical_formula()
            write(filename, self.atoms, format=fmt, comment=formula)


    def _get_surface_atoms(self, tol: float = 1e-2) -> dict[str, np.ndarray]:
        surface_indices: dict[str, np.ndarray] = {}

        positions = np.asarray(self.atoms.get_positions(), dtype=float)  # (N,3)
        symbols = np.array(self.atoms.get_chemical_symbols())

        for element in [self.A, self.X]:
            elem_global = np.where(symbols == element)[0]
            elem_pos = positions[elem_global]           # (Ne,3)
            z_max = elem_pos[:,2].max()

            surface_flags = np.isclose(elem_pos[:,2], z_max, atol=tol)

            surface_indices[element] = elem_global[surface_flags].astype(int)

        return surface_indices


    def _build_binding_sites(self) -> None:
        surface = self._get_surface_atoms()
        positions = np.array([a.position for a in self.atoms])
        symbols   = np.array([a.symbol   for a in self.atoms])

        tol = 1e-3
        plane_indices = defaultdict(lambda: defaultdict(list))

        for elem, idxs in surface.items():
            elem_global = np.where(symbols == elem)[0]
            elem_pos    = positions[elem_global]
            mins, maxs  = elem_pos.min(0), elem_pos.max(0)

            for i in idxs:
                p = positions[i]

                is_max = np.isclose(p, maxs, atol=tol)
                is_min = np.isclose(p, mins, atol=tol)

                v = tuple(int(x) for x in (is_max.astype(int) - is_min.astype(int)))
                nz = np.count_nonzero(v)

                if nz == 0:
                    continue

                plane_indices[v][elem].append(int(i))

        plane_atoms = {hkl: {elem: idxs for elem, idxs in elems.items()} for hkl, elems in plane_indices.items()}
        self.plane_atoms = plane_atoms

        idx_to_site: Dict[int, BindingSite] = {}

        for plane, elem_map in plane_atoms.items():
            for elem, indices in elem_map.items():
                for idx in indices:
                    idx = int(idx)
                    if idx in idx_to_site:
                        continue

                    idx_to_site[idx] = BindingSite(
                        index=idx,
                        symbol=elem,
                        plane=plane,
                        passivated=False,
                    )

        return list(idx_to_site.values())


    def _build_octahedra(self) -> None:
        at = self.atoms
        syms = np.array(at.get_chemical_symbols())
        pos = at.get_positions()

        b_idx = np.where(syms == self.B)[0]
        x_idx = np.where(syms == self.X)[0]

        B_pos = pos[b_idx]
        X_pos = pos[x_idx]

        tree = cKDTree(X_pos)
        r_cut = self.a + 1e-2
        neigh_lists = tree.query_ball_point(B_pos, r_cut)

        octahedra: Dict[int, Dict[str, List[int]]] = {}

        for b_loc, x_local_list in enumerate(neigh_lists):
            b_abs = int(b_idx[b_loc])
            x_abs_list = [int(x_idx[j]) for j in x_local_list]

            octahedra[b_abs] = {"X": x_abs_list, "Ligand": []}

        self.octahedra = octahedra


    def _build_B_ijk(self) -> None:
        b_keys = np.array(sorted(self.octahedra.keys()), dtype=int)

        pos = np.asarray(self.atoms.positions, dtype=float)
        b_pos = pos[b_keys]

        origin = b_pos.min(axis=0, keepdims=True)
        ijk_arr = np.rint((b_pos - origin) / float(self.a)).astype(int)

        self.B_ijk = {
            int(b): (int(ijk_arr[i, 0]), int(ijk_arr[i, 1]), int(ijk_arr[i, 2]))
            for i, b in enumerate(b_keys)
        }