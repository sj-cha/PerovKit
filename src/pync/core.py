from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np
from ase import Atoms
from ase.io import read, write

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
    n_cells: int
    indices: Optional[np.ndarray] = None

    build_surface: bool = True
    surface_atoms: Dict[str, np.ndarray] = field(init=False)
    plane_atoms: Dict[Plane, Dict[str, List[int]]] = field(default_factory=dict)
    binding_sites: List[BindingSite] = field(default_factory=list)

    def __post_init__(self):
        self.surface_atoms = self._get_surface_atoms() if self.build_surface else {}
        self.binding_sites = self._build_binding_sites() if self.build_surface else []

    def _get_surface_atoms(
            self, 
            tol: float = 1e-2
    ) -> dict[str, np.ndarray]:
        
        surface_indices = {}

        for element in [self.A, self.X]:
            # mask & positions for this element
            atoms = [i for i in self.atoms if i.symbol == element] 
            positions = np.array([atom.position for atom in atoms])

            # element-wise min/max along x,y,z
            mins = positions.min(axis=0)
            maxs = positions.max(axis=0)

            # for each atom of this element, check if it's on any boundary plane
            surface_flags = []
            for p in positions:
                on_min = np.isclose(p, mins, atol=tol)
                on_max = np.isclose(p, maxs, atol=tol)
                # surface if on min or max in at least one direction
                surface_flags.append(np.any(on_min | on_max))

            surface_flags = np.array(surface_flags, dtype=bool)
            surface_indices[element] = np.where(surface_flags)[0]

        return surface_indices

    def _build_binding_sites(self) -> None:
        surface = self._get_surface_atoms()
        positions = np.array([a.position for a in self.atoms])
        symbols   = np.array([a.symbol   for a in self.atoms])

        tol = 1e-3
        plane_indices = defaultdict(lambda: defaultdict(list))

        for elem, local_idxs in surface.items():
            # Global indices for this element
            elem_global = np.where(symbols == elem)[0]
            elem_pos    = positions[elem_global]
            mins, maxs  = elem_pos.min(0), elem_pos.max(0)

            for li in local_idxs:
                gi = elem_global[li]
                p  = positions[gi]

                is_max = np.isclose(p, maxs, atol=tol)
                is_min = np.isclose(p, mins, atol=tol)

                # +1 for max plane, -1 for min plane, 0 otherwise
                v = tuple(int(x) for x in (is_max.astype(int) - is_min.astype(int)))
                nz = np.count_nonzero(v)

                if nz == 0:
                    continue 

                plane_indices[v][elem].append(int(gi))

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
        
    @classmethod
    # Currently only supports ABX3 perovskites with cubic structure
    def build_core(
        cls,
        A,
        B,
        X,
        a: float,
        n_cells: int,
        charge_neutral: bool = True,
        random_seed: Optional[int] = None,
        tol: float = 1e-5,
    ) -> Core:

        species = [A, B, X, X, X]
        motif_coords = np.array([
            [0.0, 0.0, 0.0],  # A
            [0.5, 0.5, 0.5],  # B
            [0.5, 0.5, 0.0],  # X
            [0.5, 0.0, 0.5],  # X
            [0.0, 0.5, 0.5],  # X
        ]) * a

        N = n_cells + 1
        all_symbols: List[str] = []
        all_positions: List[np.ndarray] = []

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    shift = np.array([i, j, k], dtype=float) * a
                    for s, c in zip(species, motif_coords):
                        all_symbols.append(s)
                        all_positions.append(c + shift)

        all_positions = np.array(all_positions)

        # Build ASE Atoms
        atoms = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            pbc=False,
        )

        # Ensure AX termination
        max_coord = a * (N - 1)
        mask = np.ones(len(atoms), dtype=bool)

        pos = atoms.positions
        filter = np.any(pos > (max_coord + tol), axis=1)
        mask[filter] = False

        atoms = atoms[mask]

        # Build Core instance
        core = cls(A=A, B=B, X=X, atoms=atoms, a=a, n_cells=n_cells)

        # Remove surface Cs atoms to ensure charge neutrality
        if core.n_cells > 1 and charge_neutral:
            symbols = core.atoms.get_chemical_symbols()
            n_A = sum(s == A for s in symbols)
            n_B = sum(s == B for s in symbols)
            n_X = sum(s == X for s in symbols)

            net_charge = n_A * 1 + n_B * 2 - n_X * 1  

            plane_indices = core.plane_atoms
            planes = list(plane_indices.keys())

            corners = [i for i in planes if np.all(i) == True] 
            corner_atoms = [x for c in corners for x in plane_indices[c]["Cs"]] 
            rest = [j for j in planes if j not in corners]         
            rest_atoms = [x for r in rest for x in plane_indices[r]["Cs"]]

            # limit to 7 corner atoms if 2x2x2 core
            if core.n_cells == 2:
                corner_atoms = corner_atoms[:7]

            to_remove = set(corner_atoms)
            
            n_remove = int(net_charge - len(corner_atoms))

            if n_remove > 0:
                rng = random.Random(random_seed)
                extra_indices = rng.sample(sorted(rest_atoms), k=n_remove)
                to_remove.update(extra_indices)

            # build new Atoms object with those indices removed
            mask = np.ones(len(core.atoms), dtype=bool)
            mask[list(to_remove)] = False
            new_atoms = core.atoms[mask]

            # build new neutral core
            core = cls(A=A, B=B, X=X, atoms=new_atoms, a=a, n_cells=n_cells)
            core._build_binding_sites()

            symbols = core.atoms.get_chemical_symbols()
            n_A = sum(s == A for s in symbols)
            n_B = sum(s == B for s in symbols)
            n_X = sum(s == X for s in symbols)
            assert n_A * 1 + n_B * 2 - n_X * 1 == 0, "Core is not charge neutral!"

        return core
    
    def remove(self, indices: List[int]) -> None:
        """Remove atoms from core by index."""

        mask = np.ones(len(self.atoms), dtype=bool)
        mask[indices] = False
        self.atoms = self.atoms[mask]

        # Recompute surface atoms and binding sites
        self._build_binding_sites()
    
    def to(self, fmt: str = 'xyz', filename: str = None) -> None:
        """Export core structure to file."""

        if filename is None:
            filename = f"{self.A}{self.B}{self.X}3_{self.n_cells}.{fmt}"

        formula = self.atoms.get_chemical_formula()

        write(filename, self.atoms, format=fmt, comment=formula)