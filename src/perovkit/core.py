from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Sequence
from collections import defaultdict

import random
import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.vasp import write_vasp
from scipy.spatial import cKDTree
from ase.data import covalent_radii, atomic_numbers

from .ligand import Ligand, BindingMotif

@dataclass
class Core:
    """
    ABX3 perovskite core. 
    Currently only supports cubic phase for inorganic perovskites.

    Attributes:
        A, B, X (str): Chemical symbols for A-, B-, and X-site species.
        atoms (ASE Atoms): ASE Atoms object of Core
        a (float): Cubic lattice constant (Å).
        supercell (Tuple[int, int, int]): Number of repeating unit cells along x, y, z (nx, ny, nz).
        vacuum (float): Vacuum thickness (Å) along a z-direction (used for Slab).
        octahedra (Dict[int, Dict[str, List[int]]]): Map B-atom index -> {"X": [...], "Ligand": [...]}.
        B_ijk (Dict[int, Tuple[int, int, int]]): Map B-atom index -> integer lattice coordinates.
        build_surface (bool): Whether to compute Core metadata (surface atoms, binding sites, octahedra) on initialization. 
        binding_sites (List[BindingSite]): List of surface binding sites.
    """
    A: str | List[Ligand]
    B: str
    X: str
    atoms: Atoms
    a: float

    supercell: Optional[Tuple[int, int, int]] = None
    vacuum: Optional[float] = None

    octahedra: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    B_ijk: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    build_surface: bool = True
    binding_sites: List[BindingSite] = field(default_factory=list)
    _surface_atoms: Dict[str, np.ndarray] = field(init=False)
    _plane_atoms: Dict[Tuple[int, int, int], Dict[str, List[int]]] = field(default_factory=dict)

    def __post_init__(self):
        if self.supercell is not None:
            if len(self.supercell) != 3:
                raise ValueError(f"supercell must be length-3 (nx, ny, nz); got {self.supercell}")
            self.supercell = tuple(int(x) for x in self.supercell)

        if self.build_surface:
            self._surface_atoms = self._get_surface_atoms()
            self.binding_sites = self._build_binding_sites()
            self._build_octahedra()
            self._build_B_ijk()
        else:
            self._surface_atoms = {}
            self.binding_sites = []
            self.octahedra = {}
            self.B_ijk = {}

    @property
    def is_slab(self) -> bool:
        return any(self.atoms.pbc)

    @property
    def is_nanocrystal(self) -> bool:
        return not any(self.atoms.pbc)

    @property
    def A_label(self) -> str:
        if isinstance(self.A, list):
            lig = self.A[0]
            return lig.name if lig.name is not None else lig.atoms.get_chemical_formula()
        return self.A


    @classmethod
    def build_nanocrystal(
        cls,
        A: str | Ligand,
        B: str,
        X: str,
        a: float,
        supercell: Sequence[int],
        charge_neutral: bool = True,
        random_seed: Optional[int] = None
    ) -> Core:
        """
        Build an AX-terminated ABX3 nanocrystal.

        Args:
            A (str | Ligand): A-site species. A chemical symbol for an inorganic cation, 
                              or a Ligand instance for an organic cation.
            B, X (str): Chemical symbols for B- and X-site species.
            a (float): Cubic lattice constant (Å).
            supercell (Sequence[int]): Number of repeating unit cells along x, y, z.
            charge_neutral (bool): If True, remove surface A cations to ensure charge neutrality
                                   Corners are removed first, then additional surface A cations are randomly removed if needed.
            random_seed (Optional[int]): Seed for choosing which surface A cations to remove.

        Returns:
            A Core instance representing the nanocrystal.
        """
        if len(supercell) != 3:
            raise ValueError(f"supercell must be length-3 (nx, ny, nz); got {supercell}")
        nx, ny, nz = map(int, supercell)

        is_organic_A, A_template, A_label = cls._resolve_A_site(A)

        # A-site is handled separately
        bx_species = [B, X, X, X]
        bx_coords = np.array(
            [
                [0.5, 0.5, 0.5],  # B
                [0.5, 0.5, 0.0],  # X
                [0.5, 0.0, 0.5],  # X
                [0.0, 0.5, 0.5],  # X
            ],
            dtype=float,
        ) * float(a)

        all_symbols: List[str] = []
        all_positions: List[np.ndarray] = []
        a_site_id: List[int] = []
        next_a_id = 0

        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    shift = np.array([i, j, k], dtype=float) * float(a)

                    # A site
                    next_a_id = cls._append_A_site(
                        all_symbols, all_positions, a_site_id,
                        shift, is_organic_A, A_template, next_a_id,
                    )

                    # B and X sites
                    for s, c in zip(bx_species, bx_coords):
                        all_symbols.append(s)
                        all_positions.append(c + shift)
                        a_site_id.append(-1)

        all_positions = np.array(all_positions, dtype=float)
        a_site_id = np.array(a_site_id, dtype=int)

        atoms = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            pbc=False,
        )
        atoms.set_array("a_site_id", a_site_id)

        # Ensure AX termination
        max_coords = np.array([nx, ny, nz], dtype=float) * float(a)
        pos = atoms.positions
        ids = atoms.get_array("a_site_id")
        filt = (ids == -1) & np.any(pos > (max_coords + 1e-5), axis=1)
        atoms = atoms[~filt]

        sc = (nx, ny, nz)

        if is_organic_A:
            A_site = cls._build_A_ligands(atoms, A_template)
        else:
            A_site = A_label

        core = cls(A=A_site, B=B, X=X, atoms=atoms, a=float(a), supercell=sc)

        if charge_neutral:
            core._neutralize(random_seed)

        return core

    @classmethod
    def build_slab(
        cls,
        A: str | Ligand,
        B: str,
        X: str,
        a: float,
        supercell: Sequence[int],
        vacuum: float = 15.0,
    ) -> Core:
        """
        Build an ABX3 slab periodic in x,y with vacuum along z.

        Args:
            A (str | Ligand): A-site species. A chemical symbol for an inorganic cation, 
                              or a Ligand instance for an organic cation.
            B, X (str): Chemical symbols for B- and X-site species.
            a (float): Cubic lattice constant (Å).
            supercell (Sequence[int]): Number of repeating unit cells along x, y, z.
            vacuum (float): Vacuum thickness (Å) along a z-direction.

        Returns:
            A Core instance representing the slab.
        """
        if len(supercell) != 3:
            raise ValueError(f"supercell must be length-3 (nx, ny, nz); got {supercell}")
        nx, ny, nz = map(int, supercell)
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(f"supercell entries must be positive; got {supercell}")

        is_organic_A, A_template, A_label = cls._resolve_A_site(A)

        if is_organic_A:
            symbols = [B, X, X, X]
            scaled = np.array(
                [
                    [0.5, 0.5, 0.5],  # B
                    [0.5, 0.5, 0.0],  # X
                    [0.5, 0.0, 0.5],  # X
                    [0.0, 0.5, 0.5],  # X
                ],
                dtype=float,
            )
        else:
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
        keep = pos[:, 2] <= (z_cut + 1e-5)
        atoms = atoms[keep]

        if is_organic_A:
            n_framework = len(atoms)
            a_symbols: List[str] = []
            a_positions: List[np.ndarray] = []
            a_ids: List[int] = []
            next_a_id = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz + 1):
                        center = np.array([i, j, k], dtype=float) * float(a)
                        if center[2] > z_cut + 1e-5:
                            continue
                        next_a_id = cls._append_A_site(
                            a_symbols, a_positions, a_ids,
                            center, True, A_template, next_a_id,
                        )

            atoms += Atoms(symbols=a_symbols, positions=np.array(a_positions, dtype=float))
            ids = np.array([-1] * n_framework + a_ids, dtype=int)
            atoms.set_array("a_site_id", ids)

        atoms.set_cell([nx * float(a), ny * float(a), nz * float(a) + float(vacuum)])
        atoms.pbc = [True, True, False]

        if is_organic_A:
            A_site = cls._build_A_ligands(atoms, A_template)
        else:
            A_site = A_label

        slab = cls(
            A=A_site,
            B=B,
            X=X,
            atoms=atoms,
            a=float(a),
            supercell=(nx, ny, nz),
            vacuum=float(vacuum),
        )
        return slab


    @staticmethod
    def _resolve_A_site(A: str | Ligand) -> Tuple[bool, object, str]:
        if isinstance(A, Ligand):
            template = A.clone()
            template.atoms.positions -= template.atoms.get_positions().mean(axis=0)
            label = A.name if A.name is not None else template.atoms.get_chemical_formula()
            return True, template, label
        return False, A, A


    @staticmethod
    def _append_A_site(
        symbols: List[str],
        positions: List[np.ndarray],
        ids: List[int],
        center: np.ndarray,
        is_organic: bool,
        A_template: object,
        next_id: int,
    ) -> int:
        if is_organic:
            for sym, rel in zip(A_template.atoms.get_chemical_symbols(),
                                A_template.atoms.get_positions()):
                symbols.append(sym)
                positions.append(center + rel)
                ids.append(next_id)
        else:
            symbols.append(A_template)
            positions.append(center)
            ids.append(next_id)
        return next_id + 1


    def _remove(self, indices: Sequence[int]) -> None:
        indices = np.asarray(list(indices), dtype=int)
        if indices.size == 0:
            return

        mask = np.ones(len(self.atoms), dtype=bool)
        mask[indices] = False
        self.atoms = self.atoms[mask]

        if isinstance(self.A, list):
            template = self.A[0] if self.A else None
            self.A = self._build_A_ligands(self.atoms, template) if template else []

        if self.build_surface:
            self._surface_atoms = self._get_surface_atoms()
            self.binding_sites = self._build_binding_sites()
            self._build_octahedra()
            self._build_B_ijk()


    def _A_sites(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        positions = self.atoms.get_positions()
        if isinstance(self.A, list):
            return [(positions[np.asarray(lig.indices, dtype=int)].mean(axis=0),
                     np.asarray(lig.indices, dtype=int)) for lig in self.A]

        symbols = np.array(self.atoms.get_chemical_symbols())
        a_idx = np.where(symbols == self.A)[0]
        return [(positions[i], np.array([i], dtype=int)) for i in a_idx]


    def _neutralize(self, random_seed: Optional[int] = None) -> None:
        """
        Remove whole surface A-site cations until the Core is charge neutral.

        Each A-site cation carries a formal charge of +1, B is +2 and X is -1.
        Corner cations are removed first, then additional surface cations are
        removed at random if needed.
        """
        symbols = np.array(self.atoms.get_chemical_symbols())
        cations = self._A_sites()

        net_charge = len(cations) * 1 + int(np.sum(symbols == self.B)) * 2 - int(np.sum(symbols == self.X)) * 1
        if net_charge <= 0:
            return

        centers = np.array([c[0] for c in cations])
        pbc = self.atoms.pbc
        non_periodic = [ax for ax in range(3) if not pbc[ax]]
        has_periodic = any(pbc)
        mins, maxs = centers.min(0), centers.max(0)

        tol = 1e-3
        corner_ids, rest_ids = [], []
        for i, c in enumerate(centers):
            plane = self._surface_plane(c, mins, maxs, non_periodic, has_periodic, tol)
            nz = np.count_nonzero(plane)
            if non_periodic and nz == len(non_periodic):
                corner_ids.append(i)
            elif nz > 0:
                rest_ids.append(i)

        n_remove = int(net_charge)
        chosen = list(corner_ids[:n_remove])
        if len(chosen) < n_remove:
            rng = random.Random(random_seed)
            chosen += rng.sample(sorted(rest_ids), k=n_remove - len(chosen))

        self._remove(np.concatenate([cations[i][1] for i in chosen]))

        symbols = np.array(self.atoms.get_chemical_symbols())
        n_A = len(self._A_sites())
        n_B = int(np.sum(symbols == self.B))
        n_X = int(np.sum(symbols == self.X))
        assert n_A * 1 + n_B * 2 - n_X * 1 == 0, "Core is not charge neutral!"


    @staticmethod
    def _build_A_ligands(atoms: Atoms, template: Ligand) -> List[Ligand]:
        ids = atoms.get_array("a_site_id")
        ligands: List[Ligand] = []
        for new_id, mol_id in enumerate(sorted(np.unique(ids[ids >= 0]))):
            idx = np.where(ids == mol_id)[0]
            lig = template.clone()
            lig.atoms = atoms[idx]
            lig.indices = idx
            lig.id = new_id
            ligands.append(lig)
        return ligands


    def a_site_metadata(self) -> str | dict:
        """
        Serialize the A-site for JSON.

        Returns the element symbol for an inorganic A-site, or a dict carrying the
        molecular cation's Ligand information (one type, repeated `n_instances`
        times) for an organic A-site. Inverse of `a_site_from_metadata`.
        """
        if not isinstance(self.A, list):
            return self.A

        lig = self.A[0]
        motif = list(lig.binding_motif.atoms) if lig.binding_motif else None
        return {
            "label": self.A_label,
            "kind": "molecular",
            "n_instances": len(self.A),
            "ligand": {
                "name": lig.name,
                "smiles": lig.smiles,
                "charge": int(lig.charge),
                "binding_motif_atoms": motif,
                "binding_atoms_indices": list(getattr(lig, "binding_atoms", [])),
                "n_atoms": len(lig.atoms),
                "volume": float(lig.volume) if lig.volume is not None else None,
            },
        }


    @staticmethod
    def a_site_from_json(meta: str | dict, atoms: Atoms):
        """
        Reconstruct the A-site value from `a_site_metadata`.

        For an inorganic A-site returns the element symbol. For an organic A-site
        returns a list of placed Ligand instances and sets the "a_site_id" array
        on `atoms` (A-site molecules are the first, contiguously grouped atoms).
        """
        if not isinstance(meta, dict):
            return str(meta)

        lmeta = meta["ligand"]
        n_inst = int(meta["n_instances"])
        n_atoms = int(lmeta["n_atoms"])

        ids = np.full(len(atoms), -1, dtype=int)
        for g in range(n_inst):
            ids[g * n_atoms:(g + 1) * n_atoms] = g
        atoms.set_array("a_site_id", ids)

        motif_atoms = lmeta.get("binding_motif_atoms")
        template = Ligand._from_data(
            atoms=atoms[0:n_atoms],
            mol=None,
            smiles=lmeta["smiles"],
            charge=lmeta["charge"],
            binding_motif=BindingMotif(motif_atoms) if motif_atoms else None,
            name=lmeta["name"],
            volume=lmeta.get("volume"),
            binding_atoms=list(map(int, lmeta.get("binding_atoms_indices", []))),
        )
        return Core._build_A_ligands(atoms, template)


    def perturb(
        self, 
        bound: List[float, float], 
        random_seed: int= None
    ):
        """
        Randomly displace every atom by a fraction of its covalent radius.

        Args:
            bound (List[float, float]): bounds on the random displacement as a fraction of the covalent radius (e.g. [0.0, 0.1] for up to 10% displacement).
            random_seed (int): Optional random seed
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
            rand_uniform = rng.uniform
        else:
            rand_uniform = np.random.uniform

        lo, hi = float(bound[0]), float(bound[1])
        if lo < 0 or hi <= 0 or hi < lo:
            raise ValueError(f"Bound must satisfy 0 <= low <= high, got {bound}")

        symbols = self.atoms.get_chemical_symbols()
        radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols], dtype=float)
        mags = rand_uniform(radii * lo, radii * hi)

        dirs = rand_uniform(-1.0, 1.0, size=(len(self.atoms), 3))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs /= norms

        pos = self.atoms.get_positions()
        self.atoms.set_positions(pos + dirs * mags[:, None])


    def apply_tilt(
        self, 
        glazer: str, 
        angles: Tuple[float, float, float], 
        order: str = "xyz"
    ):
        """
        Apply a Glazer octahedral tilt.

        Args:
            glazer: Glazer notation string (e.g. "a+b-c-").
            angles: Tilt angles (deg) about x, y, z axes.
            order: Rotation order (e.g. "xyz").
        """
        from .tilt import apply_tilt
        apply_tilt(structure=self, glazer=glazer, angles=angles, order=order)


    def apply_strain(
        self, 
        strain: Sequence[float]
    ):
        """
        Apply an axial strain.

        Args:
            strain (Sequence[float]): (ex, ey, ez) fractional strains along each axis.
        """
        from .strain import apply_strain
        apply_strain(structure=self, strain=strain, strain_ligands=False)


    def to(
        self, 
        fmt: str, 
        filename: Optional[str] = None, 
        vacuum: float = 15.0
    ):
        """
        Write the structure to file.

        Args:
            fmt: File format
            filename: Output path
            vacuum: Vacuum padding (Å) for nanocrystal VASP output.
        """
        if filename is None:
            nx, ny, nz = self.supercell or (0, 0, 0)
            filename = f"{self.A_label}{self.B}{self.X}3_{nx}x{ny}x{nz}.{fmt}"

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "vasp":
            if self.is_slab:
                write_vasp(str(path), self.atoms, sort=True, direct=True)
            else:
                pos = self.atoms.get_positions()
                center = pos.mean(axis=0)
                extent = pos.max(axis=0) - pos.min(axis=0)
                cell_diag = extent + vacuum

                vasp_atoms = self.atoms.copy()
                vasp_atoms.set_cell(np.diag(cell_diag))
                vasp_atoms.positions += (cell_diag / 2 - center)
                vasp_atoms.pbc = True

                write_vasp(str(path), vasp_atoms, sort=True, direct=True)
        else:
            formula = self.atoms.get_chemical_formula()
            write(str(path), self.atoms, format=fmt, comment=formula)


    def _get_surface_atoms(
        self, 
    ) -> Dict[str, np.ndarray]:
        """
        Identify A- and X-site atoms on non-periodic surfaces.

        Args:
            tol: Coordinate tolerance (Å)

        Returns:
            surface_indices (Dict): Map element symbol -> array of surface atom indices
        """
        surface_indices: Dict[str, np.ndarray] = {}

        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        symbols = np.array(self.atoms.get_chemical_symbols())
        pbc = self.atoms.pbc
        non_periodic = [i for i in range(3) if not pbc[i]]
        has_periodic = any(pbc)

        # Organic A-sites (list of Ligands) are not atomic surface sites; only
        # consider the A symbol when it is an inorganic element.
        elements = [self.X] if isinstance(self.A, list) else [self.A, self.X]
        for element in elements:
            elem_global = np.where(symbols == element)[0]
            elem_pos = positions[elem_global]

            if elem_pos.size == 0:
                surface_indices[element] = np.array([], dtype=int)
                continue

            surface_flags = np.zeros(len(elem_pos), dtype=bool)
            tol = 1e-3
            for ax in non_periodic:
                ax_max = elem_pos[:, ax].max()
                surface_flags |= np.isclose(elem_pos[:, ax], ax_max, atol=tol)
                if not has_periodic:
                    ax_min = elem_pos[:, ax].min()
                    surface_flags |= np.isclose(elem_pos[:, ax], ax_min, atol=tol)

            surface_indices[element] = elem_global[surface_flags].astype(int)

        return surface_indices


    @staticmethod
    def _surface_plane(point, mins, maxs, non_periodic, has_periodic, tol=1e-3):
        """
        Classify a point onto a surface Miller plane (h, k, l).

        Along each non-periodic axis: +1 if the point is at the max, -1 if at the
        min (only for a fully non-periodic structure), else 0. Periodic axes are 0.
        """
        plane = [0, 0, 0]
        for ax in non_periodic:
            if np.isclose(point[ax], maxs[ax], atol=tol):
                plane[ax] = 1
            elif not has_periodic and np.isclose(point[ax], mins[ax], atol=tol):
                plane[ax] = -1
        return tuple(plane)


    def _build_binding_sites(self) -> List[BindingSite]:
        """
        Assign each surface atom a Miller index and return BindingSite list.
        """
        surface = self._get_surface_atoms()
        positions = np.array([a.position for a in self.atoms], dtype=float)
        symbols = np.array([a.symbol for a in self.atoms])
        pbc = self.atoms.pbc
        non_periodic = [i for i in range(3) if not pbc[i]]
        has_periodic = any(pbc)

        tol = 1e-3
        plane_indices = defaultdict(lambda: defaultdict(list))

        for elem, idxs in surface.items():
            elem_global = np.where(symbols == elem)[0]
            elem_pos = positions[elem_global]
            if elem_pos.size == 0:
                continue

            mins, maxs = elem_pos.min(0), elem_pos.max(0)

            for i in idxs:
                v = self._surface_plane(positions[int(i)], mins, maxs, non_periodic, has_periodic, tol)
                if np.count_nonzero(v) == 0:
                    continue

                plane_indices[v][elem].append(int(i))

        self._plane_atoms = {
            hkl: {elem: idxs for elem, idxs in elems.items()}
            for hkl, elems in plane_indices.items()
        }

        idx_to_site: Dict[int, BindingSite] = {}
        for plane, elem_map in self._plane_atoms.items():
            for elem, indices in elem_map.items():
                for idx in indices:
                    idx = int(idx)
                    if idx in idx_to_site:
                        continue
                    idx_to_site[idx] = BindingSite(index=idx, symbol=elem, plane=plane,
                                                   position=positions[idx], passivated=False)

        sites = list(idx_to_site.values())

        # Organic A-site cations are molecules, not single atoms: expose each
        # surface molecule as one (molecular) binding site.
        if isinstance(self.A, list):
            sites.extend(
                self._build_A_molecule_sites(non_periodic, has_periodic, tol)
            )

        return sites


    def _build_A_molecule_sites(
        self,
        non_periodic: List[int],
        has_periodic: bool,
        tol: float,
    ) -> List[BindingSite]:
        """
        Build one molecular BindingSite per surface A-site cation (organic A only).
        """
        a_sites = self._A_sites()
        if not a_sites:
            return []

        centers = np.array([center for center, _ in a_sites])
        mins, maxs = centers.min(0), centers.max(0)

        sites: List[BindingSite] = []
        for center, idx in a_sites:
            v = self._surface_plane(center, mins, maxs, non_periodic, has_periodic, tol)
            if np.count_nonzero(v) == 0:
                continue  # interior molecule, not on a surface

            sites.append(
                BindingSite(index=[int(i) for i in idx], symbol=self.A_label,
                            plane=v, position=center, passivated=False)
            )

        return sites


    def _build_octahedra(self):
        """
        Build the octahedral network by finding nearest X neighbors of each B.
        """
        at = self.atoms
        syms = np.array(at.get_chemical_symbols())

        b_idx = np.where(syms == self.B)[0]
        x_idx = np.where(syms == self.X)[0]

        if len(b_idx) == 0 or len(x_idx) == 0:
            self.octahedra = {}
            return

        r_cut = float(self.a)/2 + 1e-2

        if any(at.pbc):
            cell = at.get_cell()
            Lx, Ly, Lz = cell.lengths()

            scaled = at.get_scaled_positions(wrap=True)
            pos = scaled @ cell.array

            B_pos = pos[b_idx]
            X_pos = pos[x_idx]

            tree = cKDTree(X_pos, boxsize=(Lx, Ly, Lz))
        else:
            pos = at.get_positions()
            B_pos = pos[b_idx]
            X_pos = pos[x_idx]

            tree = cKDTree(X_pos)

        neigh_lists = tree.query_ball_point(B_pos, r_cut)

        octahedra: Dict[int, Dict[str, List[int]]] = {}
        for b_loc, x_local_list in enumerate(neigh_lists):
            b_abs = int(b_idx[b_loc])
            x_abs_list = [int(x_idx[j]) for j in x_local_list]
            octahedra[b_abs] = {"X": x_abs_list, "Ligand": []}

        self.octahedra = octahedra


    def _build_B_ijk(self):
        """
        Assign integer (i, j, k) lattice coordinates to each B atom.
        """
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


@dataclass
class BindingSite:
    """
    Surface site available for ligand binding (a single atom or a whole molecule).

    Attributes:
        index (int | List[int]): Atom index for a single-atom site, or the list of
            atom indices for a molecular site (e.g. an organic A-site cation).
        symbol (str): Chemical symbol of the site atom, or the A-site label
            (e.g. "MA") for a molecular A-site.
        plane (Tuple[int, int, int]): Miller index (h, k, l) indicating the surface plane.
        position (np.ndarray): Site position (Å): the atom position for a single-atom
            site, or the molecule center for a molecular site.
        passivated (bool): Whether a ligand is attached to this site.
    """
    index: int | List[int]
    symbol: str
    plane: Tuple[int, int, int]
    position: np.ndarray
    passivated: bool = False

    @property
    def is_molecular(self) -> bool:
        return isinstance(self.index, list)

    @property
    def atom_indices(self) -> List[int]:
        """All atom indices of this site (one for atomic, many for molecular)."""
        return [int(i) for i in self.index] if self.is_molecular else [int(self.index)]