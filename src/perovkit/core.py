from __future__ import annotations
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Dict, Tuple, List, Optional, Sequence
from collections import defaultdict

import random
import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree
from ase.data import covalent_radii, atomic_numbers

from .io import write_structure
from .ligand import Ligand, BindingMotif, LigandSpec
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v

@dataclass
class Core:
    """
    ABX3 perovskite core. 
    Currently only supports cubic phase for inorganic perovskites.

    Attributes:
        A (str | List[Ligand]): Chemical symbol of the A-site, or its organic cations.
        B, X (str): Chemical symbols of the B- and X-site species.
        atoms (ASE Atoms): ASE Atoms object of Core
        a (float): Cubic lattice constant (Å).
        supercell (Tuple[int, int, int]): Number of repeating unit cells along x, y, z (nx, ny, nz).
        vacuum (float): Vacuum thickness (Å) along a z-direction (used for slabs).
        octahedra (Dict[int, Dict[str, List[int]]]): Map B-atom index -> {"X": [...], "Ligand": [...]}.
        B_ijk (Dict[int, Tuple[int, int, int]]): Map B-atom index -> integer lattice coordinates.
        build_surface (bool): Whether to find the binding sites and octahedra on initialization.
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

    build_surface: InitVar[bool] = True
    binding_sites: List[BindingSite] = field(default_factory=list)

    def __post_init__(self, build_surface: bool):
        if build_surface:
            self._build_surface()

    def _build_surface(self) -> None:
        self.binding_sites = self._build_binding_sites()
        self._build_octahedra()

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
    def build_core(
        cls,
        A: str | Ligand,
        B: str,
        X: str,
        a: float,
        supercell: Sequence[int],
        pbc: bool | Sequence[bool],
        vacuum: float = 15.0,
        charge_neutral: bool = True,
        random_seed: Optional[int] = 42,
        A_site_disorder: str = "balanced",
    ) -> Core:
        """
        Build an ABX3 core periodic along the axes given by `pbc` (as in ASE, one bool for
        all three or one per axis): an AX-terminated nanocrystal for False, a slab periodic in
        x, y with vacuum along z for (True, True, False).

        Args:
            A (str | Ligand): A-site species. A chemical symbol for an inorganic cation, 
                              or a Ligand instance for an organic cation.
            B, X (str): Chemical symbols for B- and X-site species.
            a (float): Cubic lattice constant (Å).
            supercell (Sequence[int]): Number of repeating unit cells along x, y, z.
            pbc (bool | Sequence[bool]): Periodicity along x, y, z: False or (True, True, False).
            vacuum (float): Vacuum thickness (Å) along z (slab only).
            charge_neutral (bool): Remove surface A cations to ensure charge neutrality, corners
                                   first, then others at random (nanocrystal only).
            random_seed (Optional[int]): Seed for choosing which surface A cations to remove
                                         (nanocrystal only) and for `A_site_disorder`.
            A_site_disorder (str): Orientation of organic A cations, each about its own center
                                   and with a random spin about its axis (+z of the Ligand as
                                   it was passed). "balanced": random axes, half of them the
                                   opposite of the other half, which leaves no net dipole
                                   along them. "random": random axes. "none": all as passed.

        Returns:
            A Core instance.
        """
        periodic = np.broadcast_to(np.asarray(pbc, dtype=bool), 3)
        if periodic.any() and periodic.tolist() != [True, True, False]:
            raise ValueError(f"pbc must be False (nanocrystal) or (True, True, False) (slab); got {pbc}")
        if len(supercell) != 3 or min(supercell) <= 0:
            raise ValueError(f"supercell must be three positive integers (nx, ny, nz); got {supercell}")
        if A_site_disorder not in ("balanced", "random", "none"):
            raise ValueError(f"A_site_disorder must be 'balanced', 'random' or 'none'; got {A_site_disorder!r}")
        n_cells = np.array(supercell, dtype=int)
        a = float(a)

        is_organic = isinstance(A, Ligand)
        if is_organic:
            template = A.clone()
            template.atoms.positions -= template.atoms.get_positions().mean(axis=0)
            A_symbols, A_coords = template.atoms.get_chemical_symbols(), template.atoms.get_positions()
        else:
            A_symbols, A_coords = [A], np.zeros((1, 3))

        symbols = A_symbols + [B, X, X, X]
        unit = np.vstack([
            A_coords,
            np.array(
                [
                    [0.5, 0.5, 0.5],  # B
                    [0.5, 0.5, 0.0],  # X
                    [0.5, 0.0, 0.5],  # X
                    [0.0, 0.5, 0.5],  # X
                ],
                dtype=float,
            ) * a,
        ])

        # A non-periodic axis gets one more layer of cells for AX termination, with the B
        # and X past its last A plane left out
        open_axes = ~periodic
        grid = np.meshgrid(*[np.arange(m) for m in n_cells + open_axes], indexing="ij")
        corners = np.stack(grid, axis=-1).reshape(-1, 3) * a
        positions = (unit + corners[:, None]).reshape(-1, 3)

        is_A = np.tile(np.arange(len(symbols)) < len(A_symbols), len(corners))
        past_end = positions[:, open_axes] > (n_cells * a)[open_axes] + 1e-5
        keep = is_A | ~past_end.any(axis=1)
        atoms = Atoms(symbols=symbols * len(corners), positions=positions)[keep]

        if is_organic:
            ids = np.repeat(np.arange(len(corners)), len(symbols))
            ids[~is_A] = -1
            atoms.set_array("a_site_id", ids[keep])

        if periodic.any():
            atoms.set_cell(n_cells * a + [0.0, 0.0, float(vacuum)])
            atoms.pbc = periodic

        core = cls(
            A=cls._build_A_ligands(atoms, template) if is_organic else A,
            B=B,
            X=X,
            atoms=atoms,
            a=a,
            supercell=tuple(n_cells.tolist()),
            vacuum=float(vacuum) if periodic.any() else None,
            build_surface=False,
        )

        if charge_neutral and not periodic.any():
            core._neutralize(random_seed)
        if is_organic and A_site_disorder != "none":
            core._disorder_A_sites(A_site_disorder, random_seed)
        core._build_surface()

        return core


    def _A_sites(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Center and atom indices of every A-site cation.
        """
        positions = self.atoms.get_positions()
        if isinstance(self.A, list):
            return [(positions[lig.indices].mean(axis=0), lig.indices) for lig in self.A]

        symbols = np.array(self.atoms.get_chemical_symbols())
        a_idx = np.where(symbols == self.A)[0]
        return [(positions[i], np.array([i], dtype=int)) for i in a_idx]


    def _neutralize(self, random_seed: Optional[int] = None) -> None:
        """
        Remove whole surface A-site cations until the nanocrystal is charge neutral.

        Each A-site cation carries a formal charge of +1, B is +2 and X is -1.
        Corner cations are removed first, then additional surface cations are
        removed at random if needed.
        """
        symbols = np.array(self.atoms.get_chemical_symbols())
        cations = self._A_sites()

        n_remove = int(len(cations) + 2 * np.sum(symbols == self.B) - np.sum(symbols == self.X))
        if n_remove <= 0:
            return

        n_faces = np.count_nonzero(self._surface_planes(np.array([c[0] for c in cations])), axis=1)
        chosen = np.flatnonzero(n_faces == 3).tolist()[:n_remove]
        if len(chosen) < n_remove:
            rest = np.flatnonzero((n_faces > 0) & (n_faces < 3)).tolist()
            chosen += random.Random(random_seed).sample(rest, k=n_remove - len(chosen))

        keep = np.ones(len(self.atoms), dtype=bool)
        keep[np.concatenate([cations[i][1] for i in chosen])] = False
        self.atoms = self.atoms[keep]

        if isinstance(self.A, list):
            self.A = self._build_A_ligands(self.atoms, self.A[0])


    def _disorder_A_sites(self, mode: str, random_seed: Optional[int]) -> None:
        """
        Turn every organic A-site cation about its center: its +z axis to a random direction,
        with an odd one out for "balanced" if the number of cations is odd, and by a random
        spin about that axis.
        """
        rng = np.random.default_rng(random_seed)
        cations = self._A_sites()

        axes = rng.normal(size=(len(cations), 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        if mode == "balanced":
            half = len(cations) // 2
            axes[half:2 * half] = -axes[:half]
            rng.shuffle(axes)

        positions = self.atoms.get_positions()
        z = np.array([0.0, 0.0, 1.0], dtype=float)
        for (center, idx), axis, spin in zip(cations, axes, rng.uniform(0.0, 2.0 * np.pi, len(cations))):
            R = rotation_about_axis(axis, spin) @ rotation_from_u_to_v(z, axis)
            positions[idx] = center + (positions[idx] - center) @ R.T

        self.atoms.set_positions(positions)
        self.A = self._build_A_ligands(self.atoms, self.A[0])


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
        times) for an organic A-site. Inverse of `a_site_from_json`.
        """
        if not isinstance(self.A, list):
            return self.A

        lig = self.A[0]
        return {
            "label": self.A_label,
            "kind": "molecular",
            "n_instances": len(self.A),
            "ligand": {
                **lig._type_metadata(),
                "n_atoms": len(lig.atoms),
                "volume": float(lig.volume) if lig.volume is not None else None,
            },
        }


    @staticmethod
    def a_site_from_json(meta: str | dict, atoms: Atoms, B: str, X: str):
        """
        Reconstruct the A-site value from `a_site_metadata`.

        For an inorganic A-site returns the element symbol. For an organic A-site returns a
        list of placed Ligand instances and sets the "a_site_id" array on `atoms`: the
        cations are the atoms that are neither B nor X, one molecule after another.
        """
        if not isinstance(meta, dict):
            return str(meta)

        lmeta = meta["ligand"]
        n_atoms = int(lmeta["n_atoms"])

        cation_atoms = np.flatnonzero(~np.isin(atoms.numbers, [atomic_numbers[B], atomic_numbers[X]]))
        ids = np.full(len(atoms), -1, dtype=int)
        ids[cation_atoms] = np.arange(len(cation_atoms)) // n_atoms
        atoms.set_array("a_site_id", ids)

        template = Ligand._from_data(
            atoms=atoms[cation_atoms[:n_atoms]],
            mol=None,
            smiles=lmeta["smiles"],
            charge=lmeta["charge"],
            binding_motif=BindingMotif._from_metadata(lmeta),
            name=lmeta["name"],
            volume=lmeta.get("volume"),
            binding_atoms=list(map(int, lmeta.get("binding_atoms_indices", []))),
        )
        return Core._build_A_ligands(atoms, template)


    def perturb(
        self, 
        bound: Sequence[float], 
        random_seed: int= None
    ):
        """
        Randomly displace every atom by a fraction of its covalent radius.

        Args:
            bound (Sequence[float]): bounds on the random displacement as a fraction of the covalent radius (e.g. [0.0, 0.1] for up to 10% displacement).
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


    def place_ligands(self, ligand_specs: Sequence[LigandSpec], **kwargs) -> Assembly:
        """
        Place ligands on a copy of this core, which is left as it is: sites by coverage
        (or as given), then turns about the surface normals until no atoms of different
        ligands, or of a ligand and the core, are within `overlap_cutoff`.

        Args:
            ligand_specs: The ligands with their coverages or sites.
            **kwargs: Options of the placement: max_iters (10), overlap_cutoff (1.6 Å),
                coarse_step_deg (18), fine_step_deg (2), window_deg (12),
                active_radius_factor (sqrt(2)/2), random_seed (42), verbose (True); for
                multi-motif ligands also max_anchor_rmsd (0.5 Å), refine_torsions (True),
                stretch_tails (True), n_workers (32, at most the CPUs available), and any
                option of ConstellationSampler (e.g. n_refine, max_templates).

        Returns:
            Assembly: the core with its ligands.
        """
        from .assembly import Assembly
        assembly = Assembly(core=deepcopy(self), ligand_specs=list(ligand_specs))
        assembly._place_ligands(**kwargs)
        return assembly


    def place_ligands_manually(self, ligand_specs: Sequence[LigandSpec], **kwargs) -> Assembly:
        """
        Place ligands on a copy of this core, which is left as it is, without the
        rotation search: cations shifted by cat_x/y/z_move (Å) and turned by cat_z_rotate
        (degrees) about the surface normal, anions by the an_* values. Single-motif
        ligands only.

        Args:
            ligand_specs: The ligands with their coverages or sites.
            **kwargs: cat_x_move, cat_y_move, cat_z_move, cat_z_rotate, an_x_move,
                an_y_move, an_z_move, an_z_rotate (all 0), ligand_ids (None: every
                ligand; else only those ids, counted from 0 in placement order, are
                shifted and turned), random_seed (42), verbose (True).

        Returns:
            Assembly: the core with its ligands.
        """
        from .assembly import Assembly
        assembly = Assembly(core=deepcopy(self), ligand_specs=list(ligand_specs))
        assembly._place_ligands_manually(**kwargs)
        return assembly


    def to(
        self, 
        fmt: str, 
        filename: Optional[str] = None, 
        vacuum: float = 15.0,
        **kwargs,
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

        write_structure(self.atoms, filename, fmt, box=self.is_nanocrystal, vacuum=vacuum, **kwargs)


    def _surface_planes(self, points: np.ndarray) -> np.ndarray:
        """
        Surface Miller plane (h, k, l) of each of `points`: (N, 3).

        Along a non-periodic axis a point has +1 if it is among the highest of `points`, -1
        if among the lowest (only for a fully non-periodic structure), else 0. Periodic axes
        are 0.
        """
        pbc = self.atoms.pbc
        planes = np.zeros(points.shape, dtype=int)
        for ax in np.flatnonzero(~pbc):
            if not pbc.any():
                planes[np.isclose(points[:, ax], points[:, ax].min(), atol=1e-3), ax] = -1
            planes[np.isclose(points[:, ax], points[:, ax].max(), atol=1e-3), ax] = 1
        return planes


    def _build_binding_sites(self) -> List[BindingSite]:
        """
        One BindingSite per surface atom, plane by plane. Organic A-site cations are
        molecules, not single atoms: each surface one is a (molecular) site of its own.
        """
        positions = self.atoms.get_positions()
        symbols = np.array(self.atoms.get_chemical_symbols())
        organic = isinstance(self.A, list)

        by_plane = defaultdict(lambda: defaultdict(list))
        for elem in [self.X] if organic else [self.A, self.X]:
            elem_atoms = np.flatnonzero(symbols == elem)
            planes = self._surface_planes(positions[elem_atoms])
            for i, plane in zip(elem_atoms.tolist(), planes.tolist()):
                if any(plane):
                    by_plane[tuple(plane)][elem].append(i)

        sites = [BindingSite(index=i, symbol=elem, plane=plane, position=positions[i])
                 for plane, elems in by_plane.items()
                 for elem, indices in elems.items()
                 for i in indices]

        if organic:
            a_sites = self._A_sites()
            planes = self._surface_planes(np.array([center for center, _ in a_sites]))
            sites.extend(BindingSite(index=idx.tolist(), symbol=self.A_label, plane=tuple(plane), position=center)
                         for (center, idx), plane in zip(a_sites, planes.tolist()) if any(plane))

        return sites


    def _box_positions(self) -> Tuple[np.ndarray, Optional[tuple]]:
        """
        Atom positions and the `boxsize` for a KD-tree of them: wrapped into the cell of a
        periodic structure, as they are otherwise.
        """
        if self.is_nanocrystal:
            return self.atoms.get_positions(), None
        cell = self.atoms.get_cell()
        return self.atoms.get_scaled_positions(wrap=True) @ cell.array, tuple(cell.lengths())


    def _build_octahedra(self) -> None:
        """
        Build the octahedral network: the nearest X neighbors of each B, and the integer
        (i, j, k) lattice coordinates of each B.
        """
        syms = np.array(self.atoms.get_chemical_symbols())
        b_idx = np.where(syms == self.B)[0]
        x_idx = np.where(syms == self.X)[0]

        pos, boxsize = self._box_positions()
        tree = cKDTree(pos[x_idx], boxsize=boxsize)
        neigh_lists = tree.query_ball_point(pos[b_idx], float(self.a)/2 + 1e-2)

        self.octahedra = {int(b): {"X": [int(x_idx[j]) for j in neighbors], "Ligand": []}
                          for b, neighbors in zip(b_idx, neigh_lists)}

        b_pos = self.atoms.positions[b_idx]
        ijk = np.rint((b_pos - b_pos.min(axis=0)) / float(self.a)).astype(int)
        self.B_ijk = {int(b): tuple(v) for b, v in zip(b_idx, ijk.tolist())}


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
    def atom_indices(self) -> List[int]:
        """All atom indices of this site (one for atomic, many for molecular)."""
        return self.index if isinstance(self.index, list) else [self.index]