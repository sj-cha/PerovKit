from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Sequence
from collections import defaultdict

import random
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers

from .io.structure import write_structure
from .ligand import Ligand
from .utils.box import Box
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v
from .utils.selection import Part, PartAtoms

if TYPE_CHECKING:
    from .assembly import Assembly
    from .config import PlacementConfig

@dataclass
class Core(Part):
    """
    Cubic ABX3 perovskite core, a nanocrystal or a slab. Make one with build_core.

    Attributes:
        A (str | List[Ligand]): A-site element, or the organic A-site cations.
        B (str): B-site element.
        X (str): X-site element.
        atoms (Atoms): Atoms of the core (read-only inside an Assembly).
        a (float): Cubic lattice constant (Å).
        supercell (Tuple[int, int, int]): Unit cells along x, y and z.
        vacuum (float): Vacuum (Å) along z of a slab.
        octahedra (Dict[int, dict]): B atom index -> {"X": X indices, "Ligand": ligand ids}.
        B_ijk (Dict[int, Tuple[int, int, int]]): B atom index -> lattice coordinates.
        binding_sites (List[BindingSite]): Surface sites that ligands bind to.
    """
    A: str | List[Ligand]
    B: str
    X: str
    atoms: Atoms = PartAtoms()
    a: float

    supercell: Optional[Tuple[int, int, int]] = None
    vacuum: Optional[float] = None

    octahedra: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    B_ijk: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    build_surface: InitVar[bool] = True
    binding_sites: List[BindingSite] = field(default_factory=list)

    def __post_init__(self, build_surface: bool):
        if self.has_organic_A:
            self.A = self._build_A_ligands(self.A[0])
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
    def box(self) -> Box:
        return Box(self.atoms.cell.lengths(), self.atoms.pbc)

    @property
    def site_symbols(self) -> Dict[str, str]:
        """Symbol of the binding sites of each sublattice."""
        return {"A": self.A_label, "B": self.B, "X": self.X}

    @property
    def has_organic_A(self) -> bool:
        """Whether the A sites hold molecular cations (Ligands) rather than an element."""
        return isinstance(self.A, list)

    @property
    def A_label(self) -> str:
        return self.A[0].name if self.has_organic_A else self.A


    @classmethod
    def build_core(
        cls,
        A: str | Ligand,
        B: str,
        X: str,
        a: float,
        supercell: Sequence[int],
        pbc: bool | Sequence[bool],
        vacuum: float = 20.0,
        charge_neutral: bool = True,
        random_seed: Optional[int] = 42,
        A_site_disorder: str = "balanced",
    ) -> Core:
        """
        Build a cubic ABX3 nanocrystal (AX-terminated) or slab.

        Args:
            A (str | Ligand): A-site element, or an organic cation as a Ligand.
            B (str): B-site element.
            X (str): X-site element.
            a (float): Cubic lattice constant (Å).
            supercell (Sequence[int]): Unit cells along x, y and z.
            pbc (bool | Sequence[bool]): False for a nanocrystal; True or (True, True, False)
                for a slab periodic in x and y.
            vacuum (float): Vacuum (Å) along z of a slab.
            charge_neutral (bool): Remove surface A cations, corners first, until a
                nanocrystal is neutral.
            random_seed (int, optional): Seed for the removed cations and A_site_disorder.
            A_site_disorder (str): Orientation of organic A cations: "balanced" (random, no
                net dipole), "random" or "none" (as given).

        Returns:
            Core: The new core.
        """
        flags = np.ravel(pbc)
        if not all(isinstance(p, (bool, np.bool_)) for p in flags) or flags.tolist() not in (
                [False], [True], [True, True, False]) or (len(flags) == 1 and not np.isscalar(pbc)):
            raise ValueError(f"pbc must be False (nanocrystal), or True or (True, True, False) "
                             f"(slab); got {pbc!r}")
        periodic = np.array([True, True, False]) if flags[0] else np.zeros(3, dtype=bool)
        if len(supercell) != 3 or min(supercell) <= 0:
            raise ValueError(f"supercell must be three positive integers (nx, ny, nz); got {supercell}")
        if A_site_disorder not in ("balanced", "random", "none"):
            raise ValueError(f"A_site_disorder must be 'balanced', 'random' or 'none'; "
                             f"got {A_site_disorder!r}")
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
            A=[template] if is_organic else A,
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
        if self.has_organic_A:
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

        if self.has_organic_A:
            self.A = self._build_A_ligands(self.A[0])


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


    def _build_A_ligands(self, template: Ligand) -> List[Ligand]:
        """
        One copy of `template` for every molecule of the "a_site_id" array of the atoms.
        """
        ids = self.atoms.arrays["a_site_id"]
        ligands: List[Ligand] = []
        for new_id, mol_id in enumerate(np.unique(ids[ids >= 0]).tolist()):
            lig = template.clone()
            lig.id = new_id
            lig._attach(self, "a_site_id", mol_id)
            ligands.append(lig)
        return ligands


    def perturb(
        self,
        bound: Sequence[float],
        random_seed: Optional[int] = None,
    ):
        """
        Displace every atom in a random direction, in place.

        Args:
            bound (Sequence[float]): (low, high) displacement as a fraction of each atom's
                covalent radius, e.g. (0.0, 0.1).
            random_seed (int, optional): Seed; None uses numpy's global generator.
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
        Tilt the octahedra by a Glazer pattern, in place.

        Args:
            glazer (str): Glazer notation, e.g. "a+b-c-".
            angles (Tuple[float, float, float]): Tilt angles (deg) about x, y and z.
            order (str): Order of the rotations, e.g. "xyz".
        """
        from .tilt import apply_tilt
        apply_tilt(structure=self, glazer=glazer, angles=angles, order=order)


    def apply_strain(
        self, 
        strain: Sequence[float]
    ):
        """
        Strain the core along x, y and z, in place.

        Args:
            strain (Sequence[float]): Fractional strains (ex, ey, ez).
        """
        from .strain import apply_strain
        apply_strain(structure=self, strain=strain, strain_ligands=False)


    def place_ligands(
        self,
        ligands: Ligand | Sequence[Ligand],
        coverage: float | Sequence,
        config: Optional[PlacementConfig] = None,
        random_seed: int = 42,
        verbose: bool = True,
        n_workers: int = 8,
    ) -> Assembly:
        """
        Place ligands on a copy of this core.

        Sites are chosen by coverage, then the ligands turn and bend until no atoms of two
        ligands, or of a ligand and the core, are closer than `config.overlap_cutoff`. How
        each ligand binds is set by its BindingMotif(s), and neighboring reactive groups
        (ReactiveMotif) are bonded into chains.

        Args:
            ligands (Ligand | Sequence[Ligand]): One ligand, or a list of them.
            coverage (float | Sequence): One entry per ligand (the entry itself for one
                Ligand): up to 1, a fraction of the free sites; above 1, a number of
                ligands; a list, the site atom indices to take (for a multi-motif ligand,
                groups of one index per motif).
            config (PlacementConfig, optional): Every setting that shapes the result;
                PlacementConfig() by default. The assembly records it.
            random_seed (int): Seed.
            verbose (bool): Print the placement log.
            n_workers (int): Processes for the stretch of dangling chains.

        Returns:
            Assembly: The core with its ligands.
        """
        from .placement import place_ligands
        return place_ligands(self, ligands, coverage, config, random_seed, verbose, n_workers)


    def to(
        self, 
        fmt: str, 
        filename: Optional[str] = None, 
        vacuum: float = 20.0,
        **kwargs,
    ):
        """
        Write the core to a file.

        Args:
            fmt (str): File format, e.g. "xyz" or "vasp".
            filename (str, optional): Output path; by default named after the composition
                and supercell.
            vacuum (float): Vacuum (Å) around a nanocrystal written as VASP.
            **kwargs: Passed to the ASE writer.
        """
        if filename is None:
            nx, ny, nz = self.supercell or (0, 0, 0)
            filename = f"{self.A_label}{self.B}{self.X}3_{nx}x{ny}x{nz}.{fmt}"

        write_structure(self.atoms, filename, fmt, center_in_box=self.is_nanocrystal, vacuum=vacuum,
                        **kwargs)


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

        by_plane = defaultdict(lambda: defaultdict(list))
        for elem in [self.X] if self.has_organic_A else [self.A, self.X]:
            elem_atoms = np.flatnonzero(symbols == elem)
            planes = self._surface_planes(positions[elem_atoms])
            for i, plane in zip(elem_atoms.tolist(), planes.tolist()):
                if any(plane):
                    by_plane[tuple(plane)][elem].append(i)

        sites = [BindingSite(index=i, symbol=elem, plane=plane, position=positions[i])
                 for plane, elems in by_plane.items()
                 for elem, indices in elems.items()
                 for i in indices]

        if self.has_organic_A:
            a_sites = self._A_sites()
            planes = self._surface_planes(np.array([center for center, _ in a_sites]))
            sites.extend(BindingSite(index=idx.tolist(), symbol=self.A_label, plane=tuple(plane),
                                     position=center)
                         for (center, idx), plane in zip(a_sites, planes.tolist()) if any(plane))

        return sites


    def _build_octahedra(self) -> None:
        """
        Build the octahedral network: the nearest X neighbors of each B (minimum images
        along the periodic axes), and the integer (i, j, k) lattice coordinates of each B.
        """
        syms = np.array(self.atoms.get_chemical_symbols())
        b_idx = np.where(syms == self.B)[0]
        x_idx = np.where(syms == self.X)[0]

        pos = self.atoms.get_positions()
        tree, image_of = self.box.tree(pos[x_idx])
        neigh_lists = tree.query_ball_point(pos[b_idx], float(self.a)/2 + 1e-2)

        self.octahedra = {int(b): {"X": sorted({int(x_idx[image_of[j]]) for j in neighbors}), "Ligand": []}
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
        available (bool): Whether a ligand may still bind here: False once one binds
            to it, or a multi-motif ligand lies over it or close enough to cover it.
    """
    index: int | List[int]
    symbol: str
    plane: Tuple[int, int, int]
    position: np.ndarray
    available: bool = True

    @property
    def atom_indices(self) -> List[int]:
        """All atom indices of this site (one for atomic, many for molecular)."""
        return self.index if isinstance(self.index, list) else [self.index]