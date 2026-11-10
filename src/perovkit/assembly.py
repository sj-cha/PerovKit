from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

from ase import Atoms
from ase.io import read

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import PlacementConfig
from .core import Core
from .io.metadata import read_assemblies, write_metadata
from .io.structure import write_structure
from .ligand import Ligand
from .utils.rotation import rotation_about_axis


@dataclass
class Assembly:
    """
    A core with its ligands, a nanocrystal or a slab. Made by Core.place_ligands, or read
    with from_file.

    `atoms` holds every atom: the `atoms` of `core` and of each ligand are read-only views
    of it, though constraints set on them reach it, and its "ligand_id" array gives each
    atom's ligand (-1 for the core). Whole ligands deleted from `atoms` are followed, and
    the rest are renumbered 0, 1, ...

    Attributes:
        core (Core): The core.
        ligands (List[Ligand]): The ligands, in id order.
        ligand_coverage (Dict[str, float]): Requested coverage of each ligand name.
        atoms (Atoms): Every atom of the assembly.
        octahedra (Dict[int, dict]): As in Core, with the ids of the bound ligands.
        B_ijk (Dict[int, Tuple[int, int, int]]): As in Core.
        ligand_bonds (List[Tuple[int, int]]): Pairs of ligands bonded by polymerizing their
            reactive groups, (ligand id, ligand id): the tail carbon of the first to the head
            carbon of the second. Moving a ligand stretches its bonds.
        ligand_clusters (List[List[int]]): Ids of the ligands each chain or network joins,
            largest first; a reactive ligand that bonded to nothing is one alone.
        config (PlacementConfig, optional): The settings it was placed with.
    """
    core: Core
    ligands: List[Ligand] = field(default_factory=list)
    ligand_coverage: Dict[str, float] = field(default_factory=dict)
    # bonds between ligands: (ligand id, atom, ligand id, atom), atoms numbered within each ligand
    _bonds: List[Tuple[int, int, int, int]] = field(default_factory=list, repr=False)
    config: Optional[PlacementConfig] = None

    @property
    def octahedra(self):
        self._follow_deleted_atoms()
        return self.core.octahedra

    @property
    def B_ijk(self):
        self._follow_deleted_atoms()
        return self.core.B_ijk

    def __post_init__(self):
        self._build_atoms()

    @property
    def ligand_bonds(self) -> List[Tuple[int, int]]:
        self._follow_deleted_atoms()
        return [(l, m) for l, _, m, _ in self._bonds]

    @property
    def ligand_clusters(self) -> List[List[int]]:
        self._follow_deleted_atoms()
        reactive = np.array([lig.id for lig in self.ligands if lig.reactive_groups], dtype=int)
        row = {l: i for i, l in enumerate(reactive.tolist())}
        pairs = np.array([(row[l], row[m]) for l, _, m, _ in self._bonds], dtype=int).reshape(-1, 2)
        graph = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(reactive), len(reactive)))
        n, labels = connected_components(graph, directed=False)
        return sorted((reactive[labels == c].tolist() for c in range(n)), key=len, reverse=True)

    @property
    def bond_atoms(self) -> np.ndarray:
        """The atoms of each bond between ligands, as indices into `atoms`: (n_bonds, 2)."""
        self._follow_deleted_atoms()
        index = {lig.id: lig.indices for lig in self.ligands}
        return np.array([(index[l][a], index[m][b]) for l, a, m, b in self._bonds],
                        dtype=int).reshape(-1, 2)


    def remove_ligand(self, ligand_id: int) -> None:
        """
        Remove a ligand; the ligands after it move up one id.

        Args:
            ligand_id (int): Id of the ligand to remove.
        """
        self._follow_deleted_atoms()
        if not any(lig.id == ligand_id for lig in self.ligands):
            raise ValueError(f"No ligand with id={ligand_id} found.")
        del self.atoms[self.atoms.arrays["ligand_id"] == ligand_id]
        self._follow_deleted_atoms()


    def move_ligands(
        self,
        ligand_ids: int | Sequence[int],
        shift: Sequence[float] = (0.0, 0.0, 0.0),
        rotate: float = 0.0,
    ) -> None:
        """
        Move ligands by hand, in place: turn each about the normal of its site(s) through
        its anchors, then shift it. check_overlaps tells whether they now touch anything.

        Args:
            ligand_ids (int | Sequence[int]): Ligands to move.
            shift (Sequence[float]): Translation (Å) along x, y and z.
            rotate (float): Turn (deg) about the site normal.
        """
        self._follow_deleted_atoms()
        ids = [ligand_ids] if np.isscalar(ligand_ids) else list(dict.fromkeys(ligand_ids))
        unknown = [i for i in ids if not 0 <= i < len(self.ligands)]
        if unknown:
            raise ValueError(f"No ligands with ids {unknown}; ids run from 0 to "
                             f"{len(self.ligands) - 1}.")
        ligands = [self.ligands[int(i)] for i in ids]
        if rotate and any(not lig.planes for lig in ligands):
            raise ValueError("A ligand without site planes has no normal to turn about.")
        for lig in ligands:
            x = self.atoms.positions[lig.indices]
            if rotate:
                normal = np.mean([np.asarray(p, dtype=float) / np.linalg.norm(p) for p in lig.planes],
                                 axis=0)
                center = lig.anchor_positions.mean(axis=0)
                x = (x - center) @ rotation_about_axis(normal, math.radians(rotate)).T + center
            self.atoms.positions[lig.indices] = x + np.asarray(shift, dtype=float)


    def apply_tilt(
        self,
        glazer: str,
        angles: Tuple[float, float, float],
        *,
        order: str = "xyz",
        move_ligands: bool = True,
    ):
        """
        Tilt the octahedra by a Glazer pattern, in place.

        Args:
            glazer (str): Glazer notation, e.g. "a+b-c-".
            angles (Tuple[float, float, float]): Tilt angles (deg) about x, y and z.
            order (str): Order of the rotations, e.g. "xyz".
            move_ligands (bool): Turn each ligand with the octahedron it is bound to.
        """
        from .tilt import apply_tilt
        self._follow_deleted_atoms()
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
        """
        Strain the assembly along x, y and z, in place.

        Args:
            strain (Sequence[float]): Fractional strains (ex, ey, ez).
            strain_ligands (bool): Strain the ligands too; otherwise they move rigidly with
                their anchors.
        """
        from .strain import apply_strain
        self._follow_deleted_atoms()
        apply_strain(
            structure=self,
            strain=strain,
            strain_ligands=strain_ligands
        )


    def check_overlaps(self, cutoff: float = 1.6, verbose: bool = True) -> float:
        """
        Closest contact between the core and a ligand, or between two ligands; the atoms of
        a bond between ligands (ligand_bonds) are no contact.

        Args:
            cutoff (float): Pairs closer than this (Å) are listed when verbose.
            verbose (bool): Print the close pairs and the closest contact.

        Returns:
            float: Closest distance (Å); inf without ligands.
        """
        self._follow_deleted_atoms()
        log = print if verbose else (lambda *a, **k: None)
        box = self.core.box

        entities = [self.core.atoms.get_positions()] + [lig.atoms.get_positions() for lig in self.ligands]
        entity_labels = ["core"] + [f"ligand_{i}" for i in range(len(self.ligands))]

        centers = np.array([e.mean(axis=0) for e in entities])
        radii = np.array([np.linalg.norm(e - c, axis=1).max() for e, c in zip(entities, centers)])
        # entity pair (i < j) -> {atom of j: the atom of i it is bonded to}
        bonded: Dict[Tuple[int, int], Dict[int, int]] = {}
        for l, a, m, b in self._bonds:
            (i, x), (j, y) = sorted([(l + 1, a), (m + 1, b)])
            bonded.setdefault((i, j), {})[y] = x

        global_min = np.inf
        global_pair = None

        n = 0

        for i in range(len(entities)):
            # Atoms come no closer than their bounding spheres, so most pairs need no query
            gaps = np.linalg.norm(box.mic(centers[i + 1:] - centers[i]), axis=1) - radii[i] - radii[i + 1:]
            tree_i = None
            for j, gap in enumerate(gaps, start=i + 1):
                if gap > max(global_min, cutoff) + 1e-6:
                    continue

                if tree_i is None:
                    tree_i, image_of = box.tree(entities[i])
                if (i, j) in bonded:
                    # the nearest atom but the bond partner
                    near, which = tree_i.query(entities[j], k=2)
                    dists_j = near[:, 0].copy()
                    for y, x in bonded[(i, j)].items():
                        if image_of[which[y, 0]] == x:
                            dists_j[y] = near[y, 1]
                else:
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

        pair = (f" between {entity_labels[global_pair[0]]} and {entity_labels[global_pair[1]]}"
                if global_pair else "")
        log(f"[Log] global min inter-entity distance = {global_min:.3f} Å{pair} (total overlaps: {n})")

        return global_min


    def to(
        self, 
        fmt: Optional[str] = None, 
        filename: Optional[str] = None, 
        write_json: bool = True,
        vacuum: float = 20.0,
        **kwargs
    ) -> None:      
        """
        Write the assembly to a file, and its metadata to `filename`.json.

        Args:
            fmt (str, optional): File format; "vasp" for a slab and "xyz" for a
                nanocrystal by default.
            filename (str, optional): Output path; named after the formula by default.
            write_json (bool): Also write the metadata, which from_file needs.
            vacuum (float): Vacuum (Å) around a nanocrystal written as VASP.
            **kwargs: Passed to the ASE writer.
        """
        if fmt is None:
            fmt = "vasp" if self.core.is_slab else "xyz"

        if filename is None:
            filename = f"{self.atoms.get_chemical_formula()}.{fmt}"

        write_structure(self.atoms, filename, fmt, center_in_box=self.core.is_nanocrystal,
                        vacuum=vacuum, **kwargs)
        if write_json:
            self.to_json(f"{filename}.json", sort=fmt == "vasp")


    def to_json(self, filename: str, sort: bool = False) -> None:
        self._follow_deleted_atoms()
        write_metadata(self, filename, sort)


    @classmethod
    def from_file(cls, filename: str, json_filename: str) -> Assembly:
        """
        Read an assembly written by `to`.

        Args:
            filename (str): Structure file.
            json_filename (str): Its metadata file.

        Returns:
            Assembly: The assembly.
        """
        return cls.from_ase(read(filename), json_filename)


    @classmethod
    def from_ase(cls, atoms: Atoms | List[Atoms], json_filename: str) -> Assembly | List[Assembly]:
        """
        Assemblies from ASE Atoms of an assembly written by `to`, e.g. the frames of an MD
        trajectory.

        Args:
            atoms (Atoms | List[Atoms]): One structure, or a list of frames.
            json_filename (str): Metadata file of the assembly.

        Returns:
            Assembly | List[Assembly]: One assembly, or one per frame.
        """
        return read_assemblies(cls, atoms, json_filename)


    def _build_atoms(self) -> None:
        """
        Put the atoms of the core, with what they carry (arrays, cell, pbc, constraints), and
        of the ligands into `atoms`, which holds them from then on. "core_id" numbers the
        core atoms, for _follow_deleted_atoms.
        """
        core_atoms = self.core.atoms
        n_core = len(core_atoms)

        self.atoms = core_atoms.copy()
        if self.ligands:
            self.atoms.extend(Atoms(numbers=np.concatenate([lig.atoms.numbers for lig in self.ligands]),
                                    positions=np.vstack([lig.atoms.positions for lig in self.ligands])))
        if "a_site_id" in self.atoms.arrays:
            self.atoms.arrays["a_site_id"][n_core:] = -1

        n_each = [n_core] + [len(lig.atoms) for lig in self.ligands]
        self.atoms.set_array("ligand_id", np.repeat([-1] + [lig.id for lig in self.ligands], n_each))
        self.atoms.set_array("core_id", np.concatenate([np.arange(n_core), np.full(sum(n_each[1:]), -1)]))
        self._n_atoms = len(self.atoms)

        self.core._attach(self, "ligand_id", -1)
        for lig in self.ligands:
            lig._attach(self, "ligand_id", lig.id)


    def _follow_deleted_atoms(self) -> None:
        """
        After atoms were deleted from `atoms`, renumber what holds core indices, drop the
        cations and ligands that are gone and give the others the ids 0, 1, ... in order. The
        indices are mapped rather than rebuilt from positions, which a tilt or strain may have
        taken beyond what finds the octahedra. `octahedra` is replaced rather than edited:
        frames loaded together by from_ase share one.
        """
        if len(self.atoms) == self._n_atoms:
            return
        self._n_atoms = len(self.atoms)

        core_id = self.atoms.arrays["core_id"]
        kept = core_id[core_id >= 0]
        new = {old: i for i, old in enumerate(kept.tolist())}
        core_id[core_id >= 0] = np.arange(len(kept))

        core = self.core
        if core.has_organic_A:
            core.A = [lig for lig in core.A if len(lig.indices)]
        self.ligands = [lig for lig in self.ligands if len(lig.indices)]

        ligand_id = self.atoms.arrays["ligand_id"]
        old_ids = ligand_id.copy()
        new_id = {}
        for i, lig in enumerate(self.ligands):
            ligand_id[old_ids == lig.id] = i
            new_id[lig.id] = i
            lig.id = i
            lig._attach(self, "ligand_id", i)

        core.octahedra = {new[b]: {"X": [new[x] for x in octa["X"] if x in new],
                                   "Ligand": [new_id[l] for l in octa["Ligand"] if l in new_id]}
                          for b, octa in core.octahedra.items() if b in new}
        self._bonds = [(new_id[l], a, new_id[m], b) for l, a, m, b in self._bonds
                       if l in new_id and m in new_id]
        core.B_ijk = {new[b]: ijk for b, ijk in core.B_ijk.items() if b in new}
