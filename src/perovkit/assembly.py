from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

from ase import Atoms
from ase.io import read

from .core import Core
from .io.metadata import read_assemblies, write_metadata
from .io.structure import write_structure
from .ligand import Ligand


@dataclass
class Assembly:
    """
    A Core with ligands on its surface: a nanocrystal, or a slab when the Core is periodic
    (Core.build_core(..., pbc=(True, True, False))). KD-tree searches follow the core's pbc:
    minimum-image distances along its periodic axes. A slab keeps its cell.
    Made by Core.place_ligands and Core.place_ligands_manually, or read back with
    from_file and from_ase.

    `atoms` is the only copy of the atoms: whatever is done to them is done there, and the
    `atoms` of `core` and of each ligand are read-only selections of it. Its per-atom array
    "ligand_id" tells whose each atom is (-1: the core); the ids are the ligands' places in
    `ligands`. After atoms are deleted from it, the next method call, or read of `octahedra`
    or `B_ijk`, brings `core`, `octahedra`, `B_ijk` and `ligands` up to date, the remaining
    ligands taking the ids 0, 1, ... in order; deleting only part of a ligand is not
    supported.
    """
    core: Core
    ligands: List[Ligand] = field(default_factory=list)
    ligand_coverage: Dict[str, float] = field(default_factory=dict)

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


    def remove_ligand(self, id: int) -> None:
        """Remove the ligand with this id; the ligands after it move up one id."""
        self._follow_deleted_atoms()
        if not any(lig.id == id for lig in self.ligands):
            raise ValueError(f"No ligand with id={id} found.")
        del self.atoms[self.atoms.arrays["ligand_id"] == id]
        self._follow_deleted_atoms()


    def apply_tilt(
        self,
        glazer: str,
        angles: Tuple[float, float, float],
        *,
        order: str = "xyz",
        move_ligands: bool = True,
    ):
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
        from .strain import apply_strain
        self._follow_deleted_atoms()
        apply_strain(
            structure=self,
            strain=strain,
            strain_ligands=strain_ligands
        )


    def check_overlaps(self, cutoff: float = 1.6, verbose: bool = True) -> float:
        """
        Closest contact (Å) between the core and a ligand or between two ligands, inf without
        ligands. With `verbose`, also lists the pairs closer than `cutoff`.
        """
        self._follow_deleted_atoms()
        log = print if verbose else (lambda *a, **k: None)
        box = self.core.box

        entities = [self.core.atoms.get_positions()] + [lig.atoms.get_positions() for lig in self.ligands]
        entity_labels = ["core"] + [f"ligand_{i}" for i in range(len(self.ligands))]

        centers = np.array([e.mean(axis=0) for e in entities])
        radii = np.array([np.linalg.norm(e - c, axis=1).max() for e, c in zip(entities, centers)])

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
                    tree_i, _ = box.tree(entities[i])
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
        Write the structure and, with `write_json`, its metadata to `filename`.json.
        `fmt` defaults to "vasp" for a slab and "xyz" for a nanocrystal; a nanocrystal
        written as vasp is boxed with `vacuum` Å around it.
        """
        if fmt is None:
            fmt = "vasp" if self.core.is_slab else "xyz"

        if filename is None:
            filename = f"{self.atoms.get_chemical_formula()}.{fmt}"

        write_structure(self.atoms, filename, fmt, box=self.core.is_nanocrystal, vacuum=vacuum, **kwargs)
        if write_json:
            self.to_json(f"{filename}.json", sort=fmt == "vasp")


    def to_json(self, filename: str, sort: bool = False) -> None:
        self._follow_deleted_atoms()
        write_metadata(self, filename, sort)


    @classmethod
    def from_file(cls, filename: str, json_filename: str) -> Assembly:
        return cls.from_ase(read(filename), json_filename)


    @classmethod
    def from_ase(cls, atoms: Atoms | List[Atoms], json_filename: str) -> Assembly | List[Assembly]:
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
        if isinstance(core.A, list):
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
        core.B_ijk = {new[b]: ijk for b, ijk in core.B_ijk.items() if b in new}
