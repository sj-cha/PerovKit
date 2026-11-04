from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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
    """
    core: Core
    ligands: List[Ligand] = field(default_factory=list)
    ligand_coverage: Dict[str, float] = field(default_factory=dict)

    @property
    def octahedra(self):
        return self.core.octahedra

    @property
    def B_ijk(self):
        return self.core.B_ijk

    def __post_init__(self):
        self._link_atoms()


    def remove_ligand(self, id: int, re_index: bool = True) -> None:
        idx = next(
            (i for i, lig in enumerate(self.ligands) if lig.id == id),
            None,
        )
        if idx is None:
            raise ValueError(f"No ligand with id={id} found.")

        self.ligands.pop(idx)

        # Re-index remaining ligands
        if re_index:
            for i, lig in enumerate(self.ligands):
                lig.id = i

            # Remove ligand references from octahedra and remap ids. Entries are rebuilt rather
            # than edited in place: frames loaded together by from_ase share one octahedra dict.
            if self.octahedra is not None:
                self.core.octahedra = {
                    b_abs: {**octa, "Ligand": [lid - 1 if lid > id else lid
                                               for lid in octa["Ligand"] if lid != id]}
                    for b_abs, octa in self.octahedra.items()
                }

        self._link_atoms()


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


    def check_overlaps(self, cutoff: float = 1.6, verbose: bool = True) -> float:
        """
        Closest contact (Å) between the core and a ligand or between two ligands. With
        `verbose`, also lists the pairs closer than `cutoff`.
        """
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

        log(
            f"[Log] global min inter-entity distance = {global_min:.3f} Å "
            f"between {entity_labels[global_pair[0]]} and {entity_labels[global_pair[1]]}"
            f" (total overlaps: {n})"
        )

        return global_min


    def to(
        self, 
        fmt: Optional[str] = None, 
        filename: Optional[str] = None, 
        write_json: bool = True,
        vacuum: float = 15.0,
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
        write_metadata(self, filename, sort)


    @classmethod
    def from_file(cls, filename: str, json_filename: str) -> Assembly:
        return cls.from_ase(read(filename), json_filename)


    @classmethod
    def from_ase(cls, atoms: Atoms | List[Atoms], json_filename: str) -> Assembly | List[Assembly]:
        return read_assemblies(cls, atoms, json_filename)


    def _link_atoms(self) -> None:
        """
        Build the whole structure in `atoms`, with the positions of the core and the ligands
        as views into its buffer and each ligand's `indices` into it.
        """
        core_atoms = self.core.atoms
        all_numbers = [core_atoms.numbers]
        all_positions = [core_atoms.positions]

        for lig in self.ligands:
            all_numbers.append(lig.atoms.numbers)
            all_positions.append(lig.atoms.positions)

        self.atoms = Atoms(
            numbers=np.concatenate(all_numbers),
            positions=np.vstack(all_positions),
            pbc=core_atoms.pbc,
            cell=core_atoms.get_cell(),
        )

        shared = self.atoms.arrays["positions"]
        cursor = len(core_atoms)
        core_atoms.arrays["positions"] = shared[:cursor]

        for lig in self.ligands:
            n = len(lig.atoms)
            lig.indices = np.arange(cursor, cursor + n, dtype=int)
            lig.atoms.arrays["positions"] = shared[cursor:cursor + n]
            cursor += n
