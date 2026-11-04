from typing import Optional, Tuple

from ase import Atoms
import numpy as np


def select(atoms: Atoms, name: str, value: int) -> Atoms:
    """
    A copy of the atoms whose per-atom array `name` equals `value`, its arrays, cell and pbc
    read-only: writing to them would not reach `atoms`.
    """
    part = atoms[atoms.arrays[name] == value]
    for array in (*part.arrays.values(), part.pbc, part.cell.array):
        array.flags.writeable = False
    return part


class PartAtoms:
    """
    Descriptor for the `atoms` of a Part: its own Atoms, or once it is attached, a selection
    of its atoms in its parent.
    """

    def __get__(self, part, owner=None):
        if part is None:
            raise AttributeError  # tells dataclass the field has no default
        if part._parent is None:
            return part.__dict__.get("_own_atoms")
        return select(part._parent.atoms, *part._member)

    def __set__(self, part, atoms: Atoms) -> None:
        part.__dict__["_own_atoms"] = atoms


class Part:
    """
    Something with atoms of its own that can become a part of a larger structure, which
    then holds them: the `parent` whose per-atom array `name` equals `value` on them.
    """
    _parent = None
    _member: Optional[Tuple[str, int]] = None

    def _attach(self, parent, name: str, value: int) -> None:
        self._parent, self._member = parent, (name, value)
        self.__dict__.pop("_own_atoms", None)

    def _detach(self, atoms: Atoms) -> None:
        self._parent, self._member = None, None
        self.__dict__["_own_atoms"] = atoms

    @property
    def indices(self) -> Optional[np.ndarray]:
        """Indices of its atoms in the parent; None without one."""
        if self._parent is None:
            return None
        name, value = self._member
        return np.flatnonzero(self._parent.atoms.arrays[name] == value)
