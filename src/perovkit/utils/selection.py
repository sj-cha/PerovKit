import copy
from typing import Optional, Tuple

from ase import Atoms
import numpy as np


class SelectedAtoms(Atoms):
    """
    Atoms selected from a structure: its arrays, cell and pbc are read-only, as writing to
    them would not reach the structure, while constraints set on it go to the structure, in
    its numbering, in place of those it held on these atoms. Constraints that do not hold
    their atoms in `index` (FixBondLengths, Hookean) must be set on the structure. A copy
    or slice of it stands on its own.
    """
    _whole: Optional[Atoms] = None
    _where: Optional[np.ndarray] = None

    def set_constraint(self, constraint=None) -> None:
        if self._whole is None:
            super().set_constraint(constraint)
            return
        given = [] if constraint is None else list(constraint) if isinstance(constraint, (list, tuple)) else [constraint]
        for c in given:
            if not hasattr(c, "index"):
                raise TypeError(f"{type(c).__name__} cannot be set on a part: set it on the whole structure")
        kept = []
        for c in self._whole.constraints:
            if hasattr(c, "index"):
                c = copy.deepcopy(c)
                c.index = c.index[~np.isin(c.index, self._where)]
                if not len(c.index):
                    continue
            kept.append(c)
        for c in given:
            c = copy.deepcopy(c)
            c.index = self._where[c.index]
            kept.append(c)
        self._whole.set_constraint(kept)
        super().set_constraint(given)

    constraints = property(Atoms.constraints.fget, lambda self, c: self.set_constraint(c),
                           lambda self: self.set_constraint())


def select(atoms: Atoms, name: str, value: int) -> SelectedAtoms:
    """
    The atoms of `atoms` whose per-atom array `name` equals `value`.
    """
    mask = atoms.arrays[name] == value
    part = atoms[mask]
    for array in (*part.arrays.values(), part.pbc, part.cell.array):
        array.flags.writeable = False
    part.__class__ = SelectedAtoms
    part._whole, part._where = atoms, np.flatnonzero(mask)
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
