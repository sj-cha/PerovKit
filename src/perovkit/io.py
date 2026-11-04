from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.vasp import write_vasp


def write_structure(
    atoms: Atoms,
    filename: str,
    fmt: str,
    box: bool = False,
    vacuum: float = 15.0,
    **kwargs,
) -> None:
    """
    Write `atoms` to `filename`, creating its directory.

    Args:
        atoms: The structure; it is left as it is.
        filename: Output path.
        fmt: "vasp", or a format of ase.io.write, which gets the formula as a comment.
        box: For "vasp", center the atoms in a periodic orthorhombic cell first, as a
            molecule or a nanocrystal needs.
        vacuum: How much wider (Å) that cell is than the atoms along each axis.
        **kwargs: Forwarded to the writer. "vasp" always sorts the atoms by element.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "vasp":
        if box:
            pos = atoms.get_positions()
            cell_diag = pos.max(axis=0) - pos.min(axis=0) + vacuum
            center = pos.mean(axis=0)

            atoms = atoms.copy()
            atoms.set_cell(np.diag(cell_diag))
            atoms.positions += cell_diag / 2 - center
            atoms.pbc = True
        write_vasp(str(path), atoms, **{**kwargs, "sort": True})
    else:
        write(str(path), atoms, **{**kwargs, "format": fmt, "comment": atoms.get_chemical_formula()})
