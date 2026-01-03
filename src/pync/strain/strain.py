from __future__ import annotations

from typing import Sequence

import numpy as np

from pync import Core, NanoCrystal


def apply_strain(
    structure: Core | NanoCrystal,
    strain: Sequence[float],          # (ex, ey, ez)
    move_ligands: bool = True,
):
    strain = np.asarray(strain, dtype=float)
    if strain.shape != (3,):
        raise ValueError("Strain must be length-3: (ex, ey, ez)")

    atoms = structure.atoms
    pos0 = np.asarray(atoms.get_positions(), dtype=float)
    pos_new = pos0.copy()

    # Deformation gradient - no shear at the moment
    F = np.eye(3, dtype=float)
    F[0, 0] += strain[0]
    F[1, 1] += strain[1]
    F[2, 2] += strain[2]

    # Apply strain 
    pos_new = pos0[:len(atoms)] @ F.T

    # Optional move_ligand
    if isinstance(structure, NanoCrystal) and move_ligands:
        for lig in structure.ligands:
            anchor0 = getattr(lig, "anchor_pos", None)
            if anchor0 is None:
                continue

            anchor0 = np.asarray(anchor0, dtype=float).reshape(3,)
            anchor_new = F @ anchor0
            delta = anchor_new - anchor0

            lig.anchor_pos = anchor_new

            global_indices = np.asarray(lig.indices, dtype=int)
            pos_new[global_indices] += delta

    if isinstance(structure, Core):
        structure.atoms.positions[:] = pos_new[: len(structure.atoms)]
        return

    n_core = len(structure.core.atoms)
    structure.core.atoms.positions[:] = pos_new[:n_core]

    if move_ligands:
        for lig in structure.ligands:
            lig.atoms.positions[:] = pos_new[lig.indices]