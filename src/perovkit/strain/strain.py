from __future__ import annotations

from typing import Sequence

import numpy as np

from perovkit import Core, NanoCrystal, Slab


def apply_strain(
    structure: Core | NanoCrystal | Slab,
    strain: Sequence[float],          # (ex, ey, ez)
    strain_ligands: bool = True,
):
    strain = np.asarray(strain, dtype=float)
    if strain.shape != (3,):
        raise ValueError("Strain must be length-3: (ex, ey, ez)")

    # Deformation gradient (diagonal, no shear)
    F = np.eye(3, dtype=float)
    F[0, 0] += strain[0]
    F[1, 1] += strain[1]
    F[2, 2] += strain[2]

    is_periodic = isinstance(structure, Slab) or (
        isinstance(structure, Core) and structure.is_slab
    )

    pos0 = np.asarray(structure.atoms.get_positions(), dtype=float)

    if isinstance(structure, Core):
        if is_periodic:
            pos_new = pos0 @ F.T
        else:
            center = np.mean(pos0, axis=0)
            pos_new = (pos0 - center) @ F.T + center

        structure.atoms.positions[:] = pos_new

    else:
        # NanoCrystal or Slab
        n_core = len(structure.core.atoms)

        if is_periodic:
            if strain_ligands:
                pos_new = pos0 @ F.T
            else:
                pos_new = pos0.copy()
                pos_new[:n_core] = pos0[:n_core] @ F.T

                for lig in structure.ligands:
                    anchor0 = getattr(lig, "anchor_pos", None)
                    if anchor0 is None:
                        continue
                    anchor0 = np.asarray(anchor0, dtype=float).reshape(3,)
                    anchor_new = F @ anchor0
                    lig.anchor_pos = anchor_new
                    pos_new[lig.indices] += anchor_new - anchor0
        else:
            center = np.mean(pos0[:n_core], axis=0)

            if strain_ligands:
                pos_new = (pos0 - center) @ F.T + center
            else:
                pos_new = pos0.copy()
                pos_new[:n_core] = (pos0[:n_core] - center) @ F.T + center

                for lig in structure.ligands:
                    anchor0 = getattr(lig, "anchor_pos", None)
                    if anchor0 is None:
                        continue
                    anchor0 = np.asarray(anchor0, dtype=float).reshape(3,)
                    anchor_new = F @ (anchor0 - center) + center
                    lig.anchor_pos = anchor_new
                    pos_new[lig.indices] += anchor_new - anchor0

        # Single write to the shared buffer updates core and all ligands
        structure.atoms.positions[:] = pos_new

    # Update cell for periodic structures
    if is_periodic:
        if isinstance(structure, Core):
            cell = structure.atoms.get_cell().copy()
        else:
            cell = structure.core.atoms.get_cell().copy()

        for i in range(3):
            cell[i] *= (1 + float(strain[i]))

        if isinstance(structure, Core):
            structure.atoms.set_cell(cell, scale_atoms=False)
        else:
            structure.core.atoms.set_cell(cell, scale_atoms=False)
            structure.atoms.set_cell(cell, scale_atoms=False)
