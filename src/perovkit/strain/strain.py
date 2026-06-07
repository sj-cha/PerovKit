from __future__ import annotations

from typing import Sequence

import numpy as np

from perovkit import Core, NanoCrystal, Slab


def _a_cation_groups(core) -> list:
    """Atom-index groups for organic A-site cation molecules (empty for inorganic A)."""
    if isinstance(core.A, list):
        return [np.asarray(lig.indices, dtype=int) for lig in core.A]
    return []


def _rigidify_groups(pos0, pos_new, groups, affine):
    """
    Override each atom group with a rigid translation: move the group's center by
    the affine map and keep its internal geometry fixed (no internal deformation).
    """
    for g in groups:
        c0 = pos0[g].mean(axis=0)
        pos_new[g] = pos0[g] + (affine(c0) - c0)


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

    core = structure if isinstance(structure, Core) else structure.core
    a_cation_groups = _a_cation_groups(core)

    if isinstance(structure, Core):
        if is_periodic:
            affine = lambda p: p @ F.T
        else:
            center = np.mean(pos0, axis=0)
            affine = lambda p: (p - center) @ F.T + center

        pos_new = affine(pos0)
        # Organic A-site cations are rigid molecules: translate, don't deform.
        _rigidify_groups(pos0, pos_new, a_cation_groups, affine)

        structure.atoms.positions[:] = pos_new

    else:
        # NanoCrystal or Slab
        n_core = len(structure.core.atoms)

        if is_periodic:
            affine = lambda p: p @ F.T
        else:
            center = np.mean(pos0[:n_core], axis=0)
            affine = lambda p: (p - center) @ F.T + center

        if strain_ligands:
            pos_new = affine(pos0)
        else:
            pos_new = pos0.copy()
            pos_new[:n_core] = affine(pos0[:n_core])

            for lig in structure.ligands:
                anchor0 = getattr(lig, "anchor_pos", None)
                if anchor0 is None:
                    continue
                anchor0 = np.asarray(anchor0, dtype=float).reshape(3,)
                pos_new[lig.indices] += affine(anchor0) - anchor0

        # Organic A-site cations live in the core block: keep them rigid too.
        _rigidify_groups(pos0, pos_new, a_cation_groups, affine)

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
