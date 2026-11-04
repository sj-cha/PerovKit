from __future__ import annotations

from typing import Sequence

import numpy as np

from ..assembly import Assembly
from ..core import Core


def apply_strain(
    structure: Core | Assembly,
    strain: Sequence[float],          # (ex, ey, ez)
    strain_ligands: bool = True,
):
    """
    Stretch `structure` by the fractions (ex, ey, ez) along x, y and z: a slab from the
    origin, its cell with it, a nanocrystal from the center of its core. Organic A-site
    cations move rigidly with their centers, and so do the ligands with their anchors
    unless `strain_ligands`.
    """
    strain = np.asarray(strain, dtype=float)
    if strain.shape != (3,):
        raise ValueError("Strain must be length-3: (ex, ey, ez)")
    F = np.diag(1.0 + strain)   # deformation gradient, no shear

    core = structure if isinstance(structure, Core) else structure.core
    n_core = len(core.atoms)
    pos0 = np.asarray(structure.atoms.get_positions(), dtype=float)
    if core.is_slab:
        affine = lambda p: p @ F.T
    else:
        center = np.mean(pos0[:n_core], axis=0)
        affine = lambda p: (p - center) @ F.T + center

    if strain_ligands:
        pos_new = affine(pos0)
    else:
        pos_new = pos0.copy()
        pos_new[:n_core] = affine(pos0[:n_core])
        for lig in [] if structure is core else structure.ligands:
            anchor0 = lig.anchor_pos
            if anchor0 is not None:
                pos_new[lig.indices] += affine(anchor0) - anchor0

    for cation in core.A if core.has_organic_A else []:
        g = cation.indices
        c0 = pos0[g].mean(axis=0)
        pos_new[g] = pos0[g] + (affine(c0) - c0)

    structure.atoms.positions[:] = pos_new

    if core.is_slab:
        cell = structure.atoms.get_cell().copy()
        for i in range(3):
            cell[i] *= (1 + float(strain[i]))
        structure.atoms.set_cell(cell, scale_atoms=False)
