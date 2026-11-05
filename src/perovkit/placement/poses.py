from __future__ import annotations
from typing import List, Optional

import numpy as np

from ..ligand import Ligand
from ..utils.rotation import rotation_about_axis, rotation_from_u_to_v
from .fitting import Candidate, mean_normal


class Pose:
    """
    A ligand on its site(s): variants of its coordinates, best first (`sources` says where
    each came from), each spinning about `axis` through `center`; `reach` bounds its atoms'
    distance to `center`. `targets` are the anchor targets, `up` the direction the tails
    are stretched along; a pose with a `spin_axis` only turns about it in refinement.
    """
    random_start = False
    spin_axis: Optional[np.ndarray] = None

    def __init__(self, ligand: Ligand, center: np.ndarray, axis: Optional[np.ndarray],
                 variants: List[np.ndarray], sources: List[str], targets: np.ndarray,
                 up: np.ndarray):
        self.ligand = ligand
        self.center = center
        self.axis = axis
        self.can_spin = axis is not None
        self.targets = targets
        self.up = up
        self.variants = variants
        self.sources = sources
        self.variant = 0
        self.reach = 0.0
        for x in self.variants:
            self._extend_reach(x)

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    @property
    def can_bend(self) -> bool:
        return True

    def _extend_reach(self, x: np.ndarray) -> None:
        # spinning about an axis through the pivot keeps every atom's distance to it
        self.reach = max(self.reach, float(np.linalg.norm(x - self.center, axis=1).max()))

    def coords(self, theta: float, variant: int = 0) -> np.ndarray:
        x = self.variants[variant]
        if not self.can_spin or theta == 0.0:
            return x.copy()
        R = rotation_about_axis(self.axis, theta)
        return (x - self.center) @ R.T + self.center

    def add_variant(self, coords: np.ndarray, source: str = "refined", first: bool = False) -> int:
        k = 0 if first else len(self.variants)
        self.variants.insert(k, np.array(coords, dtype=float))
        self.sources.insert(k, source)
        if first:
            self.variant += 1
        self._extend_reach(self.variants[k])
        return k


class SingleMotifPose(Pose):
    """
    Single-motif ligand on one site, its anchor on the site (or its motif's `anchor_offset`
    above it), spinning about the site normal; in refinement it keeps its anchor and tilt.
    """
    random_start = True

    def __init__(self, ligand: Ligand, normal: np.ndarray, site_pos: np.ndarray):
        n = normal / np.linalg.norm(normal)
        anchor = site_pos + n * ligand.binding_motifs[0].anchor_offset
        # the oriented ligand (body along +z) turned onto the normal
        R = rotation_from_u_to_v(np.array([0.0, 0.0, 1.0]), normal)
        super().__init__(ligand, site_pos, n, [ligand.atoms.get_positions() @ R.T + anchor], ["site"],
                         anchor[None], n)
        self.spin_axis = n

    @property
    def can_bend(self) -> bool:
        # without torsions only the spin is free, which turn_pose already scans whole
        return self.ligand.torsion_model().n_torsions > 0


class MultiMotifPose(Pose):
    """
    Multi-motif ligand fitted to a constellation: its variants start as the candidate's
    alternative conformer fits, best first, and it spins about the axis of its anchors
    when it has one.
    """

    def __init__(self, ligand: Ligand, candidate: Candidate):
        lib = ligand.conformers()
        super().__init__(ligand, candidate.targets.mean(axis=0), candidate.fits[0].axis,
                         [f.coords(lib) for f in candidate.fits],
                         [f.source for f in candidate.fits],
                         candidate.targets, mean_normal(candidate.sites))
