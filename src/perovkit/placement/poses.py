from __future__ import annotations

import numpy as np

from ..ligand import Ligand
from ..utils.rotation import rotation_about_axis, rotation_from_u_to_v, rotations_about_axis
from .fitting import Candidate


def place_about_normal(
    coords_loc: np.ndarray,
    plane: np.ndarray,
    site_pos: np.ndarray,
    anchor_offset: float,
    theta: float,
) -> np.ndarray:
    """
    Turn an oriented single-motif ligand (body along +z) onto a site normal, spin it by
    `theta` about that normal and put its anchor on the site, `anchor_offset` above it.
    """
    R_align = rotation_from_u_to_v(np.array([0.0, 0.0, 1.0]), plane)
    R_rot = rotation_about_axis(plane, theta)
    anchor_pos = site_pos + plane / np.linalg.norm(plane) * float(anchor_offset)
    return (coords_loc @ R_align.T) @ R_rot.T + anchor_pos


class SingleMotifPose:
    """
    Single-motif ligand on one site, free to spin about the site normal.
    """
    random_start = True
    spins = True
    n_variants = 1
    variant = 0

    def __init__(self, ligand: Ligand, plane: np.ndarray, site_pos: np.ndarray):
        self.ligand = ligand
        self.plane = plane
        self.center = site_pos

    def coords(self, theta: float, variant: int = 0) -> np.ndarray:
        return place_about_normal(
            self.ligand.atoms.get_positions(),
            self.plane,
            self.center,
            getattr(self.ligand, "_anchor_offset", 0.0),
            theta,
        )


class MultiMotifPose:
    """
    Multi-motif ligand fitted to a constellation. Variants are coordinates of the
    candidate's alternative conformer fits, best first, followed by any refined ones;
    a stretched variant goes first. `origins` says where each came from ("fit",
    "stretched", "refined", "unstuck", "embedded"); theta is the spin about the anchor axis
    relative to each.
    """
    random_start = False

    def __init__(self, ligand: Ligand, candidate: Candidate):
        self.ligand = ligand
        self.candidate = candidate
        lib = ligand.conformers()
        self.center = candidate.targets.mean(axis=0)
        self.axis = candidate.fits[0].axis
        self.spins = self.axis is not None
        self.variants = [f.coords(lib) for f in candidate.fits]
        self.origins = [f.origin for f in candidate.fits]
        self.variant = 0
        self.reach = 0.0
        for x in self.variants:
            self._extend_reach(x)

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    def _extend_reach(self, x: np.ndarray) -> None:
        # spinning about an axis through the pivot keeps every atom's distance to it
        self.reach = max(self.reach, float(np.linalg.norm(x - self.center, axis=1).max()))

    def coords(self, theta: float, variant: int = 0) -> np.ndarray:
        x = self.variants[variant]
        if not self.spins or theta == 0.0:
            return x.copy()
        R = rotations_about_axis(self.axis, np.array([theta]))[0]
        return (x - self.center) @ R.T + self.center

    def add_variant(self, coords: np.ndarray, origin: str = "refined", first: bool = False) -> int:
        k = 0 if first else len(self.variants)
        self.variants.insert(k, np.array(coords, dtype=float))
        self.origins.insert(k, origin)
        if first:
            self.variant += 1
        self._extend_reach(self.variants[k])
        return k
