from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


class Box:
    """An orthorhombic cell, periodic along the axes marked in `pbc`; with none, plain space."""

    def __init__(self, lengths: np.ndarray, pbc: np.ndarray):
        self.lengths = np.asarray(lengths, dtype=float)
        self.axes = np.flatnonzero(pbc)
        x, y, z = [(-1, 0, 1) if periodic else (0,) for periodic in pbc]
        self.shifts = np.array([(i, j, k) for i in x for j in y for k in z], dtype=float) * self.lengths

    def mic(self, delta: np.ndarray) -> np.ndarray:
        """Minimum images of difference vectors."""
        if not len(self.axes):
            return delta
        d = np.array(delta, dtype=float)
        for ax in self.axes:
            d[..., ax] -= self.lengths[ax] * np.round(d[..., ax] / self.lengths[ax])
        return d

    def images(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        `points` and their images one cell away along the periodic axes, with the point each
        comes from.
        """
        pts = np.asarray(points, dtype=float)
        return (pts + self.shifts[:, None]).reshape(-1, 3), np.tile(np.arange(len(pts)), len(self.shifts))

    def interior(self, points: np.ndarray, depth: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corners (lo, hi) of the extent of `points` without its outer `depth` Å, along the
        axes where it has a surface: unbounded along the periodic ones.
        """
        x = np.asarray(points, dtype=float)
        lo, hi = x.min(axis=0) + depth, x.max(axis=0) - depth
        lo[self.axes], hi[self.axes] = -np.inf, np.inf
        return lo, hi

    def tree(self, points: np.ndarray) -> Tuple[cKDTree, np.ndarray]:
        """KD-tree of the images of `points`, with the point each comes from."""
        x, image_of = self.images(points)
        # most are searched a few times and rebuilt: quicker to build, no slower to search
        return cKDTree(x, balanced_tree=False, compact_nodes=False), image_of
