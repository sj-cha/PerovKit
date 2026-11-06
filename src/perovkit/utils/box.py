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

    def tree(self, points: np.ndarray) -> Tuple[cKDTree, np.ndarray]:
        """
        KD-tree of `points` and their images one cell away along the periodic axes, with the
        point each comes from.
        """
        pts = np.asarray(points, dtype=float)
        # most are searched a few times and rebuilt: quicker to build, no slower to search
        tree = cKDTree((pts + self.shifts[:, None]).reshape(-1, 3),
                       balanced_tree=False, compact_nodes=False)
        return tree, np.tile(np.arange(len(pts)), len(self.shifts))
