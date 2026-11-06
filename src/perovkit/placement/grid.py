import math

import numpy as np
from scipy.spatial import cKDTree

from ..kernels import _grid_min_distance


class CoreGrid:
    """
    Cells `h` Å wide over the core, each listing the core atoms (periodic images included)
    within `bound` of some point of the cell, so that the nearest core atom within `bound`
    of a point is among a handful of candidates.
    """

    def __init__(self, points: np.ndarray, image_of: np.ndarray, bound: float, h: float = 1.0):
        self.points = np.ascontiguousarray(points, dtype=float)
        self.image_of = np.asarray(image_of, dtype=np.int64)
        self.bound, self.h = float(bound), float(h)
        self.lo = self.points.min(axis=0) - bound
        self.shape = (np.ceil((self.points.max(axis=0) + bound - self.lo) / h).astype(np.int64) + 1)
        centers = self.lo + (np.indices(self.shape).reshape(3, -1).T + 0.5) * h
        lists = cKDTree(self.points).query_ball_point(centers, r=bound + h * math.sqrt(3.0) / 2.0)
        lens = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
        self.table = np.full((len(lists), max(1, int(lens.max()))), -1, dtype=np.int64)
        rows = np.repeat(np.arange(len(lists)), lens)
        cols = np.arange(len(rows)) - np.repeat(np.cumsum(lens) - lens, lens)
        self.table[rows, cols] = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if l])

    def min_distance(self, coords: np.ndarray, blocked: np.ndarray) -> np.ndarray:
        """
        Per-pose distance to the nearest core atom whose original index (image_of) is not
        `blocked`: (S,) for (S, m, 3), inf where none is within `bound`.
        """
        out = np.empty(coords.shape[0])
        _grid_min_distance(np.ascontiguousarray(coords, dtype=float).reshape(-1, 3), coords.shape[0],
                           self.lo, self.h, self.shape, self.table, self.points, self.image_of,
                           blocked, self.bound, out)
        return out
