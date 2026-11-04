import math

import numba
import numpy as np
from scipy.spatial import cKDTree


@numba.njit(cache=True)
def _grid_clearance(x, n_poses, lo, h, shape, table, points, labels, blocked, bound, out):
    m = x.shape[0] // n_poses
    for s in range(n_poses):
        best = np.inf
        for p in range(s * m, (s + 1) * m):
            c0 = int(np.floor((x[p, 0] - lo[0]) / h))
            c1 = int(np.floor((x[p, 1] - lo[1]) / h))
            c2 = int(np.floor((x[p, 2] - lo[2]) / h))
            if c0 < 0 or c1 < 0 or c2 < 0 or c0 >= shape[0] or c1 >= shape[1] or c2 >= shape[2]:
                continue
            row = (c0 * shape[1] + c1) * shape[2] + c2
            for k in range(table.shape[1]):
                a = table[row, k]
                if a < 0:
                    break
                if blocked[labels[a]]:
                    continue
                dx = x[p, 0] - points[a, 0]
                dy = x[p, 1] - points[a, 1]
                dz = x[p, 2] - points[a, 2]
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d < bound and d < best:
                    best = d
        out[s] = best


@numba.njit(cache=True)
def _over_sites(x, sites, normals, radii, out):
    """
    out[s]: whether pose s of x (S, m, 3) lies over any of `sites` (K, 3): an atom on the
    outer side of the site's plane (unit `normals`) within its radius of the site laterally.
    """
    for s in range(x.shape[0]):
        hit = False
        for k in range(sites.shape[0]):
            for a in range(x.shape[1]):
                dx = x[s, a, 0] - sites[k, 0]
                dy = x[s, a, 1] - sites[k, 1]
                dz = x[s, a, 2] - sites[k, 2]
                h = dx * normals[k, 0] + dy * normals[k, 1] + dz * normals[k, 2]
                if h <= 0.0:
                    continue
                lx = dx - h * normals[k, 0]
                ly = dy - h * normals[k, 1]
                lz = dz - h * normals[k, 2]
                if lx * lx + ly * ly + lz * lz < radii[a] * radii[a]:
                    hit = True
                    break
            if hit:
                break
        out[s] = hit


class CoreGrid:
    """
    Cells `h` Å wide over the core, each listing the core atoms (periodic images included)
    within `bound` of some point of the cell, so that the nearest core atom within `bound`
    of a point is among a handful of candidates.
    """

    def __init__(self, points: np.ndarray, labels: np.ndarray, bound: float, h: float = 1.0):
        self.points = np.ascontiguousarray(points, dtype=float)
        self.labels = np.asarray(labels, dtype=np.int64)
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

    def clearance(self, coords: np.ndarray, blocked: np.ndarray) -> np.ndarray:
        """
        Per-pose distance to the nearest core atom whose label is not `blocked`: (S,) for
        (S, m, 3), inf where none is within `bound`.
        """
        out = np.empty(coords.shape[0])
        _grid_clearance(np.ascontiguousarray(coords, dtype=float).reshape(-1, 3), coords.shape[0],
                        self.lo, self.h, self.shape, self.table, self.points, self.labels,
                        blocked, self.bound, out)
        return out
