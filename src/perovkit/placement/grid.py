import math

import numba
import numpy as np
from scipy.spatial import cKDTree


@numba.njit(cache=True)
def _grid_min_distance(x, n_poses, lo, h, shape, table, points, image_of, blocked, bound, out):
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
                if blocked[image_of[a]]:
                    continue
                dx = x[p, 0] - points[a, 0]
                dy = x[p, 1] - points[a, 1]
                dz = x[p, 2] - points[a, 2]
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d < bound and d < best:
                    best = d
        out[s] = best


@numba.njit(cache=True)
def _build_cells(points, lo, h, shape):
    """The first point of each cell and the next point in the cell of each point."""
    head = np.full(shape[0] * shape[1] * shape[2], -1, dtype=np.int64)
    following = np.empty(points.shape[0], dtype=np.int64)
    for p in range(points.shape[0]):
        c0 = int((points[p, 0] - lo[0]) / h)
        c1 = int((points[p, 1] - lo[1]) / h)
        c2 = int((points[p, 2] - lo[2]) / h)
        c = (c0 * shape[1] + c1) * shape[2] + c2
        following[p] = head[c]
        head[c] = p
    return head, following


@numba.njit(cache=True)
def _closest_squared(x, y, z, closest, cells):
    """The smaller of `closest` and the squared distance of (x, y, z) to the points of `cells`."""
    points, lo, h, shape, head, following = cells
    c0, c1, c2 = int(np.floor((x - lo[0]) / h)), int(np.floor((y - lo[1]) / h)), int(np.floor((z - lo[2]) / h))
    for i in range(max(c0 - 1, 0), min(c0 + 2, shape[0])):
        for j in range(max(c1 - 1, 0), min(c1 + 2, shape[1])):
            for k in range(max(c2 - 1, 0), min(c2 + 2, shape[2])):
                a = head[(i * shape[1] + j) * shape[2] + k]
                while a >= 0:
                    dx, dy, dz = x - points[a, 0], y - points[a, 1], z - points[a, 2]
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < closest:
                        closest = d2
                    a = following[a]
    return closest


@numba.njit(cache=True)
def _best_spin(x0, center, axis, thetas, reference, follow, best, core, neighbors, bound):
    """
    Among spins of x0 about `axis` through `center`, the first that keeps the closest contact
    with `core` and `neighbors` farthest, if beyond `best`: (spin, contact), a contact being
    `bound` at most. A spin stops at the first atom that rules it out. The spins are
    `reference` + `thetas`, the reference following the best spin found if `follow`.
    """
    ux, uy, uz = axis[0], axis[1], axis[2]
    rel = x0 - center
    best2, spin = best * best if best > 0.0 else -1.0, reference
    for s in range(thetas.shape[0]):
        theta = (reference + thetas[s]) % (2.0 * math.pi)
        c, si = math.cos(theta), math.sin(theta)
        t = 1.0 - c
        r00, r01, r02 = c + t * ux * ux, t * ux * uy - si * uz, t * ux * uz + si * uy
        r10, r11, r12 = t * ux * uy + si * uz, c + t * uy * uy, t * uy * uz - si * ux
        r20, r21, r22 = t * ux * uz - si * uy, t * uy * uz + si * ux, c + t * uz * uz
        closest = bound * bound
        for a in range(rel.shape[0]):
            vx, vy, vz = rel[a, 0], rel[a, 1], rel[a, 2]
            x = r00 * vx + r01 * vy + r02 * vz + center[0]
            y = r10 * vx + r11 * vy + r12 * vz + center[1]
            z = r20 * vx + r21 * vy + r22 * vz + center[2]
            closest = _closest_squared(x, y, z, closest, core)
            closest = _closest_squared(x, y, z, closest, neighbors)
            if closest <= best2:
                break
        if closest > best2:
            best2, spin = closest, theta
            if follow:
                reference = theta
    return spin, math.sqrt(best2) if best2 > 0.0 else -math.inf


@numba.njit(cache=True)
def _closest(x, core, neighbors, bound):
    closest = bound * bound
    for a in range(x.shape[0]):
        closest = _closest_squared(x[a, 0], x[a, 1], x[a, 2], closest, core)
        closest = _closest_squared(x[a, 0], x[a, 1], x[a, 2], closest, neighbors)
    return math.sqrt(closest)


def cells(points: np.ndarray, h: float) -> tuple:
    """
    `points` sorted into cells `h` Å wide, as the compiled searches take them: all points
    within `h` of a position lie in the 27 cells around it.
    """
    points = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
    lo = points.min(axis=0) if len(points) else np.zeros(3)
    span = points.max(axis=0) - lo if len(points) else np.zeros(3)
    shape = np.floor(span / h).astype(np.int64) + 1
    return (points, lo, float(h), shape, *_build_cells(points, lo, float(h), shape))


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
