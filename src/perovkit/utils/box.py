from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


class Box:
    """
    An orthorhombic cell, periodic along the axes marked in `pbc`. With no periodic axis its
    minimum images, wrapped points and trees are those of plain space.
    """

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

    def wrap(self, points: np.ndarray) -> np.ndarray:
        """Points brought into the cell along the periodic axes."""
        if not len(self.axes):
            return points
        x = np.array(points, dtype=float)
        for ax in self.axes:
            x[..., ax] -= self.lengths[ax] * np.floor(x[..., ax] / self.lengths[ax])
        return x

    def tree(self, points: np.ndarray) -> Tuple[cKDTree, np.ndarray]:
        """
        KD-tree of `points` and their images one cell away along the periodic axes, with the
        point each comes from.
        """
        pts = np.asarray(points, dtype=float)
        tree = cKDTree((pts + self.shifts[:, None]).reshape(-1, 3))
        return tree, np.tile(np.arange(len(pts)), len(self.shifts))

    def neighbor_map(self, centers: np.ndarray, radii: np.ndarray, cutoff: float) -> Dict[int, List[int]]:
        """For every sphere, those that come within `cutoff` of it."""
        if not len(self.axes):
            return build_neighbor_map(centers, radii, cutoff)

        tree, idx_map = self.tree(centers)
        rmax = float(np.max(radii))

        neighbor_map: Dict[int, List[int]] = {}
        for i in range(len(centers)):
            hits = tree.query_ball_point(centers[i], r=float(radii[i] + rmax + cutoff))
            cand = set(int(idx_map[h]) for h in hits)
            cand.discard(i)

            # exact filter with MIC on centers
            neighbor_map[i] = [j for j in cand
                               if np.linalg.norm(self.mic(centers[i] - centers[j])) < radii[i] + radii[j] + cutoff]

        return neighbor_map


def build_neighbor_map(
    centers: np.ndarray,
    radii: np.ndarray,
    cutoff: float,
) -> Dict[int, List[int]]:
    
    n = len(centers)
    neighbor_map: Dict[int, set[int]] = {i: set() for i in range(n)}

    if n == 0:
        return {}

    max_r = float(radii.max())
    max_pair_radius = 2.0 * max_r + cutoff

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=max_pair_radius)

    for i, j in pairs:
        center_dist = np.linalg.norm(centers[j] - centers[i])
        if center_dist <= radii[i] + radii[j] + cutoff:
            neighbor_map[i].add(j)
            neighbor_map[j].add(i)

    return {i: sorted(neighbor_map[i]) for i in range(n)}
