"""
Site selection for ligands with several binding motifs.

A constellation is one surface site per motif. Candidates grow from a seed site of the
first motif, keeping partner sites whose distances fall inside the anchor-anchor range of
the ligand's conformer library; every conformer is then fitted rigidly (Kabsch) to the
constellation, and the best fits are checked against the core and the ligands placed so far.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .conformers import ConformerLibrary
from .core import BindingSite


def rotations_about_axis(axis: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """
    Rodrigues rotation matrices about one unit axis for many angles: (n_theta, 3, 3).
    """
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    s = np.sin(thetas)[:, None, None]
    c = np.cos(thetas)[:, None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


@dataclass
class Fit:
    """
    Rigid placement of one conformer onto a constellation.

    Coordinates are ``(x - center) @ rotation.T + pivot``, then turned by ``theta`` about
    ``axis`` through ``pivot`` when the fit leaves that spin free (two anchors, or anchors
    on one line).
    """
    conformer: int
    rotation: np.ndarray
    center: np.ndarray
    pivot: np.ndarray
    axis: Optional[np.ndarray]
    theta: float
    rmsd: float
    alignment: float
    footprint: float
    energy: float
    cost: float

    def coords(self, lib: ConformerLibrary, theta: Optional[float] = None) -> np.ndarray:
        theta = self.theta if theta is None else theta
        x = (lib.coords[self.conformer] - self.center) @ self.rotation.T
        if self.axis is not None and theta != 0.0:
            x = x @ rotations_about_axis(self.axis, np.array([theta]))[0].T
        return x + self.pivot


@dataclass
class Candidate:
    """
    A constellation accepted for one ligand, with its feasible fits, best first.
    """
    sites: List[BindingSite]
    targets: np.ndarray
    fits: List[Fit]

    @property
    def atom_indices(self) -> List[int]:
        return [i for s in self.sites for i in s.atom_indices]


def fit_conformers(
    lib: ConformerLibrary,
    targets: np.ndarray,
    normals: np.ndarray,
    sigma_rmsd: float = 0.25,
    energy_scale: float = 5.0,
    w_align: float = 2.0,
    w_foot: float = 1.0,
    foot_scale: float = 2.0,
    n_spin: int = 72,
) -> Dict[str, np.ndarray]:
    """
    Kabsch-fit every conformer's anchors to `targets` at once.

    The cost of a conformer is ``fit + orient``, with
    ``fit = (rmsd / sigma_rmsd)^2 + energy / energy_scale`` and
    ``orient = w_align * (1 - alignment) + w_foot * (footprint / foot_scale)^2``.
    Alignment is the mean cosine between motif directions and site normals; footprint is
    the spread (Å, RMS) of the body atoms across the mean normal, small when the ligand
    stands up rather than lying on the surface. When the fit leaves a spin free, the
    spin minimizing ``orient`` is taken.

    Returns:
        Dict of per-conformer arrays: rotation (C, 3, 3), center (C, 3), rmsd, alignment,
        footprint, fit, cost (C,), plus the pivot (3,) and spin axis (3,) or None.
    """
    P = lib.anchors
    center = P.mean(axis=1)
    pivot = targets.mean(axis=0)
    Pc = P - center[:, None]
    Tc = targets - pivot

    H = np.einsum("cni,nj->cij", Pc, Tc)
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ np.transpose(U, (0, 2, 1))))
    D = np.tile(np.eye(3), (len(P), 1, 1))
    D[:, 2, 2] = np.where(d == 0, 1.0, d)
    R = V @ D @ np.transpose(U, (0, 2, 1))

    mapped = np.einsum("cij,cnj->cni", R, Pc)
    rmsd = np.sqrt(np.mean(np.sum((mapped - Tc) ** 2, axis=-1), axis=1))

    sv = np.linalg.svd(Tc, compute_uv=False)
    axis = None
    if len(targets) == 2 or sv[1] < 0.05 * sv[0]:
        axis = Tc[-1] - Tc[0] if len(targets) == 2 else np.linalg.svd(Tc)[2][0]
        axis = axis / np.linalg.norm(axis)

    has_dir = np.linalg.norm(lib.directions, axis=-1) > 0          # (C, n)
    n_dir = np.maximum(has_dir.sum(axis=1), 1)
    dirs = np.einsum("cij,cnj->cni", R, lib.directions)

    # lateral spread across the mean normal n: trace(cov) - n.cov.n of the body atoms
    nbar = normals.mean(axis=0)
    nbar = nbar / np.linalg.norm(nbar)
    body = np.einsum("cij,cmj->cmi", R, lib.coords[:, lib.body] - center[:, None])
    body -= body.mean(axis=1, keepdims=True)
    cov = np.einsum("cmi,cmj->cij", body, body) / body.shape[1]
    spread = np.trace(cov, axis1=1, axis2=2)

    if axis is not None:
        thetas = np.linspace(0.0, 2.0 * np.pi, n_spin, endpoint=False)
        spins = rotations_about_axis(axis, thetas)                    # (S, 3, 3)
        cos = np.einsum("sij,cnj,nj->csn", spins, dirs, normals)      # (C, S, n)
        align = cos.sum(axis=-1) / n_dir[:, None]
        u = np.einsum("sji,j->si", spins, nbar)                       # spin^T n
        lat2 = spread[:, None] - np.einsum("si,cij,sj->cs", u, cov, u)
        orient = w_align * (1.0 - align) + w_foot * lat2 / foot_scale ** 2
        best = np.argmin(orient, axis=1)
        idx = np.arange(len(P))
        R = spins[best] @ R
        alignment, lat2, orient = align[idx, best], lat2[idx, best], orient[idx, best]
    else:
        alignment = np.einsum("cnj,nj->cn", dirs, normals).sum(axis=-1) / n_dir
        lat2 = spread - np.einsum("i,cij,j->c", nbar, cov, nbar)
        orient = w_align * (1.0 - alignment) + w_foot * lat2 / foot_scale ** 2

    fit = (rmsd / sigma_rmsd) ** 2 + lib.energies / energy_scale
    return dict(rotation=R, center=center, pivot=pivot, axis=axis, rmsd=rmsd,
                alignment=alignment, footprint=np.sqrt(np.maximum(lat2, 0.0)),
                fit=fit, cost=fit + orient)


class ConstellationSampler:
    """
    Picks constellations for the multi-motif ligands of one structure, one ligand at a time.

    Accepted ligands are remembered, so later candidates avoid their sites and atoms.

    Args:
        core_positions (np.ndarray): Positions of all core atoms, before any are removed.
        tree_builder (Callable): Points -> (cKDTree, index map), with periodic images for slabs.
        mic (Callable): Minimum-image convention for difference vectors (identity if not periodic).
        rng (random.Random): Random generator of the placement.
        displaced (Sequence[int]): Core atoms already removed by earlier placements.
        overlap_cutoff (float): Minimum distance (Å) a fitted ligand must keep from the core.
        max_rmsd (float): Largest anchor RMSD (Å) at which a conformer still fits.
        n_fits (int): Fits kept per constellation, and offered later as alternatives.
        n_constellations (int): Feasible constellations compared per seed.
        max_branch (int): Constellations enumerated per seed.
        n_spin (int): Spin angles scanned when the fit leaves one free.
        w_align, w_foot, foot_scale (float): Orientation terms of the cost, see fit_conformers.
        w_crowd (float): Weight of the squared shortfall (Å^2) below overlap_cutoff to
            already placed ligands.
    """

    def __init__(
        self,
        core_positions: np.ndarray,
        tree_builder: Callable,
        mic: Callable[[np.ndarray], np.ndarray],
        rng: random.Random,
        displaced: Sequence[int] = (),
        overlap_cutoff: float = 1.8,
        max_rmsd: float = 0.5,
        n_fits: int = 4,
        n_constellations: int = 4,
        max_branch: int = 64,
        n_spin: int = 36,
        w_align: float = 2.0,
        w_foot: float = 1.0,
        foot_scale: float = 2.0,
        w_crowd: float = 1.0,
    ):
        self.core_tree, self.core_map = tree_builder(np.asarray(core_positions, dtype=float))
        self.tree_builder = tree_builder
        self.mic = mic
        self.rng = rng
        self.displaced = set(int(i) for i in displaced)
        self.overlap_cutoff = overlap_cutoff
        self.max_rmsd = max_rmsd
        self.n_fits = n_fits
        self.n_constellations = n_constellations
        self.max_branch = max_branch
        self.n_spin = n_spin
        self.w_align = w_align
        self.w_foot = w_foot
        self.foot_scale = foot_scale
        self.w_crowd = w_crowd
        self.placed: List[np.ndarray] = []
        self._lig_tree = None


    @staticmethod
    def capacity(motif_sites: Sequence[str], pools: Dict[str, List[BindingSite]]) -> int:
        """
        Most ligands the free sites can host: each site type serves as many ligands as it
        has free sites divided by the motifs that need it.
        """
        need = Counter(motif_sites)
        return min(sum(not s.passivated for s in pools[t]) // k for t, k in need.items())


    def sample(
        self,
        lib: ConformerLibrary,
        pools: List[List[BindingSite]],
        n_target: int,
        anchor_offset: float = 0.0,
        replace: bool = True,
    ) -> Tuple[List[Candidate], Counter]:
        """
        Accept up to `n_target` constellations, seeding them in farthest-point order over
        the first motif's pool so that they spread over the surface.

        Args:
            lib (ConformerLibrary): Conformers of the ligand.
            pools (List[List[BindingSite]]): Candidate sites of each motif.
            n_target (int): Number of ligands wanted.
            anchor_offset (float): Offset (Å) of every anchor along its site normal.
            replace (bool): Whether the ligand replaces its site atoms (else it adsorbs).

        Returns:
            The accepted candidates and a count of the reasons seeds were rejected.
        """
        accepted: List[Candidate] = []
        failures: Counter = Counter()
        for i in self._fps_order(pools[0]):
            if len(accepted) >= n_target:
                break
            if pools[0][i].passivated:
                continue
            cand, reason = self._best_candidate(lib, pools, pools[0][i], anchor_offset, replace)
            if cand is None:
                failures[reason] += 1
                continue
            self.accept(cand, lib, replace)
            accepted.append(cand)
        return accepted, failures


    def fit_sites(
        self,
        lib: ConformerLibrary,
        sites: List[BindingSite],
        anchor_offset: float = 0.0,
        replace: bool = True,
    ) -> Tuple[Optional[Candidate], str]:
        """
        Evaluate one given constellation; returns (candidate or None, rejection reason).
        """
        seed = sites[0]
        rel = np.array([self.mic(s.position - seed.position) for s in sites])
        return self._evaluate(lib, sites, seed.position + rel, anchor_offset, replace)


    def accept(self, cand: Candidate, lib: ConformerLibrary, replace: bool) -> None:
        for s in cand.sites:
            s.passivated = True
        if replace:
            self.displaced.update(cand.atom_indices)
        self.placed.append(cand.fits[0].coords(lib))
        self._lig_tree = self.tree_builder(np.vstack(self.placed))


    def _fps_order(self, pool: List[BindingSite]) -> List[int]:
        """
        Farthest-point ordering of a pool under the minimum-image convention.
        """
        n = len(pool)
        if n == 0:
            return []
        X = np.array([s.position for s in pool], dtype=float)
        order = [self.rng.randint(0, n - 1)]
        d = np.linalg.norm(self.mic(X - X[order[0]]), axis=1)
        d[order[0]] = -1.0
        while len(order) < n:
            nxt = int(np.argmax(d))
            order.append(nxt)
            d = np.minimum(d, np.linalg.norm(self.mic(X - X[nxt]), axis=1))
            d[order] = -1.0
        return order


    def _constellations(
        self,
        pools: List[List[BindingSite]],
        seed: BindingSite,
        lib: ConformerLibrary,
    ) -> Iterator[Tuple[List[BindingSite], np.ndarray]]:
        """
        Constellations starting at `seed` whose site distances all fall within the
        library's anchor-anchor range widened by 2 * max_rmsd, nearest to typical first.
        """
        n = len(pools)
        bounds = lib.distance_bounds()
        typical = np.median(lib.pair_distances(), axis=0)
        tol = 2.0 * self.max_rmsd
        rel_pools = [self.mic(np.array([s.position for s in p], dtype=float) - seed.position)
                     if p else np.zeros((0, 3)) for p in pools]

        def extend(chosen: List[BindingSite], rel: List[np.ndarray]):
            k = len(chosen)
            if k == n:
                yield list(chosen), np.array(rel)
                return
            R = rel_pools[k]
            ok = np.array([not s.passivated and all(s is not c for c in chosen) for s in pools[k]],
                          dtype=bool)
            dev = np.zeros(len(R))
            for j in range(k):
                d = np.linalg.norm(R - rel[j], axis=1)
                ok &= (d >= bounds[j, k, 0] - tol) & (d <= bounds[j, k, 1] + tol)
                dev += np.abs(d - typical[j, k])
            for i in np.flatnonzero(ok)[np.argsort(dev[ok], kind="stable")]:
                yield from extend(chosen + [pools[k][i]], rel + [R[i]])

        count = 0
        for item in extend([seed], [np.zeros(3)]):
            yield item
            count += 1
            if count >= self.max_branch:
                return


    def _best_candidate(
        self,
        lib: ConformerLibrary,
        pools: List[List[BindingSite]],
        seed: BindingSite,
        anchor_offset: float,
        replace: bool,
    ) -> Tuple[Optional[Candidate], str]:
        found: List[Tuple[float, Candidate]] = []
        reasons: Counter = Counter()
        for sites, rel in self._constellations(pools, seed, lib):
            cand, reason = self._evaluate(lib, sites, seed.position + rel, anchor_offset, replace)
            if cand is None:
                reasons[reason] += 1
                continue
            found.append((cand.fits[0].cost, cand))
            if len(found) >= self.n_constellations:
                break
        if not found:
            return None, (reasons.most_common(1)[0][0] if reasons else "no partner site in reach")
        found.sort(key=lambda fc: fc[0])
        return found[0][1], ""


    def _evaluate(
        self,
        lib: ConformerLibrary,
        sites: List[BindingSite],
        positions: np.ndarray,
        anchor_offset: float,
        replace: bool,
    ) -> Tuple[Optional[Candidate], str]:
        """
        Fit every conformer to the constellation and keep the fits clear of the core.
        """
        normals = np.array([np.asarray(s.plane, dtype=float) / np.linalg.norm(s.plane)
                            for s in sites])
        targets = positions + anchor_offset * normals
        res = fit_conformers(lib, targets, normals, w_align=self.w_align,
                             w_foot=self.w_foot, foot_scale=self.foot_scale)

        ok = np.flatnonzero(res["rmsd"] <= self.max_rmsd)
        if len(ok) == 0:
            return None, "no conformer fits the site distances"
        ok = ok[np.argsort(res["cost"][ok], kind="stable")][: self.n_fits]

        excluded = set(self.displaced)
        if replace:
            excluded.update(i for s in sites for i in s.atom_indices)
        excluded = np.fromiter(excluded, dtype=int, count=len(excluded))

        axis = res["axis"]
        thetas = (np.linspace(0.0, 2.0 * np.pi, self.n_spin, endpoint=False)
                  if axis is not None else np.zeros(1))
        spins = rotations_about_axis(axis, thetas) if axis is not None else np.eye(3)[None]
        has_dir = np.linalg.norm(lib.directions, axis=-1) > 0
        nbar = normals.mean(axis=0)
        nbar = nbar / np.linalg.norm(nbar)

        fits: List[Fit] = []
        for c in ok:
            base = Fit(conformer=int(c), rotation=res["rotation"][c], center=res["center"][c],
                       pivot=res["pivot"], axis=axis, theta=0.0, rmsd=float(res["rmsd"][c]),
                       alignment=float(res["alignment"][c]),
                       footprint=float(res["footprint"][c]), energy=float(lib.energies[c]),
                       cost=float(res["cost"][c]))
            x0 = base.coords(lib) - base.pivot
            coords = np.einsum("sij,mj->smi", spins, x0) + base.pivot          # (S, m, 3)

            d_core = self._core_clearance(coords, excluded)
            feasible = d_core >= self.overlap_cutoff
            if not feasible.any():
                continue

            dirs = np.einsum("sij,nj->sni", spins @ base.rotation, lib.directions[c])
            cos = np.einsum("sni,ni->sn", dirs, normals)
            align = (cos * has_dir[c]).sum(axis=1) / max(has_dir[c].sum(), 1)
            body = coords[:, lib.body]
            perp = body - np.einsum("smi,i->sm", body, nbar)[..., None] * nbar
            lat2 = np.mean(np.sum((perp - perp.mean(axis=1, keepdims=True)) ** 2, axis=-1), axis=1)
            d_lig = self._ligand_clearance(coords)
            short = np.maximum(0.0, self.overlap_cutoff - np.minimum(d_core, d_lig))
            score = (self.w_align * (1.0 - align) + self.w_foot * lat2 / self.foot_scale ** 2
                     + self.w_crowd * short ** 2)
            score[~feasible] = np.inf

            s = int(np.argmin(score))
            base.theta = float(thetas[s])
            base.alignment = float(align[s])
            base.footprint = float(np.sqrt(lat2[s]))
            base.cost = float(res["fit"][c]) + float(score[s])
            fits.append(base)

        if not fits:
            return None, "every fit clashes with the core"
        fits.sort(key=lambda f: f.cost)
        return Candidate(sites=list(sites), targets=targets, fits=fits), ""


    def _core_clearance(self, coords: np.ndarray, excluded: np.ndarray, k: int = 8) -> np.ndarray:
        """
        Per-pose distance to the nearest core atom not in `excluded`: (S,) for (S, m, 3).
        """
        S, m, _ = coords.shape
        n = self.core_tree.n
        dist, idx = self.core_tree.query(coords.reshape(-1, 3), k=min(k, n))
        dist = dist.reshape(S * m, -1)
        idx = idx.reshape(S * m, -1)
        valid = idx < n
        atom = np.where(valid, self.core_map[np.minimum(idx, n - 1)], -1)
        dist = np.where(valid & ~np.isin(atom, excluded), dist, np.inf)
        return dist.min(axis=1).reshape(S, m).min(axis=1)


    def _ligand_clearance(self, coords: np.ndarray) -> np.ndarray:
        """
        Per-pose distance to the nearest atom of an already placed ligand: (S,).
        """
        if self._lig_tree is None:
            return np.full(len(coords), np.inf)
        tree, _ = self._lig_tree
        dist, _ = tree.query(coords.reshape(-1, 3), k=1)
        return dist.reshape(coords.shape[0], -1).min(axis=1)
