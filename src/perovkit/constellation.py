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

from scipy.spatial import cKDTree

from .conformers import ConformerLibrary
from .core import BindingSite
from .refine import TorsionModel, refine, stretch


def rotations_about_axis(axis: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """
    Rodrigues rotation matrices about one unit axis for many angles: (n_theta, 3, 3).
    """
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    s = np.sin(thetas)[:, None, None]
    c = np.cos(thetas)[:, None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def assign_faces(sites: Sequence[BindingSite]) -> List[tuple]:
    """
    The face each site counts toward when spreading ligands over a nanocrystal.

    A site on a face counts toward it. A site on an edge or corner lies on two or three
    faces at once: along each edge (or among the corners of one kind) the sites are dealt
    to those faces in turn, so that every face gets its share instead of all edges going
    to one axis. On a slab every site keeps its plane.
    """
    faces: List[Optional[tuple]] = [None] * len(sites)
    shared: Dict[tuple, List[int]] = {}
    for i, s in enumerate(sites):
        plane = tuple(int(v) for v in s.plane)
        if sum(v != 0 for v in plane) <= 1:
            faces[i] = plane
        else:
            shared.setdefault(plane, []).append(i)
    for plane, idx in shared.items():
        options = [tuple(v if a == ax else 0 for a, v in enumerate(plane))
                   for ax in range(3) if plane[ax] != 0]
        for k, i in enumerate(sorted(idx, key=lambda i: tuple(sites[i].position))):
            faces[i] = options[k % len(options)]
    return faces


@dataclass
class Fit:
    """
    Rigid placement of one conformer onto a constellation.

    Coordinates are ``(x - center) @ rotation.T + pivot``, then turned by ``theta`` about
    ``axis`` through ``pivot`` when the fit leaves that spin free (two anchors, or anchors
    on one line). A refined or stretched fit stores its coordinates at ``theta`` in ``xyz``
    instead, and says which in ``origin``: "fit", "stretched" (a library conformer with
    its chains already stood up), "refined", or "unstuck" (chains stretched just until
    they clear the core).
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
    xyz: Optional[np.ndarray] = None
    origin: str = "fit"

    def coords(self, lib: ConformerLibrary, theta: Optional[float] = None) -> np.ndarray:
        theta = self.theta if theta is None else theta
        if self.xyz is not None:
            x = self.xyz - self.pivot
            if self.axis is not None and theta != self.theta:
                x = x @ rotations_about_axis(self.axis, np.array([theta - self.theta]))[0].T
            return x + self.pivot
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
        n_refine (int): When no constellation of a seed has a fit clear of the core, this
            many distinct conformers are refined in torsion space before the seed is given
            up (0 disables). Starting points include conformers up to 2 * max_rmsd off,
            since refinement also pulls the anchors in; the result must meet max_rmsd.
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
        n_refine: int = 8,
    ):
        self.core_tree, self.core_map = tree_builder(np.asarray(core_positions, dtype=float))
        self._n_core = len(core_positions)
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
        self.n_refine = n_refine
        self.placed: List[np.ndarray] = []
        self._lig_tree = None
        self.occupancy: Dict[tuple, float] = {}


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
        model: Optional[TorsionModel] = None,
        groups: Optional[List[List[int]]] = None,
    ) -> Tuple[List[Candidate], Counter]:
        """
        Accept up to `n_target` constellations, spread evenly over the faces of the surface.

        Seeds are the first motif's sites, grouped by face (see assign_faces: edge and
        corner sites are shared out among their faces; a slab surface is one face). The next
        seed comes from the face whose occupancy, ligands accepted over its capacity, is
        lowest; the capacity of a face is its free sites of the first motif's type divided
        by the motifs needing that type, so occupancy evens out per site, as coverage is
        defined, even where faces lost different numbers of sites. Within the face the seed
        is the one farthest from every occupied site, recomputed after each acceptance. A
        face whose seeds fail simply stays behind, and the other faces take up the shortfall.

        Args:
            lib (ConformerLibrary): Conformers of the ligand.
            pools (List[List[BindingSite]]): Candidate sites of each motif.
            n_target (int): Number of ligands wanted.
            anchor_offset (float): Offset (Å) of every anchor along its site normal.
            replace (bool): Whether the ligand replaces its site atoms (else it adsorbs).
            model, groups: Torsion model and binding groups of the ligand, to refine fits
                that clash with the core; without them such constellations are rejected.

        Returns:
            The accepted candidates and a count of the reasons seeds were rejected.
        """
        accepted: List[Candidate] = []
        failures: Counter = Counter()
        seeds = pools[0]
        if not seeds:
            return accepted, failures
        per_ligand = sum(p is seeds for p in pools)   # motifs drawing on the seed pool

        faces: Dict[tuple, List[int]] = {}
        for i, (s, f) in enumerate(zip(seeds, assign_faces(seeds))):
            if not s.passivated:
                faces.setdefault(f, []).append(i)
        capacity = {f: len(ix) / per_ligand for f, ix in faces.items()}
        placed: Counter = Counter()

        X = np.array([s.position for s in seeds], dtype=float)
        sites = {id(s): s for p in pools for s in p}.values()
        taken = np.array([s.position for s in sites if s.passivated], dtype=float).reshape(-1, 3)
        gap = np.full(len(seeds), np.inf)
        for y in taken:
            gap = np.minimum(gap, np.linalg.norm(self.mic(X - y), axis=1))
        tried = np.zeros(len(seeds), dtype=bool)

        while len(accepted) < n_target:
            open_faces = [f for f, ix in faces.items()
                          if any(not tried[i] and not seeds[i].passivated for i in ix)]
            if not open_faces:
                break
            face = min(open_faces, key=lambda f: (placed[f] / capacity[f], -capacity[f], f))
            free = [i for i in faces[face] if not tried[i] and not seeds[i].passivated]
            if np.isinf(gap[free]).all():
                i = free[self.rng.randint(0, len(free) - 1)]
            else:
                i = free[int(np.argmax(gap[free]))]
            tried[i] = True

            cand, reason = self._best_candidate(lib, pools, seeds[i], anchor_offset, replace,
                                                model, groups)
            if cand is None:
                failures[reason] += 1
                continue
            self.accept(cand, lib, replace)
            accepted.append(cand)
            placed[face] += 1
            for s in cand.sites:
                gap = np.minimum(gap, np.linalg.norm(self.mic(X - s.position), axis=1))

        self.occupancy = {f: placed[f] / capacity[f] for f in faces}
        return accepted, failures


    def fit_sites(
        self,
        lib: ConformerLibrary,
        sites: List[BindingSite],
        anchor_offset: float = 0.0,
        replace: bool = True,
        model: Optional[TorsionModel] = None,
        groups: Optional[List[List[int]]] = None,
    ) -> Tuple[Optional[Candidate], str]:
        """
        Evaluate one given constellation; returns (candidate or None, rejection reason).
        """
        seed = sites[0]
        rel = np.array([self.mic(s.position - seed.position) for s in sites])
        cand, reason, stuck = self._evaluate(lib, sites, seed.position + rel, anchor_offset, replace)
        if cand is None and stuck and model is not None:
            cand = self._rescue(stuck, model, groups)
        return cand, ("" if cand is not None else reason)


    def accept(self, cand: Candidate, lib: ConformerLibrary, replace: bool) -> None:
        for s in cand.sites:
            s.passivated = True
        if replace:
            self.displaced.update(cand.atom_indices)
        self.placed.append(cand.fits[0].coords(lib))
        self._lig_tree = self.tree_builder(np.vstack(self.placed))


    def block_covered(self, sites: Sequence[BindingSite], radius: float) -> int:
        """
        Take free sites with an accepted ligand atom within `radius` (Å) out of use: a
        ligand anchored there would start inside that ligand. Returns how many were taken.
        """
        if self._lig_tree is None:
            return 0
        tree, _ = self._lig_tree
        n = 0
        for s in sites:
            if not s.passivated and tree.query(s.position, k=1)[0] < radius:
                s.passivated = True
                n += 1
        return n


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
        model: Optional[TorsionModel],
        groups: Optional[List[List[int]]],
    ) -> Tuple[Optional[Candidate], str]:
        found: List[Tuple[float, Candidate]] = []
        reasons: Counter = Counter()
        stuck: List[tuple] = []
        for sites, rel in self._constellations(pools, seed, lib):
            cand, reason, starts = self._evaluate(lib, sites, seed.position + rel, anchor_offset, replace)
            if cand is None:
                reasons[reason] += 1
                stuck += starts
                continue
            found.append((cand.fits[0].cost, cand))
            if len(found) >= self.n_constellations:
                break
        if not found and stuck and model is not None:
            cand = self._rescue(stuck, model, groups)
            if cand is not None:
                return cand, ""
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
    ) -> Tuple[Optional[Candidate], str, Optional[tuple]]:
        """
        Fit every conformer to the constellation and keep the fits clear of the core.

        Returns (candidate or None, rejection reason, starts), where starts are refinement
        starting points for _rescue when no fit is clear: (cost, conformer id, fit, coords,
        sites, targets, excluded), from the clashing fits and the near-miss conformers.
        """
        normals = np.array([np.asarray(s.plane, dtype=float) / np.linalg.norm(s.plane)
                            for s in sites])
        targets = positions + anchor_offset * normals
        res = fit_conformers(lib, targets, normals, w_align=self.w_align,
                             w_foot=self.w_foot, foot_scale=self.foot_scale)

        near = np.flatnonzero(res["rmsd"] <= 2.0 * self.max_rmsd)
        near = near[np.argsort(res["cost"][near], kind="stable")]
        ok = near[res["rmsd"][near] <= self.max_rmsd][: self.n_fits]
        if len(near) == 0:
            return None, "no conformer fits the site distances", []

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

        def spun(c):
            base = Fit(conformer=int(c), rotation=res["rotation"][c], center=res["center"][c],
                       pivot=res["pivot"], axis=axis, theta=0.0, rmsd=float(res["rmsd"][c]),
                       alignment=float(res["alignment"][c]),
                       footprint=float(res["footprint"][c]), energy=float(lib.energies[c]),
                       cost=float(res["cost"][c]),
                       origin="stretched" if lib.stretched[c] else "fit")
            x0 = base.coords(lib) - base.pivot
            coords = np.einsum("sij,mj->smi", spins, x0) + base.pivot          # (S, m, 3)
            return base, coords, self._core_clearance(coords, excluded)

        fits: List[Fit] = []
        starts: List[tuple] = []
        for c in ok:
            base, coords, d_core = spun(c)
            feasible = d_core >= self.overlap_cutoff
            if not feasible.any():
                s = int(np.argmax(d_core))
                starts.append((base.cost, lib.unique_id(base.conformer), base, coords[s],
                               list(sites), targets, excluded))
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

        if fits:
            fits.sort(key=lambda f: f.cost)
            return Candidate(sites=list(sites), targets=targets, fits=fits), "", []

        if self.n_refine > 0:
            for c in near[res["rmsd"][near] > self.max_rmsd][: self.n_refine]:
                base, coords, d_core = spun(c)
                s = int(np.argmax(d_core))
                starts.append((base.cost, lib.unique_id(base.conformer), base, coords[s],
                               list(sites), targets, excluded))
        reason = "every fit clashes with the core" if len(ok) else "no conformer fits the site distances"
        return None, reason, starts


    def _rescue(self, starts: List[tuple], model: TorsionModel, groups: List[List[int]]) -> Optional[Candidate]:
        """
        Rescue the cheapest starting points, one per conformer up to mirroring (symmetric
        constellations repeat the same fits), up to n_refine; the first that clears the
        core wins. A start whose anchors already fit is first stretched (only its dangling
        chains move, a fraction of a refinement's cost): rigid fits mostly clash because a
        chain lies on the surface. Otherwise, or if that is not enough, it is refined.
        """
        tried = set()
        for cost, key, base, start, sites, targets, excluded in sorted(starts, key=lambda t: t[0]):
            if len(tried) >= self.n_refine:
                break
            if key in tried:
                continue
            tried.add(key)
            fit = None
            if base.rmsd <= self.max_rmsd and model.tails.any():
                fit = self._stretch_fit(base, start, model, groups, sites, targets, excluded)
            if fit is None:
                fit = self._refine_fit(base, start, model, groups, targets, excluded)
            if fit is not None:
                return Candidate(sites=sites, targets=targets, fits=[fit])
        return None


    def _surroundings(self, start: np.ndarray, pivot: np.ndarray, excluded: np.ndarray) -> list:
        """
        Trees of the core atoms (minus `excluded`) and placed ligand atoms around a pose.
        """
        reach = float(np.linalg.norm(start - pivot, axis=1).max()) + self.overlap_cutoff + 1.0
        trees = [self._local_tree(self.core_tree, self.core_map, pivot, reach, excluded)]
        if self._lig_tree is not None:
            trees.append(self._local_tree(*self._lig_tree, pivot, reach))
        return trees


    def _stretch_fit(
        self,
        base: Fit,
        start: np.ndarray,
        model: TorsionModel,
        groups: List[List[int]],
        sites: List[BindingSite],
        targets: np.ndarray,
        excluded: np.ndarray,
    ) -> Optional[Fit]:
        """
        Stretch the dangling chains of a fit that clashes with the core, only until they
        clear it; returns it as a fit with explicit coordinates ("unstuck": the stretch
        before packing still stands it up fully) if that works, else None.
        """
        normal = np.mean([np.asarray(s.plane, dtype=float) / np.linalg.norm(s.plane) for s in sites], axis=0)
        res = stretch(model, start, groups, targets, normal / np.linalg.norm(normal),
                      self._surroundings(start, base.pivot, excluded), self.overlap_cutoff + 0.2,
                      stop_at=self.overlap_cutoff + 0.1)
        if res is None or self._core_clearance(res.coords[None], excluded)[0] < self.overlap_cutoff:
            return None
        return Fit(conformer=base.conformer, rotation=base.rotation, center=base.center,
                   pivot=base.pivot, axis=base.axis, theta=0.0, rmsd=base.rmsd,
                   alignment=base.alignment, footprint=base.footprint, energy=base.energy,
                   cost=base.cost, xyz=res.coords, origin="unstuck")


    def _refine_fit(
        self,
        base: Fit,
        start: np.ndarray,
        model: TorsionModel,
        groups: List[List[int]],
        targets: np.ndarray,
        excluded: np.ndarray,
    ) -> Optional[Fit]:
        """
        Refine a fit that clashes with the core, against the core atoms and placed ligands
        around it; returns it as a fit with explicit coordinates, or None if it stays stuck.
        """
        trees = self._surroundings(start, base.pivot, excluded)
        res = refine(model, start, groups, targets, trees, contact=self.overlap_cutoff + 0.2,
                     stop_at=self.overlap_cutoff + 0.1, stop_rmsd=self.max_rmsd)
        if res.anchor_rmsd > self.max_rmsd:
            return None
        if self._core_clearance(res.coords[None], excluded)[0] < self.overlap_cutoff:
            return None
        return Fit(conformer=base.conformer, rotation=base.rotation, center=base.center,
                   pivot=base.pivot, axis=base.axis, theta=0.0, rmsd=res.anchor_rmsd,
                   alignment=base.alignment, footprint=base.footprint, energy=base.energy,
                   cost=base.cost, xyz=res.coords, origin="refined")


    @staticmethod
    def _local_tree(tree, index_map, center, radius, excluded: Optional[np.ndarray] = None):
        """
        Tree of the points of `tree` within `radius` of `center`, minus `excluded` atoms.
        """
        idx = np.asarray(tree.query_ball_point(center, r=radius), dtype=int)
        if excluded is not None and len(idx):
            idx = idx[~np.isin(index_map[idx], excluded)]
        return cKDTree(tree.data[idx]) if len(idx) else None


    def _core_clearance(self, coords: np.ndarray, excluded: np.ndarray, k: int = 8) -> np.ndarray:
        """
        Per-pose distance to the nearest core atom not in `excluded`: (S,) for (S, m, 3).

        Only contacts up to overlap_cutoff + 0.5 Å are searched, inf beyond: every use of
        the result looks at contacts below the cutoff, and bounding the search is what
        makes the scan over spins and fits affordable.
        """
        S, m, _ = coords.shape
        n = self.core_tree.n
        dist, idx = self.core_tree.query(coords.reshape(-1, 3), k=min(k, n),
                                         distance_upper_bound=self.overlap_cutoff + 0.5)
        dist = dist.reshape(S * m, -1)
        idx = idx.reshape(S * m, -1)
        valid = idx < n
        blocked = np.zeros(self._n_core + 1, dtype=bool)   # last slot: no neighbor found
        blocked[np.asarray(excluded, dtype=int)] = True
        atom = np.where(valid, self.core_map[np.minimum(idx, n - 1)], self._n_core)
        dist = np.where(valid & ~blocked[atom], dist, np.inf)
        return dist.min(axis=1).reshape(S, m).min(axis=1)


    def _ligand_clearance(self, coords: np.ndarray) -> np.ndarray:
        """
        Per-pose distance to the nearest atom of an already placed ligand: (S,), inf beyond
        overlap_cutoff, where it no longer counts as crowding.
        """
        if self._lig_tree is None:
            return np.full(len(coords), np.inf)
        tree, _ = self._lig_tree
        dist, _ = tree.query(coords.reshape(-1, 3), k=1, distance_upper_bound=self.overlap_cutoff)
        return dist.reshape(coords.shape[0], -1).min(axis=1)
