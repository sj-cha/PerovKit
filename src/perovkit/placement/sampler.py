from __future__ import annotations
from collections import Counter
import dataclasses
import itertools
import random
from typing import TYPE_CHECKING, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numba
import numpy as np
from scipy.spatial import cKDTree

from ..conformers import ConformerLibrary, embed_at, uff_energy, uff_polish
from ..core import BindingSite
from ..refine import CONTACT_MARGINS, refine, stretch
from ..torsion import TorsionModel
from ..utils.box import Box
from ..utils.rotation import rotation_about_axis
from .fitting import Candidate, Fit, fit_conformers, mean_normal, site_normals
from .grid import CoreGrid
from .sites import assign_faces

if TYPE_CHECKING:
    from rdkit import Chem
    from ..ligand import LigandSpec


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


@dataclasses.dataclass
class _Job:
    """
    A multi-motif ligand as the sampler places it: its conformer library with the templates
    learned so far (the sampler keeps it under `key`, the id of the ligand's own library),
    the torsion model that rescues fits clashing with the core (None: they are rejected), its
    binding groups and RDKit mol, the radii with which its atoms shadow the surface (zero for
    atoms that cast none: a surface site lies under one ligand at most), the offset (Å) of
    its anchors along the site normals, and whether it replaces its site atoms.
    """
    lib: ConformerLibrary
    key: int
    model: Optional[TorsionModel]
    groups: List[List[int]]
    mol: Chem.Mol
    radii: np.ndarray
    offset: float
    replace: bool


class _Start(NamedTuple):
    """A starting point for _rescue: a fit (`base`) whose coordinates are `start`."""
    cost: float
    key: object   # the conformer up to mirroring, or the embedding
    base: Fit
    start: np.ndarray
    sites: List[BindingSite]
    targets: np.ndarray
    excluded: np.ndarray


class ConstellationSampler:
    """
    Picks constellations for the multi-motif ligands of one structure, one ligand at a time.

    Accepted ligands are remembered, so later candidates avoid their sites and atoms.

    Args:
        core_positions (np.ndarray): Positions of all core atoms, before any are removed.
        box (Box): The cell of the core.
        rng (random.Random): Random generator of the placement.
        sites (Sequence[BindingSite]): Every surface site; a ligand takes the free ones under
            it out of use (accept), and a fit may not lie over one already taken.
        overlap_cutoff (float): Minimum distance (Å) a fitted ligand must keep from the core.
        max_rmsd (float): Largest anchor RMSD (Å) at which a conformer still fits.
        n_fits (int): Fits kept per constellation, and offered later as alternatives.
        n_constellations (int): Feasible constellations compared per seed.
        max_branch (int): Constellations enumerated per seed.
        n_spin (int): Spin angles scanned when the fit leaves one free.
        w_align, w_foot, foot_scale (float): Orientation terms of the cost, see fit_conformers.
        w_crowd (float): Weight of the squared shortfall (Å^2) below overlap_cutoff to
            already placed ligands.
        n_refine (int): Distinct conformers refined in torsion space before a seed whose fits
            all clash with the core is given up (0 disables); they may start up to
            2 * max_rmsd off, since refinement also pulls the anchors in.
        max_templates (int): Rescued poses kept per ligand type as templates (_learn).
        embed_constellations, embed_conformers, max_embeddings (int): Where no conformer comes
            near the sites (_embed): the most compact constellations tried per seed, the
            conformers embedded at each, and the embeddings per placement.
    """

    def __init__(
        self,
        core_positions: np.ndarray,
        box: Box,
        rng: random.Random,
        sites: Sequence[BindingSite],
        overlap_cutoff: float,
        max_rmsd: float,
        n_fits: int = 4,
        n_constellations: int = 4,
        max_branch: int = 64,
        n_spin: int = 36,
        w_align: float = 2.0,
        w_foot: float = 1.0,
        foot_scale: float = 2.0,
        w_crowd: float = 1.0,
        n_refine: int = 8,
        max_templates: int = 16,
        embed_constellations: int = 2,
        embed_conformers: int = 8,
        max_embeddings: int = 12,
    ):
        self.core_tree, self.core_map = box.tree(np.asarray(core_positions, dtype=float))
        self._n_core = len(core_positions)
        self.box = box
        self.rng = rng
        self.displaced: set = set()   # core atoms that accepted ligands replace
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
        self.max_templates = max_templates
        self.embed_constellations = embed_constellations
        self.embed_conformers = embed_conformers
        self.max_embeddings = max_embeddings
        self.placed: List[np.ndarray] = []
        self._lig_tree = None
        self._core_grid: Optional[CoreGrid] = None
        self.sites = list(sites)
        self._site_positions = np.array([s.position for s in self.sites], dtype=float).reshape(-1, 3)
        planes = np.array([s.plane for s in self.sites], dtype=float).reshape(-1, 3)
        self._site_normals = planes / np.maximum(np.linalg.norm(planes, axis=1, keepdims=True), 1e-12)
        self._site_tree = self.box.tree(self._site_positions)
        self.n_shadowed = 0
        self._n_embedded = 0
        self._libraries: Dict[int, ConformerLibrary] = {}   # with templates, by id of the original
        self.occupancy: Dict[tuple, float] = {}


    def _job(self, spec: LigandSpec, refine: bool) -> _Job:
        lig = spec.ligand
        lib = lig.conformers()
        tm = lig.torsion_model()
        # only frame atoms shadow sites: torsion-turned atoms can move out of the way
        return _Job(lib=self._libraries.get(id(lib), lib), key=id(lib), model=tm if refine else None,
                     groups=lig.binding_groups, mol=lig.mol, radii=np.where(tm.frame, tm.radii, 0.0),
                     offset=float(spec.anchor_offset), replace=not spec.adsorb)


    def sample(
        self,
        spec: LigandSpec,
        pools: List[List[BindingSite]],
        n_target: int,
        refine: bool,
    ) -> Tuple[List[Candidate], Counter]:
        """
        Accept up to `n_target` constellations, spread evenly over the faces of the surface.

        Seeds are the first motif's sites, grouped by face (assign_faces; a slab surface is
        one face). The next seed comes from the face of lowest occupancy, ligands accepted
        over its free sites of the first motif's type per ligand, and within it is the one
        farthest from every occupied site. A face whose seeds fail stays behind, and the
        other faces take up the shortfall.

        Args:
            spec (LigandSpec): The multi-motif ligand, its anchor offset and whether it adsorbs.
            pools (List[List[BindingSite]]): Candidate sites of each motif.
            n_target (int): Number of ligands wanted.
            refine (bool): Whether fits that clash with the core are refined in torsion space
                (else rejected).

        Returns:
            The accepted candidates and a count of the reasons seeds were rejected.
        """
        job = self._job(spec, refine)
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
            gap = np.minimum(gap, np.linalg.norm(self.box.mic(X - y), axis=1))
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

            cand, reason = self._best_candidate(job, pools, seeds[i])
            if cand is None:
                failures[reason] += 1
                continue
            if cand.fits[0].origin in ("unstuck", "refined"):
                cand = self._learn(job, cand)
            self.accept(cand, job)
            accepted.append(cand)
            placed[face] += 1
            for s in cand.sites:
                gap = np.minimum(gap, np.linalg.norm(self.box.mic(X - s.position), axis=1))

        self.occupancy = {f: placed[f] / capacity[f] for f in faces}
        return accepted, failures


    def fit_sites(
        self,
        spec: LigandSpec,
        sites: List[BindingSite],
        refine: bool,
    ) -> Tuple[Optional[Candidate], str]:
        """
        Evaluate one given constellation for `spec` and accept it if a fit clears the core;
        returns (candidate or None, rejection reason).
        """
        job = self._job(spec, refine)
        seed = sites[0]
        rel = np.array([self.box.mic(s.position - seed.position) for s in sites])
        cand, reason, stuck = self._evaluate(job, sites, seed.position + rel)
        if cand is None and stuck and job.model is not None:
            cand = self._rescue(job, stuck)
            if cand is not None:
                cand = self._learn(job, cand)
        if cand is None:
            return None, reason
        self.accept(cand, job)
        return cand, ""


    def accept(self, cand: Candidate, job: _Job) -> None:
        """
        Take the candidate's sites, and the free surface sites under the ligand, out of use.
        A site is under it when an atom on its outer side, projected along the site normal,
        covers it with its radius: a broad multipodal ligand lying on the surface would
        otherwise let its neighbors start inside it.
        """
        for s in cand.sites:
            s.passivated = True
        if job.replace:
            self.displaced.update(cand.atom_indices)
        x = cand.fits[0].coords(job.lib)
        self.placed.append(x)
        self._lig_tree = self.box.tree(np.vstack(self.placed))
        self.n_shadowed += self._shadow(x, job.radii)


    def _nearby_sites(self, center: np.ndarray, reach: float, taken: bool) -> np.ndarray:
        """Indices of the sites within `reach` of `center` that are taken (or free)."""
        tree, index_of = self._site_tree
        near = np.unique(index_of[np.asarray(tree.query_ball_point(center, r=reach), dtype=int)])
        return np.array([k for k in near if self.sites[k].passivated == taken], dtype=int)


    def _over(self, coords: np.ndarray, pivot: np.ndarray, radii: np.ndarray,
              sites: np.ndarray) -> np.ndarray:
        """Per pose (S, m, 3): whether it lies over any of `sites` (indices); see _over_sites."""
        out = np.zeros(len(coords), dtype=bool)
        if len(sites):
            # the sites' images nearest the pose, so that plain differences are minimum images
            P = pivot + self.box.mic(self._site_positions[sites] - pivot)
            _over_sites(np.ascontiguousarray(coords, dtype=float), np.ascontiguousarray(P),
                        np.ascontiguousarray(self._site_normals[sites]),
                        np.ascontiguousarray(radii, dtype=float), out)
        return out


    def _covers_taken(self, coords: np.ndarray, pivot: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """
        Per pose (S, m, 3): whether it lies over a site already taken, as an anchor site or
        under another ligand; a surface site lies under one ligand at most.
        """
        reach = float(np.linalg.norm(coords[0] - pivot, axis=1).max()) + float(radii.max())
        return self._over(coords, pivot, radii, self._nearby_sites(pivot, reach, taken=True))


    def _shadow(self, x: np.ndarray, radii: np.ndarray) -> int:
        """Mark the free sites under pose `x` passivated; returns how many."""
        center = x.mean(axis=0)
        reach = float(np.linalg.norm(x - center, axis=1).max()) + float(radii.max())
        free = self._nearby_sites(center, reach, taken=False)
        under = [k for k in free if self._over(x[None], center, radii, np.array([k]))[0]]
        for k in under:
            self.sites[k].passivated = True
        return len(under)


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
        library's anchor-anchor range widened by 2 * max_rmsd, nearest to typical first:
        the median distances of the templates when there are any, else of the library.
        """
        n = len(pools)
        d = np.linalg.norm(lib.anchors[:, :, None, :] - lib.anchors[:, None, :, :], axis=-1)
        bounds = np.stack([d.min(axis=0), d.max(axis=0)], axis=-1)
        # constellations shaped like the templates (poses proven at real sites) come first
        typical = np.median(d[lib.first_template:] if lib.n_templates else d, axis=0)
        tol = 2.0 * self.max_rmsd
        rel_pools = [self.box.mic(np.array([s.position for s in p], dtype=float) - seed.position)
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
        job: _Job,
        pools: List[List[BindingSite]],
        seed: BindingSite,
    ) -> Tuple[Optional[Candidate], str]:
        found: List[Tuple[float, Candidate]] = []
        reasons: Counter = Counter()
        stuck: List[_Start] = []
        tried: List[tuple] = []
        for sites, rel in self._constellations(pools, seed, job.lib):
            cand, reason, starts = self._evaluate(job, sites, seed.position + rel)
            if cand is None:
                reasons[reason] += 1
                stuck += starts
                if not starts:
                    tried.append((sites, seed.position + rel))
                continue
            found.append((cand.fits[0].cost, cand))
            if len(found) >= self.n_constellations:
                break
        if not found and stuck and job.model is not None:
            cand = self._rescue(job, stuck)
            if cand is not None:
                return cand, ""
        if not found and not stuck and tried and job.model is not None:
            cand = self._embed(job, tried)
            if cand is not None:
                return cand, ""
        if not found:
            return None, (reasons.most_common(1)[0][0] if reasons else "no partner site in reach")
        found.sort(key=lambda fc: fc[0])
        return found[0][1], ""


    def _evaluate(
        self,
        job: _Job,
        sites: List[BindingSite],
        positions: np.ndarray,
    ) -> Tuple[Optional[Candidate], str, List[_Start]]:
        """
        Fit every conformer to the constellation and keep the fits clear of the core.

        Returns (candidate or None, rejection reason, starts), where starts are refinement
        starting points for _rescue when no fit is clear, from the clashing fits and the
        near-miss conformers.
        """
        lib = job.lib
        normals = site_normals(sites)
        targets = positions + job.offset * normals
        res = fit_conformers(lib, targets, normals, w_align=self.w_align,
                             w_foot=self.w_foot, foot_scale=self.foot_scale)

        near = np.flatnonzero(res["rmsd"] <= 2.0 * self.max_rmsd)
        if len(near) == 0:
            return None, "no conformer fits the site distances", []
        near = near[np.argsort(res["cost"][near], kind="stable")]
        ok = near[res["rmsd"][near] <= self.max_rmsd][: self.n_fits]
        if lib.n_templates:
            # always try fitting templates, or clashing rigid fits get rescued again
            learned = near[(near >= lib.first_template) & (res["rmsd"][near] <= self.max_rmsd)]
            ok = np.concatenate([ok, learned[~np.isin(learned, ok)]])

        excluded = self._excluded(sites, job.replace)

        axis = res["axis"]
        thetas = (np.linspace(0.0, 2.0 * np.pi, self.n_spin, endpoint=False)
                  if axis is not None else np.zeros(1))
        spins = rotation_about_axis(axis, thetas) if axis is not None else np.eye(3)[None]
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
            coords = x0 @ spins.transpose(0, 2, 1) + base.pivot          # (S, m, 3)
            return base, coords, self._core_clearance(coords, excluded)

        def start(base, coords, d_core):
            # refinement starts from the spin farthest from the core
            return _Start(base.cost, lib.unique_id(base.conformer), base, coords[int(np.argmax(d_core))],
                          list(sites), targets, excluded)

        fits: List[Fit] = []
        starts: List[_Start] = []
        n_covering = 0
        for c in ok:
            base, coords, d_core = spun(c)
            feasible = d_core >= self.overlap_cutoff
            if feasible.any():
                free = ~self._covers_taken(coords, base.pivot, job.radii)
                if not (feasible & free).any():
                    n_covering += 1
                    continue   # no rescue: bending would not lift it off others' sites
                feasible &= free
            if not feasible.any():
                starts.append(start(base, coords, d_core))
                continue

            dirs = lib.directions[c] @ (spins @ base.rotation).transpose(0, 2, 1)
            cos = np.einsum("sni,ni->sn", dirs, normals)
            align = (cos * has_dir[c]).sum(axis=1) / max(has_dir[c].sum(), 1)
            body = coords[:, lib.body]
            perp = body - (body @ nbar)[..., None] * nbar
            lat2 = np.mean(np.sum((perp - perp.mean(axis=1, keepdims=True)) ** 2, axis=-1), axis=1)
            d_lig = np.full(len(coords), np.inf)   # spins that clash with the core score inf anyway
            d_lig[feasible] = self._ligand_clearance(coords[feasible])
            short = np.maximum(0.0, self.overlap_cutoff - np.minimum(d_core, d_lig))
            score = (self.w_align * (1.0 - align) + self.w_foot * lat2 / self.foot_scale ** 2
                     + self.w_crowd * short ** 2)
            score[~feasible] = np.inf

            s = int(np.argmin(score))
            base.theta = float(thetas[s])
            base.alignment = float(align[s])
            base.footprint = float(np.sqrt(lat2[s]))
            base.cost = float(res["fit"][c]) + float(score[s])
            if c >= lib.first_template:
                base.xyz = base.coords(lib)   # poses are rebuilt from the ligand's own library
            fits.append(base)

        if fits:
            fits.sort(key=lambda f: f.cost)
            return Candidate(sites=list(sites), targets=targets, fits=fits), "", []

        if self.n_refine > 0:
            for c in near[res["rmsd"][near] > self.max_rmsd][: self.n_refine]:
                starts.append(start(*spun(c)))
        reason = ("no conformer fits the site distances" if not len(ok) else
                  "every fit clashes with the core" if not n_covering else
                  "every fit clashes with the core or lies over another ligand's sites")
        return None, reason, starts


    def _rescue(self, job: _Job, starts: List[_Start]) -> Optional[Candidate]:
        """
        Rescue the cheapest starting points, one per conformer up to mirroring (symmetric
        constellations repeat the same fits), up to n_refine; the first that clears the
        core wins. A start whose anchors already fit is first stretched (only its dangling
        chains move, a fraction of a refinement's cost): rigid fits mostly clash because a
        chain lies on the surface. Otherwise, or if that is not enough, it is refined.
        """
        tried = set()
        for s in sorted(starts, key=lambda s: s.cost):
            if len(tried) >= self.n_refine:
                break
            if s.key in tried:
                continue
            tried.add(s.key)
            fit = None
            if s.base.rmsd <= self.max_rmsd and job.model.tails.any():
                fit = self._stretch_fit(job, s)
            if fit is None:
                fit = self._refine_fit(job, s)
            if fit is not None and self._covers_taken(fit.xyz[None], s.base.pivot, job.radii)[0]:
                fit = None
            if fit is not None:
                return Candidate(sites=s.sites, targets=s.targets, fits=[fit])
        return None


    def _embed(self, job: _Job, constellations: List[tuple]) -> Optional[Candidate]:
        """
        Fit the ligand to sites that no conformer of the library comes near (binding groups
        at the ends of long flexible arms, as in a tetrapodal cavitand): at the most compact
        constellations tried, embed fresh conformers with each binding atom and its heavy
        neighbors pinned where the best-fitting conformer, each motif moved onto its target,
        puts them, and rescue those like clashing fits.
        """
        def spread(c):
            P = c[1]
            return max(float(np.linalg.norm(P[i] - P[j]))
                       for i, j in itertools.combinations(range(len(P)), 2))

        lib, mol = job.lib, job.mol
        for sites, positions in sorted(constellations, key=spread)[:self.embed_constellations]:
            if self._n_embedded >= self.max_embeddings:
                return None
            self._n_embedded += 1
            normals = site_normals(sites)
            targets = positions + job.offset * normals
            res = fit_conformers(lib, targets, normals, w_align=self.w_align,
                                 w_foot=self.w_foot, foot_scale=self.foot_scale)
            c = int(np.argmin(res["cost"]))
            x = (lib.coords[c] - res["center"][c]) @ res["rotation"][c].T + res["pivot"]
            pins = {}
            for g, t in zip(job.groups, targets):
                shift = t - x[list(g)].mean(axis=0)
                for a in set(g) | {n.GetIdx() for i in g for n in mol.GetAtomWithIdx(int(i)).GetNeighbors()
                                   if n.GetAtomicNum() > 1}:
                    pins[a] = x[a] + shift
            ys = embed_at(mol, pins, n_conformers=self.embed_conformers, random_seed=1000 + self._n_embedded)
            excluded = self._excluded(sites, job.replace)
            starts = []
            for k, y in enumerate(ys):
                rmsd = float(np.sqrt(np.mean([np.sum((y[list(g)].mean(axis=0) - t) ** 2)
                                              for g, t in zip(job.groups, targets)])))
                base = Fit(conformer=-1, rotation=np.eye(3), center=np.zeros(3),
                           pivot=targets.mean(axis=0), axis=None, theta=0.0, rmsd=rmsd,
                           alignment=0.0, footprint=0.0, energy=0.0, cost=float(k), xyz=y,
                           origin="embedded")
                starts.append(_Start(float(k), ("embedded", self._n_embedded, k), base, y,
                                     list(sites), targets, excluded))
            cand = self._rescue(job, starts) if starts else None
            if cand is not None:
                return cand
        return None


    def _learn(self, job: _Job, cand: Candidate) -> Candidate:
        """
        Make a rescued pose a template, as a monodentate ligand is placed from one stretched
        geometry: its chains stood up against the core and relaxed with UFF (atoms near the
        core held, as UFF does not see it), it becomes the candidate's pose and joins the
        ligand's library, so that constellations of the same geometry fit it rigidly instead
        of being rescued again. If the relaxed pose, then the unrelaxed one, would touch the
        core, the rescued pose itself is the template. Returns the candidate.
        """
        lib, mol, groups = job.lib, job.mol, job.groups
        if lib.n_templates >= self.max_templates or not job.model.tails.any():
            return cand
        fit = cand.fits[0]
        x = fit.coords(lib)
        excluded = self._excluded(cand.sites, job.replace)
        core = self._local_tree(self.core_tree, self.core_map, fit.pivot, np.inf, excluded)
        res = stretch(job.model, x, groups, cand.targets, mean_normal(cand.sites), [core],
                      self.overlap_cutoff + CONTACT_MARGINS[0])
        poses = []
        if res is not None:
            held = [i for g in groups for i in g]
            held += np.flatnonzero(core.query(res.coords)[0] < self.overlap_cutoff + 0.5).tolist()
            y, e = uff_polish(mol, res.coords, held)
            poses = [(y, e), (res.coords, None)]
        for y, e in poses:
            if self._core_clearance(y[None], excluded)[0] >= self.overlap_cutoff:
                energy = (uff_energy(mol, y) if e is None else e) - lib.reference_energy
                job.lib = self._libraries[job.key] = lib.with_template(mol, groups, y, energy)
                return Candidate(sites=cand.sites, targets=cand.targets,
                                 fits=[dataclasses.replace(fit, xyz=y, origin="stretched", energy=energy)])
        energy = uff_energy(mol, x) - lib.reference_energy
        job.lib = self._libraries[job.key] = lib.with_template(mol, groups, x, energy, stretched=False)
        return cand


    def _excluded(self, sites: List[BindingSite], replace: bool) -> np.ndarray:
        """
        Core atoms a pose at `sites` may overlap: those already displaced, and its own sites'
        atoms when it replaces them.
        """
        excluded = set(self.displaced)
        if replace:
            excluded.update(i for s in sites for i in s.atom_indices)
        return np.fromiter(excluded, dtype=int, count=len(excluded))


    def _surroundings(self, start: np.ndarray, pivot: np.ndarray, excluded: np.ndarray) -> list:
        """
        Trees of the core atoms (minus `excluded`) and placed ligand atoms around a pose.
        """
        reach = float(np.linalg.norm(start - pivot, axis=1).max()) + self.overlap_cutoff + 1.0
        trees = [self._local_tree(self.core_tree, self.core_map, pivot, reach, excluded)]
        if self._lig_tree is not None:
            trees.append(self._local_tree(*self._lig_tree, pivot, reach))
        return trees


    def _stretch_fit(self, job: _Job, s: _Start) -> Optional[Fit]:
        """
        Stretch the dangling chains of a fit that clashes with the core, only until they
        clear it; returns it as a fit with explicit coordinates ("unstuck": the stretch
        before packing still stands it up fully) if that works, else None.
        """
        res = stretch(job.model, s.start, job.groups, s.targets, mean_normal(s.sites),
                      self._surroundings(s.start, s.base.pivot, s.excluded),
                      self.overlap_cutoff + CONTACT_MARGINS[0],
                      stop_at=self.overlap_cutoff + CONTACT_MARGINS[0] / 2)
        if res is None or self._core_clearance(res.coords[None], s.excluded)[0] < self.overlap_cutoff:
            return None
        return dataclasses.replace(s.base, theta=0.0, xyz=res.coords, origin="unstuck")


    def _refine_fit(self, job: _Job, s: _Start) -> Optional[Fit]:
        """
        Refine a fit that clashes with the core, against the core atoms and placed ligands
        around it; returns it as a fit with explicit coordinates, or None if it stays stuck.
        """
        trees = self._surroundings(s.start, s.base.pivot, s.excluded)
        res = refine(job.model, s.start, job.groups, s.targets, trees,
                     contact=self.overlap_cutoff + CONTACT_MARGINS[0],
                     stop_at=self.overlap_cutoff + CONTACT_MARGINS[0] / 2, stop_rmsd=self.max_rmsd)
        if res.anchor_rmsd > self.max_rmsd:
            return None
        if self._core_clearance(res.coords[None], s.excluded)[0] < self.overlap_cutoff:
            return None
        return dataclasses.replace(s.base, theta=0.0, rmsd=res.anchor_rmsd, xyz=res.coords,
                                   origin="refined")


    @staticmethod
    def _local_tree(tree, index_map, center, radius, excluded: Optional[np.ndarray] = None):
        """
        Tree of the points of `tree` within `radius` of `center`, minus `excluded` atoms.
        """
        idx = np.asarray(tree.query_ball_point(center, r=radius), dtype=int)
        if excluded is not None and len(idx):
            idx = idx[~np.isin(index_map[idx], excluded)]
        return cKDTree(tree.data[idx]) if len(idx) else None


    def _core_clearance(self, coords: np.ndarray, excluded: np.ndarray) -> np.ndarray:
        """
        Per-pose distance to the nearest core atom not in `excluded`: (S,) for (S, m, 3).

        Only contacts up to overlap_cutoff + 0.5 Å are searched, inf beyond: every use of
        the result looks at contacts below the cutoff, and bounding the search is what
        makes the scan over spins and fits affordable.
        """
        if self._core_grid is None:
            self._core_grid = CoreGrid(self.core_tree.data, self.core_map, self.overlap_cutoff + 0.5)
        blocked = np.zeros(self._n_core, dtype=bool)
        blocked[np.asarray(excluded, dtype=int)] = True
        return self._core_grid.clearance(coords, blocked)


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
