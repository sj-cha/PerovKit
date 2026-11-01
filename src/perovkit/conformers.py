from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .ligand import Ligand


@dataclass
class ConformerLibrary:
    """
    Conformers of one ligand with the anchor and binding direction of every motif.

    Built once per ligand type and reused for every placement of it: matching a ligand to
    a set of surface sites then reduces to rigid fits against these conformers.

    Attributes:
        coords (np.ndarray): (n_conf, n_atoms, 3) positions, in the atom order of Ligand.atoms.
        energies (np.ndarray): (n_conf,) UFF energies (kcal/mol) relative to the lowest
            conformer; all zero when the conformers were not minimized.
        anchors (np.ndarray): (n_conf, n_motifs, 3) centroid of each motif's binding atoms.
        directions (np.ndarray): (n_conf, n_motifs, 3) unit vector from each anchor into
            the ligand, which should point away from the surface; zero where undefined.
        body (np.ndarray): (n_atoms,) mask of the heavy atoms outside the binding groups,
            whose spread over the surface measures a conformer's footprint.
        mirrored (bool): Whether the second half of the conformers mirrors the first.
        stretched (np.ndarray): (n_conf,) conformers whose dangling chains were straightened
            and stood up (see build); placement does not stretch them again.
    """
    coords: np.ndarray
    energies: np.ndarray
    anchors: np.ndarray
    directions: np.ndarray
    body: np.ndarray
    mirrored: bool = False
    stretched: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.stretched is None:
            self.stretched = np.zeros(len(self.coords), dtype=bool)

    @property
    def n_conformers(self) -> int:
        return len(self.coords)

    def unique_id(self, conformer: int) -> int:
        """
        The conformer, or the one it mirrors: on a mirror-symmetric surface the two make
        the same attempt.
        """
        return conformer % (self.n_conformers // 2) if self.mirrored else conformer

    @classmethod
    def build(
        cls,
        ligand: Ligand,
        n_conformers: int = 50,
        prune_rms: float = 0.5,
        optimize: bool = True,
        mirror: Optional[bool] = None,
        random_seed: int = 42,
        n_threads: int = 0,
        n_stretched: int = 12,
    ) -> ConformerLibrary:
        """
        Embed conformers with ETKDG and, optionally, minimize them with UFF.

        The ligand's current geometry is kept as conformer 0. UFF is used because RDKit's
        implementation has no electrostatic term: MMFF in vacuum folds zwitterions into an
        intramolecular salt bridge, which would hide the extended conformers that bridge
        two surface sites.

        Args:
            ligand (Ligand): Ligand with an RDKit mol and resolved binding atoms.
            n_conformers (int): Number of ETKDG embeddings to attempt.
            prune_rms (float): RMSD (Å) below which ETKDG discards a conformer as a duplicate,
                comparing atoms by index rather than up to symmetry.
            optimize (bool): Minimize every conformer with UFF and record its energy.
            mirror (Optional[bool]): Add the mirror image of every conformer. A rotation
                cannot turn a conformer into its mirror image, and with three or more anchors
                it is often the mirror that keeps the body off the surface. None adds them
                when no atom carries a chiral tag, since mirroring would invert it.
            random_seed (int): Seed for ETKDG embedding.
            n_threads (int): Threads for embedding and minimization; 0 uses all.
            n_stretched (int): For a multi-motif ligand with dangling chains, add this many
                conformers with those chains straightened and standing up from a flat
                surface through the anchors, then relaxed with UFF (see _stretched).
                Embedded conformers are coils that mostly lie on the surface; without
                these, about half of the constellations of a lecithin shell had no rigid
                fit clear of the core and needed a refinement during site selection.
        """
        if ligand.mol is None:
            raise ValueError("ConformerLibrary needs the ligand's RDKit mol")
        if not ligand.binding_atoms:
            raise ValueError("ConformerLibrary needs a ligand with binding motifs")

        mol = Chem.Mol(ligand.mol)  # conformer 0 holds the current geometry
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.pruneRmsThresh = prune_rms
        # a sulfonate turned by 120 degrees is a duplicate to ETKDG, but not to a motif that
        # names two of its oxygens
        params.useSymmetryForPruning = False
        params.clearConfs = False
        params.numThreads = n_threads
        n_new = len(AllChem.EmbedMultipleConfs(mol, n_conformers, params))
        if n_new < n_conformers // 2:
            # large molecules mostly fail to embed unless started from random coordinates
            params.useRandomCoords = True
            AllChem.EmbedMultipleConfs(mol, n_conformers - n_new, params)

        energies = np.zeros(mol.GetNumConformers())
        if optimize and AllChem.UFFHasAllMoleculeParams(mol):
            results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=n_threads)
            energies = np.array([e for _, e in results], dtype=float)

        coords = np.array([c.GetPositions() for c in mol.GetConformers()], dtype=float)
        # one non-finite energy would otherwise poison every relative energy below
        finite = np.isfinite(energies) & np.isfinite(coords).all(axis=(1, 2))
        coords, energies = coords[finite], energies[finite]
        extra, extra_e = _stretched(ligand, mol, coords, n_stretched,
                                    uff=optimize and AllChem.UFFHasAllMoleculeParams(mol),
                                    base_energy=float(energies.min()))
        stretched = np.concatenate([np.zeros(len(coords), dtype=bool), np.ones(len(extra), dtype=bool)])
        if len(extra):
            coords = np.concatenate([coords, extra])
            energies = np.concatenate([energies, extra_e if optimize else np.zeros(len(extra))])
        energies = energies - energies.min()
        if mirror is None:
            mirror = all(a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
        if mirror:
            coords = np.concatenate([coords, coords * np.array([-1.0, 1.0, 1.0])])
            energies = np.concatenate([energies, energies])
            stretched = np.concatenate([stretched, stretched])
        anchors, directions = _anchor_geometry(mol, ligand.binding_groups, coords)

        body = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
        body[[i for g in ligand.binding_groups for i in g]] = False
        if not body.any():
            body[:] = True
        return cls(coords=coords, energies=energies, anchors=anchors, directions=directions,
                   body=body, mirrored=bool(mirror), stretched=stretched)

    def pair_distances(self) -> np.ndarray:
        """
        Distance between every pair of anchors in every conformer: (n_conf, n_motifs, n_motifs).
        """
        diff = self.anchors[:, :, None, :] - self.anchors[:, None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def distance_bounds(self) -> np.ndarray:
        """
        Smallest and largest anchor-anchor distance over the conformers: (n_motifs, n_motifs, 2).
        """
        d = self.pair_distances()
        return np.stack([d.min(axis=0), d.max(axis=0)], axis=-1)


def _surface_normal(x: np.ndarray, anchors: np.ndarray, directions: np.ndarray,
                    body: np.ndarray, n_spin: int = 72) -> np.ndarray:
    """
    The outward normal of a flat surface carrying a conformer by its anchors, chosen as
    the site fits are (constellation.fit_conformers): with two anchors (or anchors on a
    line) the normal turns about the anchor axis to point the motif directions outward and
    stand the body up; otherwise it is the normal of the anchor plane, on the body's side.
    """
    A = anchors - anchors.mean(axis=0)
    has = np.linalg.norm(directions, axis=1) > 0
    sv, vt = np.linalg.svd(A)[1:]
    if len(A) > 2 and sv[1] > 0.05 * sv[0]:
        n = vt[2]
        return n if (directions[has] @ n).mean() >= 0 else -n
    e1 = vt[0]
    e2 = np.cross(e1, [1.0, 0.0, 0.0] if abs(e1[0]) < 0.9 else [0.0, 1.0, 0.0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    t = np.linspace(0.0, 2.0 * np.pi, n_spin, endpoint=False)
    normals = np.cos(t)[:, None] * e2 + np.sin(t)[:, None] * e3          # (S, 3)
    align = (normals @ directions[has].T).mean(axis=1) if has.any() else np.zeros(n_spin)
    b = x[body] - x[body].mean(axis=0)
    cov = b.T @ b / len(b)
    lat2 = np.trace(cov) - np.einsum("si,ij,sj->s", normals, cov, normals)
    return normals[int(np.argmin(2.0 * (1.0 - align) + lat2 / 4.0))]


def _uff_polish(mol: Chem.Mol, x: np.ndarray, fixed: List[int], max_displ: float = 0.3,
                iters: int = 300) -> Tuple[np.ndarray, float]:
    """
    Relax a pose with UFF, `fixed` atoms held in place and other heavy atoms within
    `max_displ` Å of where they are: removes the close contacts torsion moves leave
    (stretched lecithin 160-300 -> ~60 kcal/mol) without changing the shape. Returns the
    coordinates and their UFF energy.
    """
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for a, p in enumerate(x):
        conf.SetAtomPosition(a, Point3D(*map(float, p)))
    ff = AllChem.UFFGetMoleculeForceField(m)
    fixed = set(fixed)
    for a in range(len(x)):
        if a in fixed:
            ff.UFFAddPositionConstraint(a, 0.0, 1.0e4)
        elif m.GetAtomWithIdx(a).GetAtomicNum() > 1:
            ff.UFFAddPositionConstraint(a, max_displ, 100.0)
    ff.Minimize(maxIts=iters)
    y = np.array(m.GetConformer().GetPositions())
    return y, AllChem.UFFGetMoleculeForceField(m).CalcEnergy()


def _stretched(ligand: Ligand, mol: Chem.Mol, coords: np.ndarray, n_pick: int, uff: bool = True,
               base_energy: float = 0.0, max_strain: float = 40.0, cutoff: float = 1.8,
               spacing: float = 3.0, hole: float = 2.5):
    """
    Copies of `n_pick` conformers, spread over the anchor-anchor distance, with their
    dangling chains stretched (refine.stretch) against a flat surface through the anchors:
    a square grid of points `spacing` Å apart, open within `hole` Å of each anchor where
    its own site would be. With `uff` each copy is then relaxed (_uff_polish), and copies
    more than `max_strain` kcal/mol above the lowest embedded conformer are dropped. Clashes
    with the surface are left to site selection, which fits the copies to real sites like
    any other conformer: the grid only stands in for a surface to lift the chains from.

    Returns:
        The copies (k, n_atoms, 3) and their UFF energies (k,), zero without `uff`.
    """
    empty = np.zeros((0,) + coords.shape[1:]), np.zeros(0)
    if n_pick <= 0 or ligand.n_motifs < 2:
        return empty
    model = ligand.torsion_model()
    if not model.tails.any():
        return empty
    from .refine import stretch          # refine is only needed for chains to stretch

    groups = ligand.binding_groups
    anchors, directions = _anchor_geometry(mol, groups, coords)
    body = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    body[[i for g in groups for i in g]] = False
    d = np.linalg.norm(anchors[:, 1] - anchors[:, 0], axis=1)
    order = np.argsort(d, kind="stable")
    picks = order[np.unique(np.linspace(0, len(order) - 1, min(n_pick, len(order))).round().astype(int))]

    fixed = [i for g in groups for i in g]
    out, energies = [], []
    for c in picks:
        x, A = coords[c], anchors[c]
        n = _surface_normal(x, A, directions[c], body)
        center = A.mean(axis=0)
        e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        radius = float(np.linalg.norm(x - center, axis=1).max()) + 2.0 * spacing
        g = np.arange(-radius, radius + spacing, spacing)
        uu, vv = np.meshgrid(g, g)
        grid = center + uu.ravel()[:, None] * e1 + vv.ravel()[:, None] * e2
        flat = A - center - np.outer((A - center) @ n, n)
        gap = np.linalg.norm((grid - center)[:, None] - flat[None], axis=-1).min(axis=1)
        tree = cKDTree(grid[gap > hole])
        res = stretch(model, x, groups, A, n, [tree], cutoff + 0.2)
        if res is None:
            continue
        y, e = _uff_polish(mol, res.coords, fixed) if uff else (res.coords, base_energy)
        if e - base_energy <= max_strain:
            out.append(y)
            energies.append(e)
    return np.array(out).reshape((-1,) + coords.shape[1:]), np.array(energies)


def _anchor_geometry(mol: Chem.Mol, groups: List[List[int]], coords: np.ndarray):
    """
    Anchor (centroid of the binding atoms) and binding direction of each motif.

    The direction points from the anchor to the heavy atoms bonded to the binding atoms,
    e.g. O,O -> S for a sulfonate or N -> C for an ammonium. When those cancel out (a
    quaternary ammonium), it points to the centroid of all other heavy atoms instead.
    """
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    n_conf = len(coords)
    anchors = np.zeros((n_conf, len(groups), 3))
    directions = np.zeros((n_conf, len(groups), 3))

    for k, group in enumerate(groups):
        group = np.asarray(group, dtype=int)
        bonded = sorted({n.GetIdx() for i in group for n in mol.GetAtomWithIdx(int(i)).GetNeighbors()
                         if heavy[n.GetIdx()]} - set(group.tolist()))
        rest = np.flatnonzero(heavy)
        rest = rest[~np.isin(rest, group)]

        anchor = coords[:, group].mean(axis=1)
        anchors[:, k] = anchor
        for targets in (bonded, rest):
            if len(targets) == 0:
                continue
            v = coords[:, targets].mean(axis=1) - anchor
            norm = np.linalg.norm(v, axis=1)
            todo = (np.linalg.norm(directions[:, k], axis=1) == 0) & (norm > 0.3)
            directions[todo, k] = v[todo] / norm[todo, None]

    return anchors, directions
