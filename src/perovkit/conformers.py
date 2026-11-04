from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

if TYPE_CHECKING:
    from .ligand import Ligand


@dataclass
class ConformerLibrary:
    """
    Conformers of one ligand with the anchor and binding direction of every motif, built
    once per ligand type: matching the ligand to surface sites is then a rigid fit.

    Attributes:
        coords (np.ndarray): (n_conf, n_atoms, 3) positions, in the atom order of Ligand.atoms.
        energies (np.ndarray): (n_conf,) UFF energies (kcal/mol) relative to reference_energy;
            all zero when the conformers were not minimized.
        anchors (np.ndarray): (n_conf, n_motifs, 3) centroid of each motif's binding atoms.
        directions (np.ndarray): (n_conf, n_motifs, 3) unit vector from each anchor into the
            ligand, which should point away from the surface; zero where undefined.
        body (np.ndarray): (n_atoms,) mask of the heavy atoms outside the binding groups,
            whose spread over the surface measures a conformer's footprint.
        stretched (np.ndarray): (n_conf,) templates whose chains were stood up (with_template).
        mirrored (bool): Whether the second half of the embedded conformers mirrors the first.
        reference_energy (float): Absolute UFF energy (kcal/mol) the energies are relative to.
        n_templates (int): Poses appended by with_template, the last conformers; they have
            no mirror image.
    """
    coords: np.ndarray
    energies: np.ndarray
    anchors: np.ndarray
    directions: np.ndarray
    body: np.ndarray
    stretched: np.ndarray
    mirrored: bool = False
    reference_energy: float = 0.0
    n_templates: int = 0

    @property
    def n_conformers(self) -> int:
        return len(self.coords)

    @property
    def first_template(self) -> int:
        """Index of the first template: the conformers before it are the embedded ones."""
        return self.n_conformers - self.n_templates

    def unique_id(self, conformer: int) -> int:
        """
        The conformer, or the one it mirrors: on a mirror-symmetric surface the two make
        the same attempt.
        """
        if conformer >= self.first_template or not self.mirrored:
            return conformer
        return conformer % (self.first_template // 2)

    def with_template(self, mol: Chem.Mol, groups: List[List[int]], x: np.ndarray,
                      energy: float, stretched: bool = True) -> "ConformerLibrary":
        """
        This library plus the pose `x` (relative UFF energy `energy`) as a template: a pose
        rescued at a real site, which the next sites of the same geometry fit at once.
        `stretched` says whether its chains were stood up.
        """
        anchors, directions = _anchor_geometry(mol, groups, x[None])
        return dataclasses.replace(self,
                                   coords=np.concatenate([self.coords, x[None]]),
                                   energies=np.append(self.energies, energy),
                                   anchors=np.concatenate([self.anchors, anchors]),
                                   directions=np.concatenate([self.directions, directions]),
                                   stretched=np.append(self.stretched, stretched),
                                   n_templates=self.n_templates + 1)

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
        reference = float(energies.min())
        energies = energies - reference
        if mirror is None:
            mirror = all(a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
        if mirror:
            coords = np.concatenate([coords, coords * np.array([-1.0, 1.0, 1.0])])
            energies = np.concatenate([energies, energies])
        anchors, directions = _anchor_geometry(mol, ligand.binding_groups, coords)

        body = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
        body[[i for g in ligand.binding_groups for i in g]] = False
        if not body.any():
            body[:] = True
        return cls(coords=coords, energies=energies, anchors=anchors, directions=directions,
                   body=body, stretched=np.zeros(len(coords), dtype=bool), mirrored=bool(mirror),
                   reference_energy=reference)


def uff_polish(mol: Chem.Mol, x: np.ndarray, fixed: List[int], max_displ: float = 0.3,
                iters: int = 300) -> Tuple[np.ndarray, float]:
    """
    Relax a pose with UFF, `fixed` atoms held in place and other heavy atoms kept within
    `max_displ` Å: removes the close contacts torsion moves leave without changing the
    shape. Returns the coordinates and their UFF energy.
    """
    m = Chem.Mol(mol)
    m.GetConformer().SetPositions(np.asarray(x, dtype=float))
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


def embed_at(mol: Chem.Mol, pins: dict, n_conformers: int, random_seed: int,
             n_threads: int = 0, uff_iters: int = 500) -> np.ndarray:
    """
    Conformers of `mol` with the atoms of `pins` (index -> position) where given: ETKDG
    with a coordinate map, relaxed with UFF with those atoms held, superposed on the pins.
    Returns (k, n_atoms, 3), k <= n_conformers (0 if nothing embeds).
    """
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    pinned = [int(a) for a in pins]
    target = np.array([pins[a] for a in pinned], dtype=float)
    cmap = {a: Point3D(*map(float, p)) for a, p in zip(pinned, target)}
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_conformers, coordMap=cmap,
                                           randomSeed=random_seed, numThreads=n_threads,
                                           useRandomCoords=True))
    uff = AllChem.UFFHasAllMoleculeParams(m)
    out = []
    for cid in cids:
        if uff:
            ff = AllChem.UFFGetMoleculeForceField(m, confId=cid)
            for a in pinned:
                ff.UFFAddPositionConstraint(a, 0.0, 1.0e4)
            ff.Minimize(maxIts=uff_iters)
        x = np.array(m.GetConformer(cid).GetPositions())
        # the embedding keeps the pins' distances, not their frame
        P, Q = x[pinned], target
        p0, q0 = P.mean(axis=0), Q.mean(axis=0)
        U, _, Vt = np.linalg.svd((P - p0).T @ (Q - q0))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        out.append((x - p0) @ (U @ D @ Vt) + q0)
    return np.array(out).reshape((-1, m.GetNumAtoms(), 3))


def uff_energy(mol: Chem.Mol, x: np.ndarray) -> float:
    """UFF energy (kcal/mol) of `mol` at positions `x`."""
    m = Chem.Mol(mol)
    m.GetConformer().SetPositions(np.asarray(x, dtype=float))
    return AllChem.UFFGetMoleculeForceField(m).CalcEnergy()


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
