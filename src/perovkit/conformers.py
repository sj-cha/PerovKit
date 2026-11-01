from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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
    """
    coords: np.ndarray
    energies: np.ndarray
    anchors: np.ndarray
    directions: np.ndarray

    @property
    def n_conformers(self) -> int:
        return len(self.coords)

    @classmethod
    def build(
        cls,
        ligand: Ligand,
        n_conformers: int = 50,
        prune_rms: float = 0.5,
        optimize: bool = True,
        random_seed: int = 42,
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
            prune_rms (float): RMSD (Å) below which ETKDG discards a conformer as a duplicate.
            optimize (bool): Minimize every conformer with UFF and record its energy.
            random_seed (int): Seed for ETKDG embedding.
        """
        if ligand.mol is None:
            raise ValueError("ConformerLibrary needs the ligand's RDKit mol")
        if not ligand.binding_atoms:
            raise ValueError("ConformerLibrary needs a ligand with binding motifs")

        mol = Chem.Mol(ligand.mol)  # conformer 0 holds the current geometry
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.pruneRmsThresh = prune_rms
        params.clearConfs = False
        params.numThreads = 0
        n_new = len(AllChem.EmbedMultipleConfs(mol, n_conformers, params))
        if n_new < n_conformers // 2:
            # large molecules mostly fail to embed unless started from random coordinates
            params.useRandomCoords = True
            AllChem.EmbedMultipleConfs(mol, n_conformers - n_new, params)

        energies = np.zeros(mol.GetNumConformers())
        if optimize and AllChem.UFFHasAllMoleculeParams(mol):
            results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
            energies = np.array([e for _, e in results], dtype=float)
            energies -= energies.min()

        coords = np.array([c.GetPositions() for c in mol.GetConformers()], dtype=float)
        anchors, directions = _anchor_geometry(mol, ligand.binding_groups, coords)
        return cls(coords=coords, energies=energies, anchors=anchors, directions=directions)

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
