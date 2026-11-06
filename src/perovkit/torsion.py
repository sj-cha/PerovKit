from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from ase.data import vdw_radii
import numpy as np
from rdkit import Chem

# acyclic single bonds between atoms neither terminal nor triple-bonded
ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def dihedral(p: np.ndarray) -> float:
    """
    Dihedral angle (radians) of four points, positive for a right-handed turn of p[3]
    about the axis p[1] -> p[2].
    """
    b1 = p[2] - p[1]
    b1 = b1 / np.linalg.norm(b1)
    v = (p[0] - p[1]) - np.dot(p[0] - p[1], b1) * b1
    w = (p[3] - p[2]) - np.dot(p[3] - p[2], b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


@dataclass
class TorsionModel:
    """
    Rotatable torsions of one ligand type and the pairs it must keep apart.

    Attributes:
        bonds (np.ndarray): (k, 2) atoms (i, j) of each rotatable bond; turning it moves the
            side of j, which never holds the root (the first binding atom).
        masks (np.ndarray): (k, n_atoms) atoms moved by each torsion, deepest bond first.
        far (np.ndarray): (n_atoms, n_atoms) whether two atoms are at least four bonds
            apart, the pairs kept apart within the ligand.
        quads (np.ndarray): (k, 4) atoms (a, i, j, b) defining each torsion's dihedral.
        tails (np.ndarray): (k,) torsions whose moving side holds no binding atom, so they
            turn a dangling chain without moving any anchor.
        heavy (np.ndarray): (n_atoms,) non-hydrogen atoms.
        radii (np.ndarray): (n_atoms,) van der Waals radii (Å, ase.data.vdw_radii).
    """
    bonds: np.ndarray
    masks: np.ndarray
    far: np.ndarray
    quads: np.ndarray
    tails: np.ndarray
    heavy: np.ndarray
    radii: np.ndarray

    def __post_init__(self):
        # the bonds and the atoms each torsion moves, flat, for the compiled loops
        moving = [np.flatnonzero(m) for m in self.masks]
        self._bonds = np.ascontiguousarray(self.bonds, dtype=np.int64).reshape(-1, 2)
        self._flat_moving = np.concatenate(moving + [np.zeros(0, dtype=np.int64)]).astype(np.int64)
        self._n_moving = np.array([len(m) for m in moving], dtype=np.int64)
        a, b = np.nonzero(np.triu(self.far))
        self._far_pairs = (a.astype(np.int64), b.astype(np.int64))

    @property
    def frame(self) -> np.ndarray:
        """
        (n_atoms,) atoms that no tail torsion moves: what holds the binding groups together,
        which turning the dangling chains and rings cannot take out of a neighbor's way.
        """
        return ~self.masks[self.tails].any(axis=0)

    @property
    def heavy_tails(self) -> np.ndarray:
        """
        (k,) tail torsions that move a heavy atom: they turn a chain, not just the hydrogens
        of a methyl or ammonium group (the bond's own atom j turns in place).
        """
        moved = self.masks.copy()
        moved[np.arange(self.n_torsions), self.bonds[:, 1]] = False
        return self.tails & (moved & self.heavy).any(axis=1)

    @classmethod
    def build(cls, mol: Chem.Mol, binding: Sequence[int]) -> TorsionModel:
        """Torsions of `mol` rooted at its first binding atom, with `binding` its binding atoms."""
        root = binding[0]
        n = mol.GetNumAtoms()
        heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
        nbrs = [[x.GetIdx() for x in a.GetNeighbors()] for a in mol.GetAtoms()]
        topo = Chem.GetDistanceMatrix(mol)

        def side(start: int, cut: int) -> np.ndarray:
            seen = {start}
            stack = [start]
            while stack:
                a = stack.pop()
                for b in nbrs[a]:
                    if b not in seen and not (a == start and b == cut):
                        seen.add(b)
                        stack.append(b)
            mask = np.zeros(n, dtype=bool)
            mask[list(seen)] = True
            return mask

        found = []
        for i, j in mol.GetSubstructMatches(ROTATABLE):
            mask = side(j, i)
            if mask[root]:
                i, j = j, i
                mask = side(j, i)
            found.append((topo[root, i], i, j, mask))
        found.sort(key=lambda f: -f[0])  # deepest first, see forward()

        bonds = np.array([(i, j) for _, i, j, _ in found], dtype=int).reshape(-1, 2)
        masks = np.array([m for *_, m in found], dtype=bool).reshape(-1, n)

        def outer(atom: int, other: int) -> int:
            cand = [x for x in nbrs[atom] if x != other]
            return max(cand, key=lambda x: heavy[x])  # a heavy neighbor when there is one
        quads = np.array([(outer(i, j), i, j, outer(j, i)) for i, j in bonds], dtype=int).reshape(-1, 4)
        binding = np.asarray(list(binding), dtype=int)
        tails = ~masks[:, binding].any(axis=1)
        # 1.8 Å for the elements ASE has no van der Waals radius for
        radii = np.nan_to_num(vdw_radii[[a.GetAtomicNum() for a in mol.GetAtoms()]], nan=1.8)
        return cls(bonds=bonds, masks=masks, far=topo >= 4, quads=quads, tails=tails, heavy=heavy,
                   radii=radii)

    @property
    def n_torsions(self) -> int:
        return len(self.bonds)
