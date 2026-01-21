from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.spatial import cKDTree

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from .utils.rotation import rotation_about_axis, rotation_from_u_to_v

Plane = Tuple[int, int, int]

@dataclass
class BindingMotif:
    atoms: list[str]

@dataclass
class Ligand:
    atoms: Atoms
    mol: Chem.Mol
    smiles: str
    charge: int
    binding_motif: BindingMotif
    name: str
    id: Optional[int] = None
    volume: float = None
    binding_atoms: List[int] = field(default_factory=list)
    anchor_pos: Optional[np.ndarray] = None
    plane: Optional[Plane] = None
    indices: Optional[np.ndarray] = None
    _neighbor_cutoff: float = 2.0
    _anchor_offset: float = 0.0

    def __post_init__(self):
        self._get_volume()
        self._get_binding_atoms_indices()
        self._orient_ligand()

    @classmethod
    def from_xyz(
        cls,
        xyz_path: str,
        binding_motif: BindingMotif,
        name: str,
        charge: Optional[int] = None,     
        **kwargs
    ) -> Ligand:
        
        atoms = read(xyz_path)
        mol = Chem.MolFromXYZFile(rf"{xyz_path}")

        if charge is None:
            candidate_charges = (-1, 0, 1)
        else:
            candidate_charges = (charge,)

        chosen_mol = None
        chosen_charge = None

        for q in candidate_charges:
            m = Chem.Mol(mol)
            try:
                rdDetermineBonds.DetermineBonds(m, charge=q)
                Chem.SanitizeMol(m)
            except Exception as e:
                continue

            chosen_mol = m
            chosen_charge = q
            break

        if chosen_mol is None:
            raise ValueError(
                f"Failed to infer charge. Please provide the charge as an argument."
            )

        no_H = Chem.RemoveHs(chosen_mol)
        
        return cls(atoms=atoms, 
                   mol=chosen_mol, 
                   smiles = Chem.MolToSmiles(no_H), 
                   charge=chosen_charge, 
                   binding_motif=binding_motif,
                   name=name,
                   **kwargs)
    
    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        binding_motif: BindingMotif,
        random_seed: int,
        name: str,
        **kwargs
    ) -> Ligand:
        
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        AllChem.EmbedMolecule(mol, params)
        AllChem.UFFOptimizeMolecule(mol)

        positions = mol.GetConformers()[0].GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        atoms = Atoms(positions=positions, symbols=symbols)

        return cls(
            atoms=atoms, 
            mol=mol, 
            smiles=smiles, 
            charge=Chem.GetFormalCharge(mol), 
            binding_motif=binding_motif,
            name=name,
            **kwargs)
    
    def clone(self) -> Ligand:
        lig_cloned = object.__new__(Ligand)
        lig_cloned.__dict__ = self.__dict__.copy()
        lig_cloned.atoms = self.atoms.copy()

        return lig_cloned

    def _get_volume(self) -> float:        
        self.volume = float(AllChem.ComputeMolVolume(self.mol))
    
    def _get_binding_atoms_indices(self) -> List[int]:

        symbols = self.atoms.get_chemical_symbols()
        coords  = self.atoms.get_positions()

        binding_elems = list(self.binding_motif.atoms)

        if len(binding_elems) == 2:
            elem1, elem2 = binding_elems

            idx1 = [i for i, s in enumerate(symbols) if s == elem1]
            idx2 = [i for i, s in enumerate(symbols) if s == elem2]

            if len(idx1) == 0 or len(idx2) == 0:
                raise ValueError(f"No atoms found for binding motif {binding_elems}")

            best_pair = None
            best_dist = np.inf

            for i in idx1:
                p1 = coords[i]
                for j in idx2:
                    p2 = coords[j]
                    d = np.linalg.norm(p1 - p2)
                    if i!=j and d < best_dist:
                        best_dist = d
                        best_pair = (i, j)

            self.binding_atoms = list(best_pair)

        elif len(binding_elems) == 1:
            elem = binding_elems[0]

            elem_indices = [i for i, s in enumerate(symbols) if s == elem]
            if len(elem_indices) == 0:
                raise ValueError(f"No atoms found for binding element {elem!r}")

            elem_coords = coords[elem_indices]

            # cKDTree neighbor counting within 2 Å
            tree = cKDTree(coords)
            neighbors = tree.query_ball_point(elem_coords, r=self._neighbor_cutoff)

            # coordination number excluding self
            cn = np.fromiter((len(nbrs) - 1 for nbrs in neighbors), dtype=int)

            min_cn = int(np.min(cn))
            candidate_local = np.where(cn == min_cn)[0]

            # choose the first one among candidates
            chosen_global = elem_indices[candidate_local[0]]

            self.binding_atoms = [chosen_global]

        else:
            raise NotImplementedError(
                "Binding motifs with more than 2 atoms are not yet supported."
            )

    def _orient_ligand(self, n_angles: int = 720):
        coords = self.atoms.get_positions()
        binding_idx = self.binding_atoms
        assert 2 >= len(binding_idx) > 0, "Need 1 or 2 binding atoms"

        # Compute the centroid of the binding atoms
        coords_centroid = coords[binding_idx].mean(axis=0)
        coords0 = coords - coords_centroid

        # Define the rotation axis
        if len(binding_idx) >= 2:
            axis = coords0[binding_idx[1]] - coords0[binding_idx[0]]
            axis /= np.linalg.norm(axis)
        else:
            b = binding_idx[0]  
            bind_pos = coords0[b]

            deltas = coords0 - bind_pos # (N, 3)
            d2 = np.einsum("ij,ij->i", deltas, deltas)  # squared distances
            d2[b] = -np.inf # exclude self

            # if there is at least one other atom, use the farthest one
            j = int(np.argmax(d2))
            axis = deltas[j]                 
            axis /= np.linalg.norm(axis)

            z = np.array([0.0, 0.0, 1.0], dtype=float)
            R_align = rotation_from_u_to_v(axis, z)  
            coords0 = coords0 @ R_align.T

            self.atoms.set_positions(coords0)

            return None
        
        ex = np.array([1.0, 0.0, 0.0], dtype=float)
        R_align = rotation_from_u_to_v(axis, ex)  
        coords0 = coords0 @ R_align.T

        # Rest of the atoms 
        mask = np.ones(len(coords0), dtype=bool)
        mask[binding_idx] = False
        others = coords0[mask]

        assert len(others) > 0, "No other atoms to orient ligand"

        # Rotate to maximize z-projection
        thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
        best_theta = 0.0
        best_score = -np.inf

        for th in thetas:
            R = rotation_about_axis(ex, th)
            rotated = others @ R.T  # (N_other, 3)
            score = rotated[:, 2].sum()
            if score > best_score:
                best_score = score
                best_theta = th

        R_best = rotation_about_axis(ex, best_theta)
        rotated_all = coords0 @ R_best.T

        # update ASE atoms
        self.atoms.set_positions(rotated_all)

    def to(self, fmt: str = 'xyz', filename: str = None) -> None:
        """Export ligand to file."""

        formula = self.atoms.get_chemical_formula()

        write(filename, self.atoms, format=fmt, comment=formula)

@dataclass
class LigandSpec:
    ligand: Ligand
    coverage: float
    anchor_offset: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.ligand.name