from __future__ import annotations
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.io.vasp import write_vasp
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolTransforms

from .utils.rotation import rotation_about_axis, rotation_from_u_to_v

MAX_BITE_DISTANCE = 3.5


@dataclass
class BindingMotif:
    """
    Specification of how a ligand anchors to a surface.
    """
    atoms: list[str]
    indices: Optional[list[int]] = None

    def __post_init__(self):
        if not 1 <= len(self.atoms) <= 2:
            raise NotImplementedError(
                f"Binding motifs with more than 2 atoms are not yet supported; "
                f"got {len(self.atoms)}: {self.atoms}"
            )
        if self.indices is None:
            return
        self.indices = [int(i) for i in self.indices]
        if len(self.indices) != len(self.atoms):
            raise ValueError(
                f"BindingMotif: {len(self.indices)} indices for {len(self.atoms)} atoms; "
                f"give one index per element symbol, in the same order"
            )
        if any(i < 0 for i in self.indices):
            raise ValueError(f"BindingMotif: indices must be non-negative, got {self.indices}")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"BindingMotif: indices must be distinct, got {self.indices}")


@dataclass
class Ligand:
    """
    Surface-passivating ligand molecule.

    Attributes:
        atoms (ASE Atoms): ASE Atoms object.
        mol (Chem.Mol): RDKit Mol object.
        smiles (str): SMILES string of the ligand without explicit hydrogens.
        charge (int): Formal charge of the ligand.
        binding_motif (BindingMotif): Specification of how a ligand anchors to a surface.
        name (str): User-defined identifier.
        id (int): Ligand ID assigned after placement.
        volume (float): Molecular volume (Å^3).
        binding_atoms (List[int]): Local indices of atoms used as anchors.
        plane (Tuple[int, int, int]): Miller index of the surface the ligand is bound to.
        indices (np.ndarray): Global atom indices when placed in a parent NanoCrystal/Slab.
        _anchor_offset (float): Offset (Å) applied along the axis perpendicular to the plane.
    """
    atoms: Atoms
    mol: Chem.Mol
    smiles: str
    charge: int
    binding_motif: BindingMotif
    name: str
    id: Optional[int] = None
    volume: float = None
    binding_atoms: List[int] = field(default_factory=list)
    plane: Optional[Tuple[int, int, int]] = None
    indices: Optional[np.ndarray] = None
    _anchor_offset: float = 0.0

    def __post_init__(self):
        mol = self.mol
        if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != self.atoms.get_chemical_symbols():
            raise ValueError("mol and atoms must list the same elements in the same order")
        self._get_volume()
        if self.binding_motif:
            self._get_binding_atoms_indices()
            self._orient_ligand()


    @property
    def anchor_pos(self) -> Optional[np.ndarray]:
        if not self.binding_atoms:
            return None
        pos = self.atoms.get_positions()
        return pos[np.asarray(self.binding_atoms, dtype=int)].mean(axis=0)


    @classmethod
    def from_file(
        cls,
        filename: str,
        binding_motif: BindingMotif,
        name: str,
        charge: Optional[int] = None,
        **kwargs
    ) -> Ligand:
        """
        Build a Ligand from a file. 
        Currently only XYZ files are supported.

        Args:
            filename (str): Path to the XYZ file.
            binding_motif (BindingMotif): Specification of how a ligand anchors to a surface.
            name (str): User-defined identifier.
            charge (Optional[int]): Formal charge; if None, tries -1, 0, +1 in order.
            **kwargs: Forwarded to the Ligand constructor.

        Returns:
            A Ligand instance.
        """
        atoms = read(filename)
        mol = Chem.MolFromXYZFile(rf"{filename}")

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
        binding_motif: BindingMotif = None,
        random_seed: int = 42,
        name: str = None,
        optimize: bool = False,
        **kwargs
    ) -> Ligand:
        """
        Build a Ligand from a SMILES string.

        Args:
            smiles (str): SMILES string of the ligand.
            binding_motif (BindingMotif): Specification of how a ligand anchors to a surface.
            random_seed (int): Random seed for ETKDG embedding.
            name (str): User-defined identifier.
            optimize (bool): If True, fully relax with UFF; otherwise set rotatable bonds 
                             to 180° (extended) and relax with torsions fixed.
            **kwargs: Forwarded to the Ligand constructor.

        Returns:
            A Ligand instance.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        conf_id = AllChem.EmbedMolecule(mol, params)

        if conf_id < 0:
            mol.RemoveAllConformers()
            params.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, params)

        if conf_id < 0:
            raise ValueError(
                f"RDKit failed to generate a conformer for ligand."
            )

        if optimize:
            AllChem.UFFOptimizeMolecule(
                mol,
                confId=conf_id,
                maxIters=1000,
            )
        else:
            # Set all rotatable bond dihedrals to 180°
            rotatable = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
            matches = mol.GetSubstructMatches(rotatable)
            conf = mol.GetConformer()
            torsion_indices = []
            for match in matches:
                i, j = match
                neighbors_i = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
                neighbors_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
                if neighbors_i and neighbors_j:
                    quad = (neighbors_i[0], i, j, neighbors_j[0])
                    rdMolTransforms.SetDihedralDeg(conf, *quad, 180.0)
                    torsion_indices.append(quad)

            # Relax bonds/angles while keeping torsions fixed at 180°
            ff = AllChem.UFFGetMoleculeForceField(mol)
            for a, b, c, d in torsion_indices:
                ff.UFFAddTorsionConstraint(a, b, c, d, False, 180.0, 180.0, 1.0e5)
            ff.Minimize(maxIts=200)

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
        """
        Return a shallow copy with independent atoms and indices.
        """
        lig = copy.copy(self)
        lig.atoms = self.atoms.copy()
        if self.indices is not None:
            lig.indices = self.indices.copy()
        return lig


    @classmethod
    def _from_data(cls, **kwargs) -> Ligand:
        """
        Construct a Ligand bypassing __post_init__. 
        Used when deserializing from the metadata.
        """
        lig = object.__new__(cls)
        # Set dataclass defaults for optional fields
        lig.id = None
        lig.volume = None
        lig.binding_atoms = []
        lig.plane = None
        lig.indices = None
        lig._anchor_offset = 0.0
        for k, v in kwargs.items():
            setattr(lig, k, v)
        return lig


    def _get_volume(self):
        """
        Compute molecular volume (Å^3).
        """
        self.volume = float(AllChem.ComputeMolVolume(self.mol))


    def _coordination_numbers(self) -> np.ndarray:
        """
        Coordination number of every atom, from the RDKit bond graph.
        """
        return np.array([a.GetDegree() for a in self.mol.GetAtoms()], dtype=int)


    def _get_binding_atoms_indices(self) -> List[int]:
        """
        Detect the atom indices that match the binding motif.
        """
        symbols = self.atoms.get_chemical_symbols()
        coords  = self.atoms.get_positions()

        binding_elems = list(self.binding_motif.atoms)
        explicit = self.binding_motif.indices

        if explicit is not None:
            idx = [int(i) for i in explicit]
            n = len(self.atoms)
            for i in idx:
                if not (0 <= i < n):
                    raise ValueError(
                        f"binding motif index {i} out of range for a {n}-atom ligand"
                    )
            if len(set(idx)) != len(idx):
                raise ValueError(f"binding motif indices must be distinct, got {idx}")
            for i, elem in zip(idx, binding_elems):
                if symbols[i] != elem:
                    raise ValueError(
                        f"binding motif index {i} is {symbols[i]!r}, but the motif declares "
                        f"{elem!r}; the indices and the element symbols disagree"
                    )
            self.binding_atoms = idx
            return self.binding_atoms

        cn = self._coordination_numbers()

        if len(binding_elems) == 2:
            elem1, elem2 = binding_elems

            idx1 = [i for i, s in enumerate(symbols) if s == elem1]
            idx2 = [i for i, s in enumerate(symbols) if s == elem2]

            if len(idx1) == 0 or len(idx2) == 0:
                raise ValueError(f"No atoms found for binding motif {binding_elems}")

            best_pair = None
            best_key = None

            for i in idx1:
                p1 = coords[i]
                for j in idx2:
                    if i == j:
                        continue
                    p2 = coords[j]
                    dist = float(np.linalg.norm(p1 - p2))
                    key = (dist > MAX_BITE_DISTANCE,
                           int(cn[i]) + int(cn[j]),
                           dist)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pair = (i, j)

            if best_pair is None:
                raise ValueError(
                    f"No distinct atom pair found for binding motif {binding_elems}"
                )

            self.binding_atoms = list(best_pair)

        elif len(binding_elems) == 1:
            elem = binding_elems[0]

            elem_indices = [i for i, s in enumerate(symbols) if s == elem]
            if len(elem_indices) == 0:
                raise ValueError(f"No atoms found for binding element {elem!r}")

            elem_cn = cn[np.asarray(elem_indices, dtype=int)]

            min_cn = int(np.min(elem_cn))
            candidate_local = np.where(elem_cn == min_cn)[0]

            # choose the first one among candidates
            chosen_global = elem_indices[candidate_local[0]]

            self.binding_atoms = [chosen_global]

        else:
            raise NotImplementedError(
                "Binding motifs with more than 2 atoms are not yet supported."
            )

        return self.binding_atoms


    def _orient_ligand(self, n_angles: int = 720):
        """
        Orient the ligand so the binding axis is canonical and the body points up.
        """
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
            others = np.delete(coords0, b, axis=0)  # coords0[b] is the origin

            if len(others) == 0:
                # monatomic ligand: nothing to orient
                self.atoms.set_positions(coords0)
                return None

            # point binding atom -> unweighted centroid of the rest along +z
            axis = others.mean(axis=0)
            norm = np.linalg.norm(axis)
            if norm < 1e-3:
                # centroid sits on the binding atom (e.g. NH4+): fall back to the farthest atom
                d2 = np.einsum("ij,ij->i", others, others)  # squared distances
                axis = others[int(np.argmax(d2))]
                norm = np.linalg.norm(axis)
            axis /= norm

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


    def to(
        self, 
        fmt: str = 'xyz', 
        filename: str = None,
        vacuum: float = 15.0,
        **kwargs
    ) -> None:
        """
        Write the ligand to file.

        Args:
            fmt (str): File format.
            filename (str): Output path.
        """
        formula = self.atoms.get_chemical_formula()

        if filename is None:
            if self.name is not None:
                filename = f"{self.name}.{fmt}"
            else:
                filename = f"{formula}.{fmt}"
        
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)  

        if fmt == "vasp":
            ase_kwargs = {**kwargs, "sort": True}
            pos = self.atoms.get_positions()
            center = pos.mean(axis=0)
            extent = pos.max(axis=0) - pos.min(axis=0)
            cell_diag = extent + vacuum

            vasp_atoms = self.atoms.copy()
            vasp_atoms.set_cell(np.diag(cell_diag))
            vasp_atoms.positions += (cell_diag / 2 - center)
            vasp_atoms.pbc = True

            write_vasp(str(path), vasp_atoms, **ase_kwargs)
        else:
            formula = self.atoms.get_chemical_formula()
            ase_kwargs = {**kwargs, "format": fmt, "comment": formula}
            write(str(path), self.atoms, **ase_kwargs)



@dataclass
class LigandSpec:
    """
    Pairs a Ligand with placement parameters.

    Attributes:
        ligand (Ligand): The ligand to attach.
        site (Optional[str]): Which sublattice to target, one of "A", "B", or "X".
            This selects the binding sites explicitly instead of inferring them from
            the ligand charge. If None, the target is inferred from the ligand charge
            (positive -> A-site, otherwise X-site) for backward compatibility.
        coverage (Optional[float]): Fractional coverage of available binding sites.
        binding_sites (Optional[list[int]]): Explicit surface site indices to use.
        anchor_offset (float): Offset (Å) along the binding axis. When `adsorb` is
            True this sets the spacing between the surface species and the adsorbate.
        adsorb (bool): If True, adsorb the ligand on top of the surface site instead
            of replacing it (the surface atom is kept). Use `anchor_offset` to control
            the spacing.
        name (Optional[str]): User-defined identifier.
    """
    ligand: Ligand
    site: Optional[str] = None
    coverage: Optional[float] = None
    binding_sites: Optional[list[int]] = None
    anchor_offset: float = 0.0
    adsorb: bool = False
    name: Optional[str] = None

    def __post_init__(self):
        if self.site is not None and self.site not in ("A", "B", "X"):
            raise ValueError(
                f"site must be one of 'A', 'B', 'X' (or None); got {self.site!r}"
            )
        if self.name is None:
            self.name = self.ligand.name