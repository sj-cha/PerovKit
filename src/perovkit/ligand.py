from __future__ import annotations
import copy
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Optional, Tuple, List

import numpy as np
from ase import Atoms
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from .conformers import ConformerLibrary
from .io.structure import write_structure
from .torsion import TorsionModel
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v
from .utils.selection import Part, PartAtoms


@dataclass
class BindingMotif:
    """
    How one binding group of a ligand anchors to the surface. A ligand has one motif per
    binding group, e.g. two for a zwitterion.

    Attributes:
        atoms (list[str]): Elements of the binding atoms (1 or 2), e.g. ["O", "O"].
        anchor (str, optional): SMARTS of the binding group, e.g. "C(=O)[O-]"; the binding
            atoms are taken from its first match and their neighbors.
        indices (list[int], optional): Binding atom indices, one per element of `atoms`,
            used instead of `anchor`.
        site (str, optional): Sublattice it binds to: "A", "B" or "X". Required with several
            motifs; a single motif without one takes A for a cation, X otherwise.
        adsorb (bool): Adsorb on top of the site, keeping its surface atom(s), instead of
            replacing them.
        anchor_offset (float, optional): Height (Å) of the anchor above its site. Default
            3.0 when adsorbing, else 0.
    """
    atoms: list[str]
    anchor: Optional[str] = None
    indices: Optional[list[int]] = None
    site: Optional[str] = None
    adsorb: bool = False
    anchor_offset: Optional[float] = None

    def __post_init__(self):
        if self.anchor_offset is None:
            self.anchor_offset = 3.0 if self.adsorb else 0.0
        self.anchor_offset = float(self.anchor_offset)
        if not 1 <= len(self.atoms) <= 2:
            raise NotImplementedError(
                f"Binding motifs with more than 2 atoms are not yet supported; "
                f"got {len(self.atoms)}: {self.atoms}"
            )
        if self.anchor is None and self.indices is None:
            raise ValueError("BindingMotif: give an anchor SMARTS (or explicit indices)")
        if self.anchor is not None and Chem.MolFromSmarts(self.anchor) is None:
            raise ValueError(f"BindingMotif: invalid anchor SMARTS {self.anchor!r}")
        if self.site is not None and self.site not in ("A", "B", "X"):
            raise ValueError(f"BindingMotif: site must be one of 'A', 'B', 'X' (or None); got {self.site!r}")
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

def pack_motifs(value) -> List[BindingMotif]:
    """A binding_motif value as a list: one motif, several, or none for None."""
    if value is None:
        return []
    if isinstance(value, BindingMotif):
        return [value]
    motifs = list(value)
    if not all(isinstance(m, BindingMotif) for m in motifs):
        raise TypeError("binding_motif must be a BindingMotif or a list of them")
    return motifs


class _SyncedMol:
    """
    Descriptor for Ligand.mol: on every access the conformer is overwritten with the
    current ASE positions, so it follows the ligand's atoms wherever they are moved, in
    an Assembly too.
    """
    def __get__(self, lig, owner=None):
        if lig is None:
            raise AttributeError  # tells dataclass the field has no default
        mol = lig.__dict__.get("_mol")
        # a size mismatch is left for the __post_init__ check to report
        if mol is not None and mol.GetNumAtoms() == len(lig.atoms):
            mol.GetConformer().SetPositions(lig.atoms.get_positions())
        return mol

    def __set__(self, lig, mol):
        lig.__dict__["_mol"] = mol


@dataclass
class Ligand(Part):
    """
    Ligand molecule. Make one with from_smiles or from_file.

    Attributes:
        atoms (Atoms): Atoms of the ligand (read-only inside an Assembly).
        mol (Chem.Mol): RDKit molecule; its conformer follows `atoms`.
        smiles (str): SMILES string.
        charge (int): Formal charge.
        binding_motifs (List[BindingMotif]): One motif per binding group.
        name (str): Name; the chemical formula by default.
        id (int): Index in its Assembly, after placement.
        binding_atoms (List[int]): Indices of the binding atoms, motif after motif.
        planes (List[Tuple[int, int, int]]): Miller index of the surface under each motif,
            after placement.
    """
    atoms: Atoms = PartAtoms()
    mol: Chem.Mol = _SyncedMol()
    smiles: str
    charge: int
    binding_motifs: List[BindingMotif]
    name: str
    id: Optional[int] = None
    volume: float = None
    binding_atoms: List[int] = field(default_factory=list)
    planes: List[Tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        mol = self.mol
        if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != self.atoms.get_chemical_symbols():
            raise ValueError("mol and atoms must list the same elements in the same order")
        if self.name is None:
            self.name = self.atoms.get_chemical_formula()
        self.binding_motifs = pack_motifs(self.binding_motifs)
        if len(self.binding_motifs) == 1 and self.binding_motifs[0].site is None:
            # on a copy: the motif given may be shared with other ligands
            self.binding_motifs = [replace(self.binding_motifs[0], site="A" if self.charge > 0 else "X")]
        missing = [k for k, m in enumerate(self.binding_motifs) if m.site is None]
        if missing:
            raise ValueError(
                f"every motif of a multi-motif ligand needs a site ('A', 'B' or 'X'); "
                f"motif(s) {missing} of ligand {self.name!r} have none"
            )
        # a coarse grid will do: the volume only orders ligands
        self.volume = float(AllChem.ComputeMolVolume(mol, gridSpacing=0.4))
        if self.binding_motifs:
            self.binding_atoms = self._find_binding_atoms()
            self._orient_ligand()


    @property
    def n_motifs(self) -> int:
        return len(self.binding_motifs)

    @property
    def binding_groups(self) -> List[List[int]]:
        """
        ``binding_atoms`` split per motif: each motif owns as many atoms as it lists elements.
        """
        groups, start = [], 0
        for m in self.binding_motifs:
            groups.append(list(self.binding_atoms[start:start + len(m.atoms)]))
            start += len(m.atoms)
        return groups

    @property
    def anchor_pos(self) -> Optional[np.ndarray]:
        if not self.binding_atoms:
            return None
        return self.atoms.get_positions()[self.binding_atoms].mean(axis=0)

    @property
    def anchor_positions(self) -> Optional[np.ndarray]:
        """
        Anchor of every motif, the centroid of its binding atoms: (n_motifs, 3).
        """
        if not self.binding_atoms:
            return None
        pos = self.atoms.get_positions()
        return np.array([pos[g].mean(axis=0) for g in self.binding_groups])

    @property
    def plane(self) -> Optional[Tuple[int, int, int]]:
        return self.planes[0] if self.planes else None


    @classmethod
    def from_file(
        cls,
        filename: str,
        binding_motif: BindingMotif | List[BindingMotif],
        name: str,
        charge: Optional[int] = None,
    ) -> Ligand:
        """
        Read a ligand from an XYZ file.

        Args:
            filename (str): Path to the XYZ file.
            binding_motif (BindingMotif | List[BindingMotif]): One motif, or one per
                binding group.
            name (str): Name of the ligand.
            charge (int, optional): Formal charge; by default -1, 0 and +1 are tried.

        Returns:
            Ligand: The ligand.
        """
        atoms = read(filename)
        mol = Chem.MolFromXYZFile(str(filename))

        for q in (-1, 0, 1) if charge is None else (charge,):
            m = Chem.Mol(mol)
            try:
                rdDetermineBonds.DetermineBonds(m, charge=q)
                Chem.SanitizeMol(m)
            except Exception:
                continue
            break
        else:
            raise ValueError("Failed to infer charge. Please provide the charge as an argument.")

        return cls(atoms=atoms,
                   mol=m,
                   smiles=Chem.MolToSmiles(Chem.RemoveHs(m)),
                   charge=q,
                   binding_motifs=binding_motif,
                   name=name)


    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        binding_motif: BindingMotif | List[BindingMotif] = None,
        random_seed: int = 42,
        name: str = None,
    ) -> Ligand:
        """
        Build a ligand from SMILES, embedded with ETKDG and relaxed with UFF.

        Args:
            smiles (str): SMILES string.
            binding_motif (BindingMotif | List[BindingMotif], optional): One motif, or one
                per binding group; an A-site cation needs none.
            random_seed (int): Seed for the embedding.
            name (str, optional): Name; the chemical formula by default.

        Returns:
            Ligand: The ligand.
        """
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        conf_id = AllChem.EmbedMolecule(mol, params)

        if conf_id < 0:
            mol.RemoveAllConformers()
            params.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, params)

        if conf_id < 0:
            raise ValueError("RDKit failed to generate a conformer for ligand.")

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

        atoms = Atoms(positions=mol.GetConformer().GetPositions(),
                      symbols=[atom.GetSymbol() for atom in mol.GetAtoms()])

        return cls(
            atoms=atoms,
            mol=mol,
            smiles=smiles,
            charge=Chem.GetFormalCharge(mol),
            binding_motifs=binding_motif,
            name=name)


    def conformers(self, **kwargs) -> ConformerLibrary:
        """
        Conformer library of this ligand, built on first use and shared with clones made after.

        Args:
            **kwargs: Forwarded to ConformerLibrary.build; giving any rebuilds the library.
        """
        lib = getattr(self, "_conformers", None)
        if lib is None or kwargs:
            lib = ConformerLibrary.build(self, **kwargs)
            self._conformers = lib
        return lib


    def torsion_model(self) -> TorsionModel:
        """
        Rotatable torsions used to refine this ligand, rooted at its first binding atom;
        built on first use and shared with clones made after.
        """
        model = getattr(self, "_torsions", None)
        if model is None:
            model = TorsionModel.build(self.mol, self.binding_atoms)
            self._torsions = model
        return model


    def clone(self) -> Ligand:
        """
        Return a shallow copy with independent atoms and mol, part of no structure.
        """
        lig = copy.copy(self)
        mol = self.mol
        lig._detach(self.atoms.copy())
        if mol is not None:
            lig.mol = Chem.Mol(mol)  # own conformer, so it can follow its own atoms
        lig.planes = list(self.planes)
        return lig


    @classmethod
    def _from_data(cls, **kwargs) -> Ligand:
        """
        Construct a Ligand bypassing __post_init__.
        Used when deserializing from the metadata.
        """
        lig = object.__new__(cls)
        lig.binding_atoms = []
        lig.planes = []
        for k, v in kwargs.items():
            setattr(lig, k, v)
        return lig


    def _anchor_region(self, motif: BindingMotif, claimed: set) -> List[int]:
        """
        Atoms of the first anchor match and their bonded neighbors, sorted by index.

        Matches touching `claimed` atoms (bound by an earlier motif) are skipped, so motifs
        sharing one SMARTS take its groups in order.
        """
        anchor = motif.anchor
        mol = self.mol
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(anchor))
        if not matches:
            raise ValueError(f"anchor {anchor!r} matches nothing in ligand {self.smiles!r}")
        matches = [m for m in matches if claimed.isdisjoint(m)]
        if not matches:
            raise ValueError(f"anchor {anchor!r} has no group left in ligand {self.smiles!r} "
                             f"once the earlier motifs took theirs")

        region = set(matches[0])
        for i in matches[0]:
            region.update(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors())
        return sorted(region - claimed)


    def _find_binding_atoms(self) -> List[int]:
        """
        The atom indices that match the binding motifs, motif after motif.
        """
        binding_atoms: List[int] = []
        for motif in self.binding_motifs:
            binding_atoms += self._resolve_motif(motif, set(binding_atoms))

        # Matches sharing atoms are the same group seen from another atom
        anchors = [m.anchor for m in self.binding_motifs if m.indices is None]
        for anchor in dict.fromkeys(anchors):
            groups: List[set] = []
            for match in self.mol.GetSubstructMatches(Chem.MolFromSmarts(anchor)):
                if all(g.isdisjoint(match) for g in groups):
                    groups.append(set(match))
            if len(groups) > anchors.count(anchor):
                print(f"[Warning] anchor {anchor!r} matches {len(groups)} groups in ligand "
                      f"{self.name}, but only {anchors.count(anchor)} "
                      f"motif(s) use it; the other groups stay unbound.")
        return binding_atoms


    def _resolve_motif(self, motif: BindingMotif, claimed: set) -> List[int]:
        """
        Binding atom indices of one motif, never reusing `claimed` atoms: those it gives, or
        else the atoms of its elements in the anchor region with the fewest bonds between
        them and, among those, the closest pair.
        """
        symbols = self.atoms.get_chemical_symbols()

        if motif.indices is not None:
            for i in motif.indices:
                if i >= len(symbols):
                    raise ValueError(
                        f"binding motif index {i} out of range for a {len(symbols)}-atom ligand"
                    )
            if not claimed.isdisjoint(motif.indices):
                raise ValueError(f"binding motif indices {motif.indices} reuse atoms of another motif")
            for i, elem in zip(motif.indices, motif.atoms):
                if symbols[i] != elem:
                    raise ValueError(
                        f"binding motif index {i} is {symbols[i]!r}, but the motif declares "
                        f"{elem!r}; the indices and the element symbols disagree"
                    )
            return motif.indices

        region = self._anchor_region(motif, claimed)
        coords = self.atoms.get_positions()
        degree = [a.GetDegree() for a in self.mol.GetAtoms()]

        options = [[i for i in region if symbols[i] == elem] for elem in motif.atoms]
        combos = [c for c in product(*options) if len(set(c)) == len(c)]
        if not combos:
            raise ValueError(f"No atoms found for binding motif {motif.atoms} "
                             f"around anchor {motif.anchor!r}")

        return list(min(combos, key=lambda c: (sum(degree[i] for i in c),
                                               float(np.linalg.norm(coords[c[0]] - coords[c[-1]])))))


    def _orient_ligand(self) -> None:
        """
        Orient the ligand so the binding axis is canonical and the body points up: one
        binding atom sits at the origin with the rest of the ligand along +z, two lie along
        x with the rest turned toward +z.

        A ligand with several motifs is only centered on the mean of its anchors: its
        orientation is set at placement, by fitting every anchor to its own site.
        """
        coords = self.atoms.get_positions()
        if self.n_motifs > 1:
            self.atoms.set_positions(coords - self.anchor_positions.mean(axis=0))
            return

        binding_idx = self.binding_atoms
        coords0 = coords - coords[binding_idx].mean(axis=0)
        others = np.delete(coords0, binding_idx, axis=0)

        if len(binding_idx) == 2:
            axis = coords0[binding_idx[1]] - coords0[binding_idx[0]]
            axis /= np.linalg.norm(axis)

            ex = np.array([1.0, 0.0, 0.0], dtype=float)
            coords0 = coords0 @ rotation_from_u_to_v(axis, ex).T

            # Turn about x by the angle t that maximizes the z sum of the rest,
            # sum(y) sin t + sum(z) cos t
            others = np.delete(coords0, binding_idx, axis=0)
            theta = np.arctan2(others[:, 1].sum(), others[:, 2].sum())
            coords0 = coords0 @ rotation_about_axis(ex, theta).T

        elif len(others) > 0:
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
            coords0 = coords0 @ rotation_from_u_to_v(axis, z).T

        self.atoms.set_positions(coords0)


    def to(
        self, 
        fmt: str = 'xyz', 
        filename: str = None,
        vacuum: float = 20.0,
        **kwargs
    ) -> None:
        """
        Write the ligand to a file.

        Args:
            fmt (str): File format, e.g. "xyz" or "vasp".
            filename (str, optional): Output path; `name.fmt` by default.
            vacuum (float): Vacuum (Å) around the ligand for VASP.
            **kwargs: Passed to the ASE writer.
        """
        if filename is None:
            filename = f"{self.name}.{fmt}"

        write_structure(self.atoms, filename, fmt, center_in_box=True, vacuum=vacuum, **kwargs)
