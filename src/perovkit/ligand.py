from __future__ import annotations
import copy
from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Tuple, List

import numpy as np
from ase import Atoms
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolTransforms

from .conformers import ConformerLibrary
from .io.structure import write_structure
from .torsion import ROTATABLE, TorsionModel
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v
from .utils.selection import Part, PartAtoms


@dataclass
class BindingMotif:
    """
    Specification of how a ligand anchors to a surface.

    A ligand holds one motif per independent binding group; a zwitterion, for example,
    has one for its cationic and one for its anionic end.

    Attributes:
        atoms (list[str]): Element symbols of the binding atoms (1 or 2), e.g. ["O", "O"].
        anchor (str): SMARTS of the binding group, e.g. "P" for a phosphonate. The binding
            atoms are searched among the atoms of its first match and their bonded neighbors.
        indices (list[int]): Explicit binding atom indices, one per entry of ``atoms``.
            When given they are used as-is and ``anchor`` is not required.
        site (str): Sublattice this group binds to, one of "A", "B" or "X". Required for
            every motif of a ligand with more than one; for a single motif it may be left
            to ``LigandSpec.site`` or the ligand charge.
    """
    atoms: list[str]
    anchor: Optional[str] = None
    indices: Optional[list[int]] = None
    site: Optional[str] = None

    def __post_init__(self):
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
    """A binding_motifs value as a list: one motif, several, or none for None."""
    if value is None:
        return []
    if isinstance(value, BindingMotif):
        return [value]
    motifs = list(value)
    if not all(isinstance(m, BindingMotif) for m in motifs):
        raise TypeError("binding_motifs must be a BindingMotif or a list of them")
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
    Surface-passivating ligand molecule.

    Attributes:
        atoms (ASE Atoms): ASE Atoms object; in an Assembly, a read-only selection of its atoms.
        mol (Chem.Mol): RDKit Mol object; its conformer follows ``atoms`` positions.
        smiles (str): SMILES string of the ligand without explicit hydrogens.
        charge (int): Formal charge of the ligand.
        binding_motifs (List[BindingMotif]): How the ligand anchors to a surface, one motif
            per binding group; a single motif may be given as it is.
        name (str): User-defined identifier; the chemical formula when none is given.
        id (int): Ligand ID assigned after placement.
        volume (float): Molecular volume (Å^3).
        binding_atoms (List[int]): Local indices of atoms used as anchors, motif after motif
            (see ``binding_groups`` for the split).
        planes (List[Tuple[int, int, int]]): Miller index of the surface each motif is bound
            to; ``plane`` is the first one.
        _anchor_offset (float): Offset (Å) applied along the axis perpendicular to the plane.
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
    _anchor_offset: float = 0.0

    def __post_init__(self):
        mol = self.mol
        if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != self.atoms.get_chemical_symbols():
            raise ValueError("mol and atoms must list the same elements in the same order")
        if self.name is None:
            self.name = self.atoms.get_chemical_formula()
        self.binding_motifs = pack_motifs(self.binding_motifs)
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
        binding_motifs: BindingMotif | List[BindingMotif],
        name: str,
        charge: Optional[int] = None,
    ) -> Ligand:
        """
        Build a Ligand from a file. 
        Currently only XYZ files are supported.

        Args:
            filename (str): Path to the XYZ file.
            binding_motifs (BindingMotif | List[BindingMotif]): How the ligand anchors to a
                surface; a list gives one motif per binding group.
            name (str): User-defined identifier.
            charge (Optional[int]): Formal charge; if None, tries -1, 0, +1 in order.

        Returns:
            A Ligand instance.
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
                   binding_motifs=binding_motifs,
                   name=name)


    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        binding_motifs: BindingMotif | List[BindingMotif] = None,
        random_seed: int = 42,
        name: str = None,
        optimize: bool = False,
    ) -> Ligand:
        """
        Build a Ligand from a SMILES string.

        Args:
            smiles (str): SMILES string of the ligand.
            binding_motifs (BindingMotif | List[BindingMotif]): How the ligand anchors to a
                surface; a list gives one motif per binding group.
            random_seed (int): Random seed for ETKDG embedding.
            name (str): User-defined identifier; the chemical formula by default.
            optimize (bool): If True, fully relax with UFF; otherwise set rotatable bonds 
                             to 180° (extended) and relax with torsions fixed.

        Returns:
            A Ligand instance.
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

        if optimize:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)
        else:
            # Set all rotatable bond dihedrals to 180°
            conf = mol.GetConformer()
            torsion_indices = []
            for i, j in mol.GetSubstructMatches(ROTATABLE):
                neighbor_i = next(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j)
                neighbor_j = next(n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i)
                quad = (neighbor_i, i, j, neighbor_j)
                rdMolTransforms.SetDihedralDeg(conf, *quad, 180.0)
                torsion_indices.append(quad)

            # Relax bonds/angles while keeping torsions fixed at 180°
            ff = AllChem.UFFGetMoleculeForceField(mol)
            for a, b, c, d in torsion_indices:
                ff.UFFAddTorsionConstraint(a, b, c, d, False, 180.0, 180.0, 1.0e5)
            ff.Minimize(maxIts=200)

        atoms = Atoms(positions=mol.GetConformer().GetPositions(),
                      symbols=[atom.GetSymbol() for atom in mol.GetAtoms()])

        return cls(
            atoms=atoms,
            mol=mol,
            smiles=smiles,
            charge=Chem.GetFormalCharge(mol),
            binding_motifs=binding_motifs,
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
        Write the ligand to file.

        Args:
            fmt (str): File format.
            filename (str): Output path.
            vacuum (float): Vacuum padding (Å) for VASP output.
        """
        if filename is None:
            filename = f"{self.name}.{fmt}"

        write_structure(self.atoms, filename, fmt, center_in_box=True, vacuum=vacuum, **kwargs)


@dataclass
class LigandSpec:
    """
    Pairs a Ligand with placement parameters.

    Attributes:
        ligand (Ligand): The ligand to attach.
        site (Optional[str]): Which sublattice to target, one of "A", "B", or "X". If None,
            the target is the motif's own ``site`` or, failing that, follows from the ligand
            charge (positive -> A-site, otherwise X-site). Ligands with several motifs take
            the site of each motif and leave this None.
        coverage (Optional[float]): Fractional coverage of available binding sites; above 1,
            a number of ligands. For a multi-motif ligand the fraction is of the most ligands
            the free sites can host, each sublattice serving its free sites divided by the
            motifs that need it.
        binding_site_indices (Optional[list[int]]): Atom indices of the surface sites to use
            (any atom of a molecular A-site). For a multi-motif ligand, a list of groups with
            one index per motif, in motif order.
        anchor_offset (float): Offset (Å) along the binding axis. When `adsorb` is
            True this sets the spacing between the surface species and the adsorbate.
        adsorb (bool): If True, adsorb the ligand on top of the surface site instead
            of replacing it (the surface atom is kept). Use `anchor_offset` to control
            the spacing.
    """
    ligand: Ligand
    site: Optional[str] = None
    coverage: Optional[float] = None
    binding_site_indices: Optional[list[int]] = None
    anchor_offset: float = 0.0
    adsorb: bool = False

    def __post_init__(self):
        if self.site is not None and self.site not in ("A", "B", "X"):
            raise ValueError(
                f"site must be one of 'A', 'B', 'X' (or None); got {self.site!r}"
            )
        motifs = self.ligand.binding_motifs
        if len(motifs) == 1 and motifs[0].site is not None:
            if self.site is None:
                self.site = motifs[0].site
            elif self.site != motifs[0].site:
                raise ValueError(
                    f"LigandSpec.site={self.site!r} contradicts the motif's site={motifs[0].site!r}"
                )
        elif len(motifs) > 1:
            if self.site is not None:
                raise ValueError(
                    "LigandSpec.site applies to single-motif ligands; give each BindingMotif "
                    "of a multi-motif ligand its own site instead"
                )
            missing = [k for k, m in enumerate(motifs) if m.site is None]
            if missing:
                raise ValueError(
                    f"every motif of a multi-motif ligand needs a site ('A', 'B' or 'X'); "
                    f"motif(s) {missing} of ligand {self.ligand.name!r} have none"
                )
        if self.coverage is None and self.binding_site_indices is None:
            raise ValueError("LigandSpec needs a coverage or binding_site_indices")
        if self.coverage is not None and self.coverage < 0:
            raise ValueError(f"coverage must be at least 0; got {self.coverage}")