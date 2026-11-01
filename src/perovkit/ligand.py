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

from .conformers import ConformerLibrary
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v


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

    def _key(self) -> tuple:
        return (tuple(self.atoms), self.anchor,
                tuple(self.indices) if self.indices else None, self.site)

    def _to_metadata(self) -> dict:
        return {
            "atoms": list(self.atoms),
            "anchor": self.anchor,
            "indices": list(self.indices) if self.indices else None,
            "site": self.site,
        }

    @classmethod
    def _from_metadata(cls, meta: dict) -> Optional[BindingMotif | List[BindingMotif]]:
        """
        Rebuild from serialized ligand metadata, in the form ``Ligand.binding_motif`` takes:
        one motif, a list of several, or None when the ligand has none.
        """
        if meta.get("binding_motifs") is not None:
            return _pack_motifs([cls(**m) for m in meta["binding_motifs"]])
        atoms = meta.get("binding_motif_atoms")
        if not atoms:
            return None
        anchor = meta.get("binding_motif_anchor")
        indices = meta.get("binding_motif_indices")
        if anchor is None and indices is None:
            # written before anchors existed: pin the binding atoms resolved back then
            indices = meta.get("binding_atoms_indices")
        return cls(atoms, anchor=anchor, indices=indices)


def _pack_motifs(value) -> Optional[BindingMotif | List[BindingMotif]]:
    """
    Normalize a binding_motif value: a single motif stays as it is (also when given as a
    one-element list), several become a list, and an empty one becomes None.
    """
    if value is None or isinstance(value, BindingMotif):
        return value
    motifs = list(value)
    if not all(isinstance(m, BindingMotif) for m in motifs):
        raise TypeError("binding_motif must be a BindingMotif or a list of them")
    if not motifs:
        return None
    return motifs[0] if len(motifs) == 1 else motifs


class _SyncedMol:
    """
    Descriptor for Ligand.mol: on every access the conformer is overwritten with the
    current ASE positions, so it follows orientation, placement and any later edit
    made through the parent structure's shared position buffer.
    """
    def __get__(self, lig, owner=None):
        if lig is None:
            raise AttributeError  # tells dataclass the field has no default
        mol = lig.__dict__.get("_mol")
        # a size mismatch is left for the __post_init__ check to report
        if mol is not None and mol.GetNumConformers() and mol.GetNumAtoms() == len(lig.atoms):
            mol.GetConformer().SetPositions(lig.atoms.get_positions())
        return mol

    def __set__(self, lig, mol):
        lig.__dict__["_mol"] = mol


@dataclass
class Ligand:
    """
    Surface-passivating ligand molecule.

    Attributes:
        atoms (ASE Atoms): ASE Atoms object.
        mol (Chem.Mol): RDKit Mol object; its conformer follows ``atoms`` positions.
        smiles (str): SMILES string of the ligand without explicit hydrogens.
        charge (int): Formal charge of the ligand.
        binding_motif (BindingMotif | List[BindingMotif]): How the ligand anchors to a
            surface: one motif, or a list with one motif per binding group.
        name (str): User-defined identifier.
        id (int): Ligand ID assigned after placement.
        volume (float): Molecular volume (Å^3).
        binding_atoms (List[int]): Local indices of atoms used as anchors, motif after motif
            (see ``binding_groups`` for the split).
        planes (List[Tuple[int, int, int]]): Miller index of the surface each motif is bound
            to; ``plane`` is the first one.
        indices (np.ndarray): Global atom indices when placed in a parent NanoCrystal/Slab.
        _anchor_offset (float): Offset (Å) applied along the axis perpendicular to the plane.
    """
    atoms: Atoms
    mol: Chem.Mol = _SyncedMol()
    smiles: str
    charge: int
    binding_motif: Optional[BindingMotif | List[BindingMotif]]
    name: str
    id: Optional[int] = None
    volume: float = None
    binding_atoms: List[int] = field(default_factory=list)
    planes: List[Tuple[int, int, int]] = field(default_factory=list)
    indices: Optional[np.ndarray] = None
    _anchor_offset: float = 0.0

    def __post_init__(self):
        mol = self.mol
        if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != self.atoms.get_chemical_symbols():
            raise ValueError("mol and atoms must list the same elements in the same order")
        self.binding_motif = _pack_motifs(self.binding_motif)
        self._get_volume()
        if self.binding_motif:
            self._get_binding_atoms_indices()
            self._orient_ligand()


    @property
    def binding_motifs(self) -> List[BindingMotif]:
        """
        The binding motifs as a list, whether the ligand has none, one or several.
        """
        m = self.binding_motif
        if m is None:
            return []
        return [m] if isinstance(m, BindingMotif) else list(m)

    @property
    def n_motifs(self) -> int:
        return len(self.binding_motifs)

    @property
    def binding_groups(self) -> List[List[int]]:
        """
        ``binding_atoms`` split per motif: each motif owns as many atoms as it lists elements.
        """
        motifs = self.binding_motifs
        if not motifs:
            return [list(self.binding_atoms)] if self.binding_atoms else []
        groups, start = [], 0
        for m in motifs:
            groups.append(list(self.binding_atoms[start:start + len(m.atoms)]))
            start += len(m.atoms)
        return groups

    @property
    def anchor_pos(self) -> Optional[np.ndarray]:
        if not self.binding_atoms:
            return None
        pos = self.atoms.get_positions()
        return pos[np.asarray(self.binding_atoms, dtype=int)].mean(axis=0)

    @property
    def anchor_positions(self) -> Optional[np.ndarray]:
        """
        Anchor of every motif, the centroid of its binding atoms: (n_motifs, 3).
        """
        if not self.binding_atoms:
            return None
        pos = self.atoms.get_positions()
        return np.array([pos[np.asarray(g, dtype=int)].mean(axis=0) for g in self.binding_groups])

    @property
    def plane(self) -> Optional[Tuple[int, int, int]]:
        return self.planes[0] if self.planes else None

    @plane.setter
    def plane(self, value) -> None:
        self.planes = [] if value is None else [value]


    @classmethod
    def from_file(
        cls,
        filename: str,
        binding_motif: BindingMotif | List[BindingMotif],
        name: str,
        charge: Optional[int] = None,
        **kwargs
    ) -> Ligand:
        """
        Build a Ligand from a file. 
        Currently only XYZ files are supported.

        Args:
            filename (str): Path to the XYZ file.
            binding_motif (BindingMotif | List[BindingMotif]): How the ligand anchors to a
                surface; a list gives one motif per binding group.
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
        binding_motif: BindingMotif | List[BindingMotif] = None,
        random_seed: int = 42,
        name: str = None,
        optimize: bool = False,
        **kwargs
    ) -> Ligand:
        """
        Build a Ligand from a SMILES string.

        Args:
            smiles (str): SMILES string of the ligand.
            binding_motif (BindingMotif | List[BindingMotif]): How the ligand anchors to a
                surface; a list gives one motif per binding group.
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


    def clone(self) -> Ligand:
        """
        Return a shallow copy with independent atoms, mol and indices.
        """
        lig = copy.copy(self)
        lig.atoms = self.atoms.copy()
        if self.mol is not None:
            lig.mol = Chem.Mol(self.mol)  # own conformer, so it can follow its own atoms
        if self.indices is not None:
            lig.indices = self.indices.copy()
        lig.planes = list(self.planes)
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
        lig.planes = []
        lig.indices = None
        lig._anchor_offset = 0.0
        for k, v in kwargs.items():
            setattr(lig, k, v)
        lig.binding_motif = _pack_motifs(getattr(lig, "binding_motif", None))
        return lig


    def _type_key(self) -> tuple:
        """
        Everything that makes two placed ligands the same JSON ligand type.
        """
        # The motifs and resolved binding atoms are part of the TYPE: the per-type metadata
        # stores one set of them for every instance sharing this key. Without them, two
        # placements of the same molecule anchored through DIFFERENT atoms collide into one
        # type and both reload with the first one's anchor, silently. `from_smiles` derives
        # charge from the SMILES and defaults name to None, so that collision needs no opt-in.
        return (
            self.name,
            self.smiles,
            self.charge,
            tuple(m._key() for m in self.binding_motifs),
            tuple(self.binding_atoms or ()),
        )


    def _type_metadata(self) -> dict:
        """
        Per-type JSON metadata: identity, binding motifs and resolved binding atoms.

        ``binding_motifs`` is what this version reads back; the flat ``binding_motif_*``
        keys are kept for single-motif ligands so that older versions can still load them.
        """
        motifs = self.binding_motifs
        legacy = motifs[0] if len(motifs) == 1 else None
        meta = {
            "name": self.name,
            "smiles": self.smiles,
            "charge": self.charge,
        }
        if len(motifs) <= 1:
            meta.update({
                "binding_motif_atoms": list(legacy.atoms) if legacy else None,
                "binding_motif_anchor": legacy.anchor if legacy else None,
                "binding_motif_indices": (list(legacy.indices)
                                          if legacy and legacy.indices else None),
            })
        meta["binding_atoms_indices"] = list(self.binding_atoms or [])
        meta["binding_motifs"] = [m._to_metadata() for m in motifs]
        return meta


    def _instance_metadata(self) -> dict:
        """
        Per-instance plane metadata; ``planes`` is written only when there is more than one.
        """
        meta = {"plane": list(self.plane)}
        if len(self.planes) > 1:
            meta["planes"] = [list(p) for p in self.planes]
        return meta


    @staticmethod
    def _planes_from_metadata(inst_meta: dict) -> list:
        return inst_meta.get("planes") or [inst_meta["plane"]]


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


    def _anchor_region(self, motif: BindingMotif, claimed: set = frozenset()) -> List[int]:
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

        first = set(matches[0])
        # matches sharing atoms with the first one are the same group seen from another atom
        if self.n_motifs == 1 and any(first.isdisjoint(m) for m in matches[1:]):
            print(f"[Warning] anchor {anchor!r} matches more than one group in ligand "
                  f"{self.name or self.smiles}; using the first match {sorted(first)}.")

        region = set(first)
        for i in first:
            region.update(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors())
        return sorted(region - claimed)


    def _get_binding_atoms_indices(self) -> List[int]:
        """
        Detect the atom indices that match the binding motifs, motif after motif.
        """
        binding_atoms: List[int] = []
        for motif in self.binding_motifs:
            binding_atoms += self._resolve_motif(motif, set(binding_atoms))
        self.binding_atoms = binding_atoms

        # One motif warns in _anchor_region; several sharing a SMARTS are checked together
        if self.n_motifs > 1:
            anchors = [m.anchor for m in self.binding_motifs if m.indices is None]
            for anchor in dict.fromkeys(anchors):
                groups: List[set] = []
                for match in self.mol.GetSubstructMatches(Chem.MolFromSmarts(anchor)):
                    if all(g.isdisjoint(match) for g in groups):
                        groups.append(set(match))
                if len(groups) > anchors.count(anchor):
                    print(f"[Warning] anchor {anchor!r} matches {len(groups)} groups in ligand "
                          f"{self.name or self.smiles}, but only {anchors.count(anchor)} "
                          f"motif(s) use it; the other groups stay unbound.")
        return self.binding_atoms


    def _resolve_motif(self, motif: BindingMotif, claimed: set) -> List[int]:
        """
        Binding atom indices of one motif, never reusing `claimed` atoms.
        """
        symbols = self.atoms.get_chemical_symbols()
        coords  = self.atoms.get_positions()

        binding_elems = list(motif.atoms)
        explicit = motif.indices

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
            if not claimed.isdisjoint(idx):
                raise ValueError(f"binding motif indices {idx} reuse atoms of another motif")
            for i, elem in zip(idx, binding_elems):
                if symbols[i] != elem:
                    raise ValueError(
                        f"binding motif index {i} is {symbols[i]!r}, but the motif declares "
                        f"{elem!r}; the indices and the element symbols disagree"
                    )
            return idx

        region = self._anchor_region(motif, claimed)
        cn = self._coordination_numbers()

        if len(binding_elems) == 2:
            elem1, elem2 = binding_elems

            idx1 = [i for i in region if symbols[i] == elem1]
            idx2 = [i for i in region if symbols[i] == elem2]

            if len(idx1) == 0 or len(idx2) == 0:
                raise ValueError(f"No atoms found for binding motif {binding_elems} "
                                 f"around anchor {motif.anchor!r}")

            best_pair = None
            best_key = None

            for i in idx1:
                p1 = coords[i]
                for j in idx2:
                    if i == j:
                        continue
                    p2 = coords[j]
                    dist = float(np.linalg.norm(p1 - p2))
                    key = (int(cn[i]) + int(cn[j]), dist)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pair = (i, j)

            if best_pair is None:
                raise ValueError(
                    f"No distinct atom pair found for binding motif {binding_elems}"
                )

            return list(best_pair)

        elif len(binding_elems) == 1:
            elem = binding_elems[0]

            elem_indices = [i for i in region if symbols[i] == elem]
            if len(elem_indices) == 0:
                raise ValueError(f"No atoms found for binding element {elem!r} "
                                 f"around anchor {motif.anchor!r}")

            elem_cn = cn[np.asarray(elem_indices, dtype=int)]

            min_cn = int(np.min(elem_cn))
            candidate_local = np.where(elem_cn == min_cn)[0]

            # choose the first one among candidates
            chosen_global = elem_indices[candidate_local[0]]

            return [chosen_global]

        raise NotImplementedError(
            "Binding motifs with more than 2 atoms are not yet supported."
        )


    def _orient_ligand(self, n_angles: int = 720):
        """
        Orient the ligand so the binding axis is canonical and the body points up.

        A ligand with several motifs is only centered on the mean of its anchors: its
        orientation is set at placement, by fitting every anchor to its own site.
        """
        coords = self.atoms.get_positions()
        if self.n_motifs > 1:
            self.atoms.set_positions(coords - self.anchor_positions.mean(axis=0))
            return None

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
            the ligand charge. If None, the target is taken from the motif's own ``site``
            or, failing that, inferred from the ligand charge (positive -> A-site,
            otherwise X-site) for backward compatibility. Ligands with several motifs
            take the site of each motif and leave this None.
        coverage (Optional[float]): Fractional coverage of available binding sites; above 1,
            a number of ligands. For a multi-motif ligand the fraction is of the most ligands
            the free sites can host, each site type serving its free sites divided by the
            motifs that need it.
        binding_sites (Optional[list[int]]): Explicit surface site indices to use. For a
            multi-motif ligand, a list of groups with one site index per motif, in motif order.
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
        if self.name is None:
            self.name = self.ligand.name