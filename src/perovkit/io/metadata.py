from __future__ import annotations
from dataclasses import asdict
import json
from typing import Dict, List, Optional

from ase import Atoms
from ase.data import atomic_numbers
import numpy as np

from ..core import Core
from ..ligand import BindingMotif, Ligand, pack_motifs


def motif_from_metadata(meta: dict) -> Optional[BindingMotif | List[BindingMotif]]:
    """
    Rebuild from serialized ligand metadata, in the form ``Ligand.binding_motif`` takes:
    one motif, a list of several, or None when the ligand has none.
    """
    if meta.get("binding_motifs") is not None:
        return pack_motifs([BindingMotif(**m) for m in meta["binding_motifs"]])
    # older files hold their one motif in flat keys
    atoms = meta.get("binding_motif_atoms")
    if not atoms:
        return None
    anchor = meta.get("binding_motif_anchor")
    indices = meta.get("binding_motif_indices")
    if anchor is None and indices is None:
        # written before anchors existed: pin the binding atoms resolved back then
        indices = meta.get("binding_atoms_indices")
    return BindingMotif(atoms, anchor=anchor, indices=indices)


def ligand_type_key(lig: Ligand) -> tuple:
    """
    Everything that makes two placed ligands the same JSON ligand type. The metadata
    holds one set of motifs and binding atoms per type, so the same molecule anchored
    through different atoms has to make two types.
    """
    motifs = tuple((tuple(m.atoms), m.anchor, tuple(m.indices) if m.indices else None, m.site)
                   for m in lig.binding_motifs)
    return lig.name, lig.smiles, lig.charge, motifs, tuple(lig.binding_atoms or ())


def ligand_type_metadata(lig: Ligand) -> dict:
    """
    Per-type JSON metadata: identity, binding motifs and resolved binding atoms.
    """
    return {
        "name": lig.name,
        "smiles": lig.smiles,
        "charge": lig.charge,
        "binding_atoms_indices": list(lig.binding_atoms or []),
        "binding_motifs": [asdict(m) for m in lig.binding_motifs],
    }


def ligand_instance_metadata(lig: Ligand) -> dict:
    """
    Per-instance plane metadata; ``planes`` is written only when there is more than one.
    """
    meta = {"plane": list(lig.plane)}
    if len(lig.planes) > 1:
        meta["planes"] = [list(p) for p in lig.planes]
    return meta


def a_site_metadata(core: Core) -> str | dict:
    """
    Serialize the A-site for JSON.

    Returns the element symbol for an inorganic A-site, or a dict carrying the
    molecular cation's Ligand information (one type, repeated `n_instances`
    times) for an organic A-site. Inverse of `a_site_from_metadata`.
    """
    if not isinstance(core.A, list):
        return core.A

    lig = core.A[0]
    return {
        "label": core.A_label,
        "kind": "molecular",
        "n_instances": len(core.A),
        "ligand": {
            **ligand_type_metadata(lig),
            "n_atoms": len(lig.atoms),
            "volume": float(lig.volume) if lig.volume is not None else None,
        },
    }


def a_site_from_metadata(meta: str | dict, atoms: Atoms, B: str, X: str):
    """
    Reconstruct the A-site value from `a_site_metadata`.

    For an inorganic A-site returns the element symbol. For an organic A-site returns a
    list of placed Ligand instances and sets the "a_site_id" array on `atoms`: the
    cations are the atoms that are neither B nor X, one molecule after another.
    """
    if not isinstance(meta, dict):
        return str(meta)

    lmeta = meta["ligand"]
    n_atoms = int(lmeta["n_atoms"])

    cation_atoms = np.flatnonzero(~np.isin(atoms.numbers, [atomic_numbers[B], atomic_numbers[X]]))
    ids = np.full(len(atoms), -1, dtype=int)
    ids[cation_atoms] = np.arange(len(cation_atoms)) // n_atoms
    atoms.set_array("a_site_id", ids)

    template = Ligand._from_data(
        atoms=atoms[cation_atoms[:n_atoms]],
        mol=None,
        smiles=lmeta["smiles"],
        charge=lmeta["charge"],
        binding_motif=motif_from_metadata(lmeta),
        name=lmeta["name"],
        volume=lmeta.get("volume"),
        binding_atoms=list(map(int, lmeta.get("binding_atoms_indices", []))),
    )
    return Core._build_A_ligands(atoms, template)


def write_metadata(assembly, filename: str, sort: bool = False) -> None:
    """
    Write the metadata of `assembly` to `filename`; with `sort`, the order of its atoms in a
    file sorted by element too.
    """
    core_meta = {
        "A": a_site_metadata(assembly.core),
        "B": assembly.core.B,
        "X": assembly.core.X,
        "a": assembly.core.a,
        "supercell": list(assembly.core.supercell),
    }
    if assembly.core.is_slab:
        core_meta["cell"] = assembly.core.atoms.get_cell().array.tolist()
        core_meta["pbc"] = [bool(x) for x in assembly.core.atoms.pbc]
        core_meta["vacuum"] = float(assembly.core.vacuum)
    core_meta["n_core_atoms"] = len(assembly.core.atoms)

    ligand_types_meta = []
    type_key_to_id: Dict[tuple, int] = {}
    ligands_meta = []

    for i, lig in enumerate(assembly.ligands):
        key = ligand_type_key(lig)

        if key not in type_key_to_id:
            type_key_to_id[key] = len(ligand_types_meta)
            ligand_types_meta.append(
                {
                    "id": type_key_to_id[key],
                    **ligand_type_metadata(lig),
                    "coverage": assembly.ligand_coverage.get(lig.name),
                    "n_atoms": len(lig.atoms),
                    "volume": float(lig.volume),
                    "n_instances": 0,
                }
            )

        spec_id = type_key_to_id[key]
        ligand_types_meta[spec_id]["n_instances"] += 1
        ligands_meta.append({"ligand_id": i, "spec_id": spec_id, **ligand_instance_metadata(lig)})

    topo = {
        "schema_version": 1,
        "type": "slab" if assembly.core.is_slab else "nanocrystal",
        "n_total_atoms": len(assembly.atoms),
        "core": core_meta,
        "ligand_types": ligand_types_meta,
        "core_indices": {"octahedra": assembly.octahedra, "B_ijk": assembly.B_ijk},
        "ligands": ligands_meta,
    }

    if sort:
        topo["sort_idx"] = np.argsort(assembly.atoms.symbols).tolist()

    with open(filename, "w") as f:
        json.dump(topo, f, indent=2)


def read_assemblies(cls, atoms: Atoms | List[Atoms], json_filename: str):
    """
    Assemblies of class `cls` from the structure `atoms`, or a list of them for a list of
    frames, and the metadata in `json_filename`.
    """
    with open(json_filename) as f:
        topo = json.load(f)

    schema_version = topo.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {schema_version!r}")

    file_type = topo.get("type")
    if file_type not in (None, "nanocrystal", "slab"):
        raise ValueError(
            f"JSON type is '{file_type}', expected 'nanocrystal' or 'slab'."
        )

    single_input = isinstance(atoms, Atoms)
    atoms_list: List[Atoms] = [atoms] if single_input else list(atoms)

    sort_idx = topo.get("sort_idx", None)
    n_total_atoms_json = topo["n_total_atoms"]

    core_meta = topo["core"]
    n_core_atoms = core_meta["n_core_atoms"]
    periodic = "cell" in core_meta
    if periodic:
        core_cell = np.array(core_meta["cell"], dtype=float)
        core_pbc = np.array(core_meta["pbc"], dtype=bool)

    core_indices_meta = topo["core_indices"]

    octa_raw = core_indices_meta.get("octahedra", None)
    if octa_raw is not None:
        octahedra = {
            int(b): {"X": v["X"], "Ligand": v["Ligand"]}
            for b, v in octa_raw.items()
        }
    else:
        octahedra = None

    B_ijk_raw = core_indices_meta.get("B_ijk")
    B_ijk = {int(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in B_ijk_raw.items()}

    ligand_types_meta = topo["ligand_types"]
    type_id_to_meta: Dict[int, dict] = {
        t["id"]: t for t in ligand_types_meta
    }
    ligands_meta = topo["ligands"]

    # Everything that is the same for every frame is resolved once, outside the loop.
    order = np.argsort(sort_idx) if sort_idx is not None else None
    motifs = {t["id"]: motif_from_metadata(t) for t in ligand_types_meta}

    lig_slices: List[slice] = []
    cursor = n_core_atoms
    for inst_meta in ligands_meta:
        n_atoms = type_id_to_meta[inst_meta["spec_id"]]["n_atoms"]
        lig_slices.append(slice(cursor, cursor + n_atoms))
        cursor += n_atoms

    assemblies: List[Assembly] = []
    for atoms_i in atoms_list:
        if n_total_atoms_json != len(atoms_i):
            raise ValueError(
                f"Atom count mismatch: JSON={n_total_atoms_json}, XYZ={len(atoms_i)}."
            )

        # split plain arrays: slicing atoms_i per part dominated the per-frame cost
        numbers = atoms_i.numbers
        positions = atoms_i.positions
        if order is not None:
            numbers = numbers[order]
            positions = positions[order]

        core_atoms = Atoms(
            numbers=numbers[:n_core_atoms],
            positions=positions[:n_core_atoms],
            pbc=core_pbc if periodic else False,
            cell=core_cell if periodic else atoms_i.cell,
        )

        core = Core(
            A=a_site_from_metadata(core_meta["A"], core_atoms, core_meta["B"], core_meta["X"]),
            B=core_meta["B"],
            X=core_meta["X"],
            atoms=core_atoms,
            a=core_meta["a"],
            supercell=tuple(core_meta["supercell"]),
            vacuum=core_meta.get("vacuum"),
            build_surface=False,
        )

        ligands: List[Ligand] = []

        for inst_meta, sl in zip(ligands_meta, lig_slices):
            spec_id = inst_meta["spec_id"]
            tmeta = type_id_to_meta[spec_id]

            lig_atoms = Atoms(
                numbers=numbers[sl],
                positions=positions[sl],
                pbc=atoms_i.pbc,
                cell=atoms_i.cell,
            )

            binding_atoms = list(map(int, tmeta.get("binding_atoms_indices", [])))

            lig = Ligand._from_data(
                atoms=lig_atoms,
                mol=None,
                smiles=tmeta["smiles"],
                charge=tmeta["charge"],
                binding_motif=motifs[spec_id],
                name=tmeta["name"],
                id=inst_meta["ligand_id"],
                planes=inst_meta.get("planes") or [inst_meta["plane"]],
                volume=tmeta["volume"],
                binding_atoms=binding_atoms,
            )

            ligands.append(lig)

        core.octahedra = octahedra
        core.B_ijk = B_ijk
        assemblies.append(cls(core=core, ligands=ligands,
                              ligand_coverage={t["name"]: t["coverage"] for t in ligand_types_meta}))

    return assemblies[0] if single_input else assemblies
