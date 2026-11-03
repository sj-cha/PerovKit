# PerovKit

PerovKit provides modular, Pythonic data structures for constructing ligand-passivated ABX<sub>3</sub> perovskite models (nanocrystals, nanorods, and periodic slabs) for downstream atomistic simulations. 

Under active development by Seungjun Cha (scha@gatech.edu)

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/sj-cha/PerovKit.git
cd perovkit
pip install -e .
```

Dependencies: `numpy`, `scipy`, `ase`, `rdkit`, `tqdm`

## Key Concepts

- **Core**: The inorganic ABX<sub>3</sub> perovskite structure. Built as either a finite nanocrystal (`build_nanocrystal`) or a periodic slab (`build_slab`). `place_ligands` puts ligands on a copy of it, choosing the sites and turning the ligands until they clear each other; `place_ligands_manually` places them with fixed shifts and turns instead, optionally only those given by `ligand_ids`. Both return an `Assembly` and leave the core as it is.
- **Ligand**: An organic molecule that passivates surface sites. Created from SMILES (`from_smiles`) or an existing file (`from_file`). The `BindingMotif` specifies which atoms coordinate the surface. Cationic ligands take A-sites and anionic ligands take X-sites.
- **LigandSpec**: Pairs a `Ligand` with placement parameters: `coverage` (fraction or absolute count of surface sites), optional `binding_sites` (explicit site indices), and `anchor_offset` (controls a vertical offset from the surface).
- **Assembly**: A `Core` with its `Ligand`s, for a nanocrystal and a slab alike (a periodic `Core` gets minimum-image distances), as returned by `Core.place_ligands` or read with `Assembly.from_file`. `NanoCrystal` and `Slab` remain as other names for it.

## Quick Start

```python
from perovkit import Core, Ligand, LigandSpec, BindingMotif, Assembly
```

### Build a Nanocrystal with Ligands

```python
# 1. Build a 3x3x3 CsPbBr3 core
core = Core.build_nanocrystal(
    A="Cs", B="Pb", X="Br",
    a=5.95,
    supercell=(3,3,3),
    charge_neutral=True,
    random_seed=42,
)

# 2. Define ligands from SMILES or .xyz file
cationic_lig = Ligand.from_smiles(
    smiles="C[NH3+]",
    binding_motif=BindingMotif(["N"]),
    random_seed=42,
    name="MA",
)

anionic_lig = Ligand.from_file(
    filename="ligands/OP.xyz",
    binding_motif=BindingMotif(["O", "O"]),
    name="OP",
)

# 3. Specify coverage and assemble
specs = [
    LigandSpec(ligand=cationic_lig, coverage=0.5),
    LigandSpec(ligand=anionic_lig, coverage=0.5),
]

nc = core.place_ligands(specs)
```

### Build a Slab with Ligands

```python
# Build a 2x2x3 CsPbBr3 slab
core = Core.build_slab(
    A="Cs", B="Pb", X="Br",
    a=5.95,
    supercell=(2,2,3),
)

specs = [
    LigandSpec(ligand=cationic_lig, coverage=0.5),
    LigandSpec(ligand=anionic_lig, coverage=0.5),
]

slab = core.place_ligands(specs)
```

### Serialization

An `Assembly` can be exported as a JSON file, alongside the structure file. 

```python
# Save
nc.to(fmt="xyz", filename="structure.xyz", write_json=True)

# Load
nc = Assembly.from_file("structure.xyz", "structure.xyz.json")
```