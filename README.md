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

- **Core**: The inorganic ABX<sub>3</sub> perovskite structure. Built as either a finite nanocrystal (`build_nanocrystal`) or a periodic slab (`build_slab`).
- **Ligand**: An organic molecule that passivates surface sites. Created from SMILES (`from_smiles`) or an existing file (`from_file`). The `BindingMotif` specifies which atoms coordinate the surface. Cationic ligands take A-sites and anionic ligands take X-sites.
- **LigandSpec**: Pairs a `Ligand` with placement parameters: `coverage` (fraction or absolute count of surface sites), optional `binding_sites` (explicit site indices), and `anchor_offset` (controls a vertical offset from the surface).
- **NanoCrystal / Slab**: Combines a `Core` with `Ligand`. Handles site selection and ligand placement by rotation optimization.

## Quick Start

```python
from perovkit import Core, Ligand, LigandSpec, BindingMotif, NanoCrystal, Slab
```

### Build a `Nanocrystal` with Ligands

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

nc = NanoCrystal(core=core, ligand_specs=specs)
nc.place_ligands()
```

### Build a `Slab` with Ligands

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

slab = Slab(core=core, ligand_specs=specs)
slab.place_ligands()
```

### Serialization

Both `NanoCrystal` and `Slab` can be exported as a JSON file, alongside the structure file. 

```python
# Save
nc.to(fmt="xyz", filename="structure.xyz", write_json=True)

# Load
nc = NanoCrystal.from_file("structure.xyz", "structure.xyz.json")
```