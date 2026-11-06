# PerovKit

PerovKit provides modular, Pythonic data structures for constructing ligand-passivated ABX<sub>3</sub> perovskite models for downstream atomistic simulations. 

Under active development by Seungjun Cha (scha@gatech.edu)

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/sj-cha/PerovKit.git
cd perovkit
pip install -e .
```

## Supported systems

* Cubic ABX<sub>3</sub> systems; organic or inorganic A-site
* Monodentate, bridging, and multidentate ligands (including zwitterionic, multipodal, etc.) with specified surface coverage
* Nanocrystals, nanorods, and periodic slabs

## Quick Start

```python
from perovkit import Core, Ligand, BindingMotif, Assembly
```

### Build a Nanocrystal with Co-passivation

```python
# Build a 5x5x5 CsPbBr3 nanocrystal
core = Core.build_core(A="Cs", B="Pb", X="Br", a=5.95, supercell=(5, 5, 5), pbc=False)

# Oleylammonium
OAm = Ligand.from_smiles(
    smiles=r"CCCCCCCC/C=C\CCCCCCCC[NH3+]",
    binding_motif=BindingMotif(atoms=["N"], anchor="[NH3+]", site="A"),
    name="OAm",
)

# Oleate
OA = Ligand.from_smiles(
    smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)[O-]",
    binding_motif=BindingMotif(atoms=["O", "O"], anchor="C(=O)[O-]", site="X"),
    name="OA",
)

# Place ligands substitutionally. Returns `Assembly`
nc = core.place_ligands(ligands=[OAm, OA], coverage=[0.5, 0.5])
nc.to(fmt="xyz", filename="nc_OA_OAm.xyz")
```

### Build a Slab with Multidentate Ligands

```python
# A-site foraminidium
FA = Ligand.from_smiles(
    smiles="NC=[NH2+]", 
    name="FA",
)

# PO-T2T with three binding anchors that adsorbes on FA
POT2T = Ligand.from_smiles(
    smiles="O=P(c1ccccc1)(c1ccccc1)c1cccc(-c2nc(-c3cccc(P(=O)(c4ccccc4)c4ccccc4)c3)nc(-c3cccc(P(=O)(c4ccccc4)c4ccccc4)c3)n2)c1", 
    binding_motif=[BindingMotif(atoms=["O"], anchor="P(=O)", site="A", adsorb = True),
                    BindingMotif(atoms=["O"], anchor="P(=O)", site="A", adsorb = True),
                    BindingMotif(atoms=["O"], anchor="P(=O)", site="A", adsorb = True)],
    name="POT2T",
)

# Build a 4x4x3 FAPbI3 slab
core = Core.build_core(
    A=FA, B="Pb", X="Br",
    a=6.3691,
    supercell=(4,4,3),
    pbc=True
)


# POT2T adsorbs above the surface FA
slab = core.place_ligands(ligands=POT2T, coverage=0.25)
slab.to(fmt="vasp", filename="slab_POT2T.vasp")
```

### Serialization

An `Assembly` can be exported as a JSON file, alongside the structure file. 

```python
nc = Assembly.from_file("structure.xyz", "structure.xyz.json")
```

### License
MIT license. Hu Group, Georgia Institute of Technology