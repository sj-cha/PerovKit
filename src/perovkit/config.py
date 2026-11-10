from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
import math
from typing import Tuple


@dataclass(frozen=True)
class PlacementConfig:
    """
    Settings of Core.place_ligands.
    """
    # contacts
    overlap_cutoff: float = 1.6                       # closest allowed ligand-ligand or ligand-core pair (Å)
    contact_margins: Tuple[float, ...] = (0.2, 0.05)  # margin over the cutoff that refinement aims for in order (Å)
    contact_reach: float = 0.5                        # distances below (overlap_cutoff + contact_reach) (Å) are measured exactly
    interior_depth: float = 0.5                       # allowed ligand penetration into the core (Å)

    # site selection
    site_selection: str = "farthest"                  # order in which sites are taken: "farthest" from the occupied ones, or "random"

    # objective of the refinements (torsion space, L-BFGS)
    refine_torsions: bool = True                      # refine torsions for ligands that cannot be resolved by rigid-body rotation (fitting and packing)
    k_contact: float = 40.0                           # penalty for pairs under the cutoff (per Å^2)
    k_contact_weights: Tuple[float, ...] = (1.0, 2.0, 4.0)  # multipliers applied to k_contact, tried in order upon failure
    k_anchor: float = 20.0                            # reward for each anchor's proximity to its ideal site (per Å^2)
    w_torsion: float = 0.02                           # reward for torsions back to the conformer, multiplied by (1 - cos)
    intra_scale: float = 0.7                          # limit for atoms 4+ bonds apart in a ligand, as a fraction of the vdW sum
    intra_weight: float = 10.0                        # multiplier applied to k_contact for such pairs under their limit
    interior_weight: float = 10.0                     # multiplier applied to k_contact for atoms penetrating the core
    max_anchor_rmsd: float = 0.5                      # max RMS anchor-site distance allowed per ligand (Å)
    refine_iters: int = 400                           # L-BFGS iterations per refinement

    # packing
    max_iters: int = 10                               # outer iteration of refinement rounds
    coarse_step_deg: int = 18                         # coarse rotation scan step
    fine_step_deg: int = 2                            # finer rotation scan step within [-window_deg, +window_deg] of the best coarse rotation
    window_deg: int = 12                              # half-width of the fine rotation scan (deg)
    active_radius_factor: float = math.sqrt(2) / 2    # radius of active neighbors in lattice constants

    # stretch
    stretch_tails: bool = True                        # stretch dangling chains along the surface normal before packing
    stretch_iters: int = 30                           # L-BFGS iterations per stretch before packing
    k_pull: float = 1.0                               # reward per each heavy tail atom's height along the surface normal (per Å)

    # multi-motif: conformer library
    n_conformers: int = 50                            # number of distinct ETKDG conformers embedded per ligand
    rmsd_cutoff: float = 0.5                          # RMSD cutoff required among conformers to be considered distinct (Å)

    # multi-motif: rigid fits
    n_constellations: int = 4                         # number of feasible constellations per seed site
    n_fits: int = 4                                   # number of conformer fits kept per given constellation
    max_enumerated: int = 64                          # number of constellations enumerated per seed site
    n_spin: int = 36                                  # number of spins scanned about a free axis to clear the core and placed ligands
    fit_n_spin: int = 72                              # number of spins scanned about a free axis for the fit cost
    # fit cost: w_align*(1 - align) + w_foot*foot^2 + w_rmsd*rmsd^2 + w_energy*E
    w_align: float = 2.0                              # penalty for motifs misaligned with their site normals
    w_foot: float = 0.25                              # penalty for the body's spread across the normal (per Å^2)
    w_rmsd: float = 16.0                              # penalty for anchor RMSD to the sites (per Å^2)
    w_energy: float = 0.2                             # penalty for the conformer's UFF energy (per kcal/mol)
    w_crowd: float = 1.0                              # penalty for pairs short of the cutoff to placed ligands (per Å^2)

    # multi-motif: rescue
    near_rmsd_factor: float = 2.0                     # multiplier applied to max_anchor_rmsd for fits still worth refining
    n_refine: int = 8                                 # number of conformers refined before a seed is given up
    fit_stretch_iters: int = 200                      # L-BFGS iterations per stretch in multi-motif fitting
    max_templates: int = 16                           # maximum rescued poses kept per ligand type as templates

    # multi-motif: embedding
    embed_constellations: int = 2                     # number of constellations tried per seed when no conformer fits
    embed_conformers: int = 8                         # number of conformers embedded at each constellation
    max_embeddings: int = 12                          # maximum embeddings per placement

    # multi-motif: covered sites
    cover_margin: float = 1.0                         # free sites within (overlap_cutoff + cover_margin) (Å) of multi-motif ligands are excluded

    # polymerization
    polymerize: bool = True                           # bond neighboring reactive groups into chains
    bond_reach: float = 12.0                          # farthest reactive carbons distance (Å)
    near_reach: float = 8.0                           # distance within which reactive carbons are bonded first (Å)
    k_link: float = 160.0                             # reward for pulling reactive carbons together (per Å^2)
    link_tolerance: float = 0.1                       # max distance allowed between a bond point and its partner carbon to keep the bond (Å)
    link_stop: float = 0.02                           # bond point-partner distance required to stop the refinement early (Å)
    whole_units: int = 20                             # max number of units in a network refined as a whole
    hops: int = 2                                     # number of bonds from newly bonded units within which units are refined in larger networks

    def __post_init__(self):
        if self.site_selection not in ("farthest", "random"):
            raise ValueError(f"site_selection must be 'farthest' or 'random'; got {self.site_selection!r}")

    def to_json(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filename: str) -> PlacementConfig:
        """Fields missing from the file keep their defaults."""
        with open(filename) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, values: dict) -> PlacementConfig:
        tuples = {f.name for f in fields(cls) if str(f.type).startswith("Tuple")}
        return cls(**{k: tuple(v) if k in tuples else v for k, v in values.items()})
