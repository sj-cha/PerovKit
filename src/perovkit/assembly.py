from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple, Sequence
import json
from pathlib import Path

import random
import math
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.io.vasp import write_vasp

from scipy.spatial import cKDTree 

from .core import Core, BindingSite
from .ligand import Ligand, LigandSpec, BindingMotif
from .constellation import ConstellationSampler
from .refine import INTRA_SCALE
from .placement import (SitePose, FitPose, Contacts, pack, place_about_normal, motif_pools,
                        select_constellations, intramolecular_contacts)
from .utils.geometry import farthest_point_sampling, per_face_farthest_point_sampling, build_neighbor_map


@dataclass
class Assembly:
    """
    A Core with ligands on its surface: a nanocrystal, or a slab when the Core is periodic
    (Core.build_core(..., pbc=(True, True, False))). KD-tree searches follow the core's pbc:
    minimum-image distances along its periodic axes. A slab keeps its cell.
    Made by Core.place_ligands and Core.place_ligands_manually, or read back with
    from_file and from_ase.
    """
    core: Core
    ligand_specs: List[LigandSpec] = field(default_factory=list)
    ligands: List[Ligand] = field(default_factory=list)
    ligand_coverage: Dict[str, float] = field(default_factory=dict)

    @property
    def octahedra(self):
        return self.core.octahedra

    @property
    def B_ijk(self):
        return self.core.B_ijk

    def __post_init__(self):
        # Multi-motif ligands need site combinations, so they choose first
        self.ligand_specs.sort(
            key=lambda spec: (
                0 if spec.ligand.n_motifs > 1 else 1,
                0 if spec.ligand.charge > 0 else 1,
                -spec.ligand.volume,
            )
        )
        self._link_atoms()


    def _symbol_map(self) -> Dict[str, str]:
        return {
            "A": self.core.A_label,
            "B": self.core.B,
            "X": self.core.X,
        }


    def _take_sites(
        self,
        spec: LigandSpec,
        rng: random.Random,
        displaced: List[int],
        log: Callable,
    ) -> List[Tuple[Ligand, BindingSite]]:
        """
        Take free sites for a single-motif spec, those it names or else as many as its
        coverage asks for, spread by farthest-point sampling. The site type is `spec.site`,
        or else A for a cation and X otherwise. Returns a copy of the ligand for each site;
        the surface atoms the ligands replace are added to `displaced`.
        """
        lig = spec.ligand
        kind = spec.site if spec.site is not None else ("A" if lig.charge > 0 else "X")
        symbol = self._symbol_map()[kind]
        available = [s for s in self.core.binding_sites if s.symbol == symbol and not s.passivated]

        if spec.binding_sites is not None:
            chosen_sites = []
            for idx in spec.binding_sites:
                site = next((s for s in available if idx in s.atom_indices), None)
                if site is None:
                    print(f"[Warning] Requested binding site index {idx} not available for ligand {lig.name}.")
                elif site not in chosen_sites:
                    chosen_sites.append(site)
        else:
            if 0.0 < spec.coverage <= 1.0:
                n_target = int(math.ceil(spec.coverage * len(available)))
            else:
                n_target = int(spec.coverage)
                if n_target > len(available):
                    print(f"[Warning] Requested {n_target} sites for ligand, "
                          f"but only {len(available)} available.")
                    n_target = len(available)

            coords = np.array([s.position for s in available])
            sc = self.core.supercell
            if self.core.is_nanocrystal and sc is not None and len(set(sc)) > 1:
                planes = [s.plane for s in available]
                chosen_indices = per_face_farthest_point_sampling(coords, planes, n_target, rng)
            else:
                chosen_indices = farthest_point_sampling(coords, n_target, rng)
            chosen_sites = [available[i] for i in chosen_indices]

        taken = []
        for site in chosen_sites:
            clone = lig.clone()
            clone.plane = site.plane
            clone._anchor_offset = float(spec.anchor_offset)
            site.passivated = True
            taken.append((clone, site))
            # Replace mode removes the surface atom(s); adsorb mode keeps them.
            if not spec.adsorb:
                displaced.extend(site.atom_indices)

        mode = "Adsorbed" if spec.adsorb else "Placed"
        log(f"[Log] {mode} total {len(chosen_sites)} {kind}-site ligands.")
        self.ligand_coverage[lig.name] = spec.coverage
        return taken


    def _finish_placement(
        self,
        ligands: List[Ligand],
        coords_list: List[np.ndarray],
        displaced: List[int],
    ) -> None:
        """
        Give the ligands their coordinates and ids, and rebuild the core without the
        `displaced` atoms.
        """
        max_z = 0
        for i, (lig, coords) in enumerate(zip(ligands, coords_list)):
            lig.atoms.set_positions(coords)
            lig.id = i
            max_z = max(max_z, coords[:, 2].max())
        self.ligands = list(ligands)

        old = self.core
        keep = np.ones(len(old.atoms), dtype=bool)
        keep[displaced] = False
        stripped_atoms = Atoms(
            numbers=old.atoms.numbers[keep],
            positions=old.atoms.positions[keep],
            pbc=False,
        )
        if old.is_slab:
            # Periodic along z too, across the vacuum: an MLIP then sees what VASP sees
            stripped_atoms.set_pbc(True)
            stripped_atoms.set_cell(old.atoms.get_cell())
            stripped_atoms.cell[2, 2] = max_z + old.vacuum

        # Rebuild the A-site. For an organic A-site, replacing a surface cation
        # removes its molecule, so re-derive the surviving cations from the
        # preserved a_site_id grouping; otherwise keep the symbol.
        A_value = old.A
        if "a_site_id" in old.atoms.arrays:
            stripped_atoms.set_array("a_site_id", old.atoms.get_array("a_site_id")[keep])
            if isinstance(old.A, list) and old.A:
                A_value = Core._build_A_ligands(stripped_atoms, old.A[0])

        self.core = Core(
            A=A_value,
            B=old.B,
            X=old.X,
            atoms=stripped_atoms,
            a=old.a,
            supercell=old.supercell,
            vacuum=old.vacuum,
            build_surface=False,
        )
        self.core._build_octahedra()
        self.core._build_B_ijk()

        self._assign_ligands_to_octahedra()
        self._link_atoms()


    def _place_ligands(
        self,
        max_iters: int = 10,
        overlap_cutoff: float = 1.6,   # Å
        coarse_step_deg: int = 18,
        fine_step_deg: int = 2,
        window_deg: int = 12,
        active_radius_factor: float = math.sqrt(2)/2,
        random_seed: int = 42,
        verbose: bool = True,
        max_anchor_rmsd: float = 0.5,   # Å
        refine_torsions: bool = True,
        stretch_tails: bool = True,
        n_workers: int = 32,   # at most the CPUs available
        **sampler_options,
    ) -> None:

        log = print if verbose else (lambda *a, **k: None)
        rng = random.Random(random_seed)
        sites = self.core.binding_sites
        core_positions = self.core.atoms.get_positions()

        displaced: List[int] = []
        poses: List[SitePose | FitPose] = []
        sampler: Optional[ConstellationSampler] = None

        for spec in self.ligand_specs:
            if spec.ligand.n_motifs > 1:
                if sampler is None:
                    sampler = ConstellationSampler(
                        core_positions, self._build_pbc_tree, self._mic, rng,
                        displaced=displaced, overlap_cutoff=overlap_cutoff,
                        max_rmsd=max_anchor_rmsd, sites=sites, **sampler_options,
                    )
                pools, by_type = motif_pools(spec, sites, self._symbol_map())
                for lig_cloned, cand in select_constellations(spec, pools, by_type, sampler, log,
                                                              refine=refine_torsions):
                    poses.append(FitPose(lig_cloned, cand))
                    if not spec.adsorb:
                        displaced.extend(cand.atom_indices)
                self.ligand_coverage[spec.ligand.name] = spec.coverage
                continue

            if sampler is not None:
                # sites under a multi-motif ligand cannot take one more
                n_blocked = sampler.block_covered(sites, overlap_cutoff + 1.0)
                if n_blocked:
                    log(f"[Log] Left out {n_blocked} free sites covered by multi-motif ligands.")

            for lig_cloned, site in self._take_sites(spec, rng, displaced, log):
                poses.append(SitePose(lig_cloned, np.array(site.plane, dtype=float),
                                      np.array(site.position, dtype=float)))

        assert poses, "No ligands to place."

        core_tree, _ = self._build_pbc_tree(np.delete(core_positions, displaced, axis=0))
        contacts = Contacts(core_tree, self._build_pbc_tree, self._mic, self._wrap,
                            self._build_neighbor_map, overlap_cutoff)
        coords_list = pack(
            poses, contacts, rng, max_iters=max_iters, coarse_step_deg=coarse_step_deg,
            fine_step_deg=fine_step_deg, window_deg=window_deg,
            active_radius=active_radius_factor * self.core.a + 1e-2, max_rmsd=max_anchor_rmsd,
            refine_torsions=refine_torsions, stretch_tails=stretch_tails, n_workers=n_workers,
            log=log, verbose=verbose,
        )
        self._finish_placement([p.ligand for p in poses], coords_list, displaced)

        fit_poses = [p for p in poses if isinstance(p, FitPose)]
        if fit_poses:
            dev = np.concatenate([
                np.linalg.norm(self._mic(p.ligand.anchor_positions - p.candidate.targets), axis=1)
                for p in fit_poses
            ])
            bent = sum(p.origins[p.variant] != "fit" for p in fit_poses)
            closest, n_close = intramolecular_contacts(poses, coords_list)
            log(f"[Log] Multi-motif anchors sit {dev.mean():.2f} Å (max {dev.max():.2f} Å) from their sites; "
                f"{bent} of {len(fit_poses)} ligands refined in torsion space; closest intramolecular "
                f"contact {closest:.2f} Å ({n_close} pairs within {INTRA_SCALE:g}x their vdW radii).")


    def _place_ligands_manually(
        self,
        cat_x_move: float = 0.0,
        cat_y_move: float = 0.0,
        cat_z_move: float = 0.0,
        cat_z_rotate: float = 0.0,
        an_x_move: float = 0.0,
        an_y_move: float = 0.0,
        an_z_move: float = 0.0,
        an_z_rotate: float = 0.0,
        ligand_ids: Optional[Sequence[int]] = None,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Place every ligand at its site without the rotation search, cations shifted by
        cat_x/y/z_move (Å) and turned by cat_z_rotate (degrees), anions by the an_* values.
        With `ligand_ids`, only those ligands (ids count from 0 in placement order) are
        shifted and turned; the rest sit at their sites unmoved.
        """
        if any(spec.ligand.n_motifs > 1 for spec in self.ligand_specs):
            raise NotImplementedError("Manual placement supports single-motif ligands only.")

        log = print if verbose else (lambda *a, **k: None)
        rng = random.Random(random_seed)
        displaced: List[int] = []
        placed = [pair for spec in self.ligand_specs
                  for pair in self._take_sites(spec, rng, displaced, log)]
        assert placed, "No ligands to place."

        moved = set(range(len(placed))) if ligand_ids is None else {int(i) for i in ligand_ids}
        unknown = sorted(moved - set(range(len(placed))))
        if unknown:
            print(f"[Warning] No ligands with ids {unknown}; ids run from 0 to {len(placed) - 1}.")

        cation = np.array([cat_x_move, cat_y_move, cat_z_move], dtype=float), math.radians(cat_z_rotate)
        anion = np.array([an_x_move, an_y_move, an_z_move], dtype=float), math.radians(an_z_rotate)
        unmoved = np.zeros(3), 0.0

        coords_list = []
        for i, (lig, site) in enumerate(placed):
            shift, theta = (cation if lig.charge > 0 else anion) if i in moved else unmoved
            coords = place_about_normal(lig.atoms.get_positions(), np.array(site.plane, dtype=float),
                                        np.array(site.position, dtype=float), lig._anchor_offset, theta)
            coords_list.append(coords + shift)

        self._finish_placement([lig for lig, _ in placed], coords_list, displaced)


    def remove_ligand(self, id: int, re_index: bool = True) -> None:
        idx = next(
            (i for i, lig in enumerate(self.ligands) if lig.id == id),
            None,
        )
        if idx is None:
            raise ValueError(f"No ligand with id={id} found.")

        self.ligands.pop(idx)

        # Re-index remaining ligands
        if re_index:
            for i, lig in enumerate(self.ligands):
                lig.id = i

            # Remove ligand references from octahedra and remap ids. Entries are rebuilt rather
            # than edited in place: frames loaded together by from_ase share one octahedra dict.
            if self.octahedra is not None:
                self.core.octahedra = {
                    b_abs: {**octa, "Ligand": [lid - 1 if lid > id else lid
                                               for lid in octa["Ligand"] if lid != id]}
                    for b_abs, octa in self.octahedra.items()
                }

        self._link_atoms()


    def apply_tilt(
        self,
        glazer: str,
        angles: Tuple[float, float, float],
        *,
        order: str = "xyz",
        move_ligands: bool = True,
    ):
        from .tilt import apply_tilt 
        apply_tilt(
            structure=self,
            glazer=glazer,
            angles=angles,
            order=order,
            move_ligands=move_ligands,
        )

    
    def apply_strain(
        self,
        strain: Sequence[float],          # (ex, ey, ez)
        strain_ligands: bool = True,
    ):
        from .strain import apply_strain
        apply_strain(
            structure=self,
            strain=strain,
            strain_ligands=strain_ligands
        )


    def check_overlaps(self, cutoff: float = 1.6, verbose: bool = True) -> float:
        """
        Closest contact (Å) between the core and a ligand or between two ligands. With
        `verbose`, also lists the pairs closer than `cutoff`.
        """
        log = print if verbose else (lambda *a, **k: None)

        entities = [self.core.atoms.get_positions()] + [lig.atoms.get_positions() for lig in self.ligands]
        entity_labels = ["core"] + [f"ligand_{i}" for i in range(len(self.ligands))]

        centers = np.array([e.mean(axis=0) for e in entities])
        radii = np.array([np.linalg.norm(e - c, axis=1).max() for e, c in zip(entities, centers)])

        global_min = np.inf
        global_pair = None

        n = 0

        for i in range(len(entities)):
            # Atoms come no closer than their bounding spheres, so most pairs need no query
            gaps = np.linalg.norm(self._mic(centers[i + 1:] - centers[i]), axis=1) - radii[i] - radii[i + 1:]
            tree_i = None
            for j, gap in enumerate(gaps, start=i + 1):
                if gap > max(global_min, cutoff) + 1e-6:
                    continue

                if tree_i is None:
                    tree_i, _ = self._build_pbc_tree(entities[i])
                dists_j, _ = tree_i.query(entities[j], k=1)
                d_min = float(np.min(dists_j))

                if d_min < global_min:
                    global_min = d_min
                    global_pair = (i, j)

                if d_min < cutoff:
                    log(
                        f"[OVERLAP] {entity_labels[i]} vs {entity_labels[j]}: "
                        f"min distance = {d_min:.3f} Å < {cutoff:.3f} Å"
                    )
                    n += 1

        log(
            f"[Log] global min inter-entity distance = {global_min:.3f} Å "
            f"between {entity_labels[global_pair[0]]} and {entity_labels[global_pair[1]]}"
            f" (total overlaps: {n})"
        )

        return global_min


    def _mic(self, delta: np.ndarray) -> np.ndarray:
        pbc = self.core.atoms.pbc
        if not pbc.any():
            return delta
        box = self.core.atoms.cell.lengths()
        d = np.array(delta, dtype=float)
        for ax in np.flatnonzero(pbc):
            d[..., ax] -= box[ax] * np.round(d[..., ax] / box[ax])
        return d


    def _wrap(self, points: np.ndarray) -> np.ndarray:
        pbc = self.core.atoms.pbc
        if not pbc.any():
            return points
        box = self.core.atoms.cell.lengths()
        x = np.array(points, dtype=float)
        for ax in np.flatnonzero(pbc):
            x[..., ax] -= box[ax] * np.floor(x[..., ax] / box[ax])
        return x


    def _build_pbc_tree(self, points: np.ndarray) -> Tuple[cKDTree, np.ndarray]:
        """
        KD-tree of `points` and their images one box away along the periodic axes, with the
        point each comes from.
        """
        pts = np.asarray(points, dtype=float)
        x, y, z = [(-1, 0, 1) if periodic else (0,) for periodic in self.core.atoms.pbc]
        shifts = np.array([(i, j, k) for i in x for j in y for k in z], dtype=float)
        shifts = shifts * self.core.atoms.cell.lengths()
        return cKDTree((pts + shifts[:, None]).reshape(-1, 3)), np.tile(np.arange(len(pts)), len(shifts))


    def _build_neighbor_map(self, centers: np.ndarray, radii: np.ndarray, cutoff: float):
        if not self.core.atoms.pbc.any():
            return build_neighbor_map(centers, radii, cutoff)

        tree, idx_map = self._build_pbc_tree(centers)
        rmax = float(np.max(radii))

        neighbor_map: Dict[int, List[int]] = {}
        for i in range(len(centers)):
            hits = tree.query_ball_point(centers[i], r=float(radii[i] + rmax + cutoff))
            cand = set(int(idx_map[h]) for h in hits)
            cand.discard(i)

            # exact filter with MIC on centers
            neighbor_map[i] = [j for j in cand
                               if np.linalg.norm(self._mic(centers[i] - centers[j])) < radii[i] + radii[j] + cutoff]

        return neighbor_map


    def _assign_ligands_to_octahedra(self) -> None:
        octahedra = self.octahedra
        if not octahedra:
            return

        at = self.core.atoms
        syms = np.array(at.get_chemical_symbols())
        b_idx = np.where(syms == self.core.B)[0]

        if self.core.is_slab:
            cell = at.get_cell()
            boxsize = tuple(cell.lengths())
            core_pos = at.get_scaled_positions(wrap=True) @ cell.array
        else:
            boxsize = None
            core_pos = at.get_positions()
        B_pos = core_pos[b_idx]

        anchor_positions = []
        ligand_ids = []

        for lig in self.ligands:
            if lig.n_motifs > 1:
                # every motif bound to an X or B site belongs to the octahedron there
                for motif, pos in zip(lig.binding_motifs, lig.anchor_positions):
                    if motif.site in ("X", "B"):
                        anchor_positions.append(pos)
                        ligand_ids.append(int(lig.id))
            elif lig.charge <= 0 and lig.anchor_pos is not None:
                anchor_positions.append(np.asarray(lig.anchor_pos, dtype=float))
                ligand_ids.append(int(lig.id))

        if anchor_positions:
            anchor_positions = np.vstack(anchor_positions)

            b_tree = cKDTree(B_pos, boxsize=boxsize)
            _, b_loc = b_tree.query(anchor_positions, k=1)

            for j, b_loc_j in enumerate(b_loc):
                b_abs = int(b_idx[int(b_loc_j)])
                if ligand_ids[j] not in octahedra[b_abs]["Ligand"]:
                    octahedra[b_abs]["Ligand"].append(ligand_ids[j])


    def to(
        self, 
        fmt: Optional[str] = None, 
        filename: Optional[str] = None, 
        write_json: bool = True,
        vacuum: float = 15.0,
        **kwargs
    ) -> None:      
        """
        Write the structure and, with `write_json`, its metadata to `filename`.json.
        `fmt` defaults to "vasp" for a slab and "xyz" for a nanocrystal; a nanocrystal
        written as vasp is boxed with `vacuum` Å around it.
        """
        at = self.atoms
        formula = at.get_chemical_formula()
        if fmt is None:
            fmt = "vasp" if self.core.is_slab else "xyz"

        if filename is None:
            filename = f"{formula}.{fmt}"
        
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "vasp":
            if self.core.is_nanocrystal:
                pos = at.get_positions()
                center = pos.mean(axis=0)
                cell_diag = pos.max(axis=0) - pos.min(axis=0) + vacuum

                at = at.copy()
                at.set_cell(np.diag(cell_diag))
                at.positions += cell_diag / 2 - center
                at.pbc = True
            write_vasp(str(path), at, **{**kwargs, "sort": True})
        else:
            write(str(path), at, **{**kwargs, "format": fmt, "comment": formula})

        if write_json:
            self.to_json(str(path) + ".json", sort=fmt == "vasp")


    def to_json(self, filename: str, sort: bool = False) -> None:

        core_meta = {
            "A": self.core.a_site_metadata(),
            "B": self.core.B,
            "X": self.core.X,
            "a": self.core.a,
            "supercell": list(self.core.supercell),
        }
        if self.core.is_slab:
            core_meta["cell"] = self.core.atoms.get_cell().array.tolist()
            core_meta["pbc"] = [bool(x) for x in self.core.atoms.pbc]
            core_meta["vacuum"] = float(self.core.vacuum)
        core_meta["n_core_atoms"] = len(self.core.atoms)

        ligand_types_meta = []
        type_key_to_id: Dict[tuple, int] = {}
        ligands_meta = []

        for i, lig in enumerate(self.ligands):
            key = lig._type_key()

            if key not in type_key_to_id:
                type_key_to_id[key] = len(ligand_types_meta)
                ligand_types_meta.append(
                    {
                        "id": type_key_to_id[key],
                        **lig._type_metadata(),
                        "coverage": self.ligand_coverage.get(lig.name),
                        "n_atoms": len(lig.atoms),
                        "volume": float(lig.volume),
                        "n_instances": 0,
                    }
                )

            spec_id = type_key_to_id[key]
            ligand_types_meta[spec_id]["n_instances"] += 1
            ligands_meta.append({"ligand_id": i, "spec_id": spec_id, **lig._instance_metadata()})

        topo = {
            "schema_version": 1,
            "type": "slab" if self.core.is_slab else "nanocrystal",
            "n_total_atoms": len(self.atoms),
            "core": core_meta,
            "ligand_types": ligand_types_meta,
            "core_indices": {"octahedra": self.octahedra, "B_ijk": self.B_ijk},
            "ligands": ligands_meta,
        }

        if sort:
            topo["sort_idx"] = np.argsort(self.atoms.symbols).tolist()

        with open(filename, "w") as f:
            json.dump(topo, f, indent=2)


    @classmethod
    def from_file(cls, filename: str, json_filename: str) -> Assembly:
        return cls.from_ase(read(filename), json_filename)


    @classmethod
    def from_ase(cls,
                 atoms: Atoms | List[Atoms],
                 json_filename: str
                 ) -> Assembly | List[Assembly]:

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
        motifs = {t["id"]: BindingMotif._from_metadata(t) for t in ligand_types_meta}

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

            # Split plain arrays instead of slicing atoms_i: every ASE slice copies all
            # arrays and re-indexes the constraints, which dominated the per-frame cost.
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
                A=Core.a_site_from_json(core_meta["A"], core_atoms),
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
                    planes=Ligand._planes_from_metadata(inst_meta),
                    volume=tmeta["volume"],
                    binding_atoms=binding_atoms,
                )

                ligands.append(lig)

            core.octahedra = octahedra
            core.B_ijk = B_ijk
            assemblies.append(cls(core=core, ligands=ligands,
                                  ligand_coverage={t["name"]: t["coverage"] for t in ligand_types_meta}))

        return assemblies[0] if single_input else assemblies


    def _link_atoms(self) -> None:
        """
        Build the whole structure in `atoms`, with the positions of the core and the ligands
        as views into its buffer and each ligand's `indices` into it.
        """
        core_atoms = self.core.atoms
        all_numbers = [core_atoms.numbers]
        all_positions = [core_atoms.positions]

        for lig in self.ligands:
            all_numbers.append(lig.atoms.numbers)
            all_positions.append(lig.atoms.positions)

        self.atoms = Atoms(
            numbers=np.concatenate(all_numbers),
            positions=np.vstack(all_positions),
            pbc=core_atoms.pbc,
            cell=core_atoms.get_cell(),
        )

        shared = self.atoms.arrays["positions"]
        cursor = len(core_atoms)
        core_atoms.arrays["positions"] = shared[:cursor]

        for lig in self.ligands:
            n = len(lig.atoms)
            lig.indices = np.arange(cursor, cursor + n, dtype=int)
            lig.atoms.arrays["positions"] = shared[cursor:cursor + n]
            cursor += n
