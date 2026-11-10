from concurrent.futures import ProcessPoolExecutor
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import PlacementConfig
from .kernels import _nearest, make_cells
from .refine import inside_depth, stretch


WORKER_JOBS = 32      # ligands a stretch process is worth
_WORKER_CORE = None


def _single_threaded_blas() -> None:
    """
    Run the OpenBLAS that scipy uses on one thread (Linux; silently nothing elsewhere):
    every L-BFGS step wakes all its threads, which starve each other in a pool of workers.
    The copy bundled with numpy wheels, which refinement barely uses, is left alone.
    """
    import ctypes
    try:
        paths = {line.split()[-1] for line in open("/proc/self/maps")
                 if "openblas" in line and "/numpy.libs/" not in line}
    except OSError:
        return
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for name in ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
                     "openblas_set_num_threads64_", "openblas_set_num_threads"):
            if hasattr(lib, name):
                getattr(lib, name)(1)


def _init_stretch_worker(core_points: np.ndarray) -> None:
    global _WORKER_CORE
    _single_threaded_blas()
    _WORKER_CORE = core_points


def _stretch_job(job: tuple, core_points=None) -> Optional[np.ndarray]:
    model, coords, groups, targets, normal, interior, config = job
    res = stretch(model, coords, groups, targets, normal,
                  [_WORKER_CORE if core_points is None else core_points],
                  config.overlap_cutoff + config.contact_margins[0], config, config.stretch_iters,
                  interior=interior)
    return None if res is None else res.coords


def available_cpus() -> int:
    """CPUs this process may run on (a Slurm allocation or taskset included)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _stretched(poses: Sequence, core_points: np.ndarray, interior: tuple, config: PlacementConfig,
               n_workers: int) -> List[Optional[np.ndarray]]:
    """Stretched coordinates of the best variant of each of `poses`; None where it cannot be."""
    jobs = [(p.ligand.torsion_model(), p.variants[0], p.ligand.binding_groups, p.targets, p.up, interior,
             config) for p in poses]
    # a process costs more than a few ligands: each one has to load the compiled code
    n_workers = min(n_workers, available_cpus(), len(jobs) // WORKER_JOBS)
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_stretch_worker,
                                 initargs=(core_points,)) as pool:
            return list(pool.map(_stretch_job, jobs, chunksize=max(1, len(jobs) // (4 * n_workers))))
    return [_stretch_job(job, core_points) for job in jobs]


def stretch_poses(poses: Sequence, core_points: np.ndarray, interior: tuple, config: PlacementConfig,
                  n_workers: int = 1) -> int:
    """
    Put a stretched copy of each pose's best variant first: its dangling chains anti and
    raised along the pose's `up`, its anchors and orientation untouched.

    Single-motif ligands of one kind on one plane start from one shape and stand alike, so
    one of them is stretched and the others take the result; one that would then touch the
    core or reach into its `interior` box is stretched on its own. Stretching sees only the
    core, so the ligands are independent and go to up to `n_workers` processes, no more
    than the CPUs available and one per WORKER_JOBS ligands; each receives `core_points`
    (the core atoms and their periodic images) once. The result does not depend on
    `n_workers`.

    Returns:
        How many poses now start stretched: they had chains to stretch and the result
        clears the core and its interior, or site selection had already stretched them.
    """
    cutoff = config.overlap_cutoff
    core_cells = make_cells(core_points, cutoff)

    def take(pose, coords: Optional[np.ndarray]) -> bool:
        if coords is None or float(_nearest(coords, core_cells, cutoff).min()) < cutoff \
                or inside_depth(coords, interior).any():
            return False
        pose.variant = pose.add_variant(coords, source="stretched", first=True)
        return True

    done = sum(p.sources[p.variant] == "stretched" for p in poses)
    alike: Dict[tuple, List] = {}
    for p in poses:
        if p.sources[p.variant] != "stretched" and p.ligand.torsion_model().heavy_tails.any():
            key = (id(p),) if p.spin_axis is None else (id(p.ligand.torsion_model()), *p.spin_axis)
            alike.setdefault(key, []).append(p)

    groups = list(alike.values())
    alone = []
    firsts = _stretched([g[0] for g in groups], core_points, interior, config, n_workers)
    for (first, *rest), coords in zip(groups, firsts):
        done += take(first, coords)
        for p in rest:
            if coords is not None and take(p, coords + p.targets - first.targets):
                done += 1
            else:
                alone.append(p)
    return done + sum(map(take, alone, _stretched(alone, core_points, interior, config, n_workers)))
