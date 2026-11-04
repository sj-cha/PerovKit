from concurrent.futures import ProcessPoolExecutor
import os
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..refine import CONTACT_MARGINS, stretch
from .fitting import mean_normal
from .poses import MultiMotifPose


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
    _WORKER_CORE = cKDTree(core_points)


def _stretch_job(job: tuple, core=None) -> Optional[np.ndarray]:
    model, coords, groups, targets, normal, contact = job
    res = stretch(model, coords, groups, targets, normal,
                  [_WORKER_CORE if core is None else core], contact)
    return None if res is None else res.coords


def available_cpus() -> int:
    """CPUs this process may run on (a Slurm allocation or taskset included)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def stretch_poses(poses: Sequence, core_tree, cutoff: float, n_workers: int = 1) -> int:
    """
    Put a stretched copy of each multi-motif pose's best variant first: its dangling
    chains anti and raised along the mean normal of its sites, its anchors untouched.

    Stretching sees only the core, so the ligands are independent and, with `n_workers`
    above 1, go to that many processes (no more than the CPUs available); each builds the
    core tree once. The result does not depend on `n_workers`.

    Returns:
        How many poses now start stretched: they had chains to stretch and the result
        clears the core, or site selection had already stretched them.
    """
    multi_motif_poses = [p for p in poses if isinstance(p, MultiMotifPose)]
    done = sum(p.origins[p.variant] == "stretched" for p in multi_motif_poses)
    todo = [p for p in multi_motif_poses
            if p.origins[p.variant] != "stretched" and p.ligand.torsion_model().tails.any()]
    jobs = [(p.ligand.torsion_model(), p.variants[0], p.ligand.binding_groups,
             p.candidate.targets, mean_normal(p.candidate.sites), cutoff + CONTACT_MARGINS[0]) for p in todo]
    n_workers = min(n_workers, available_cpus(), len(jobs))
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_stretch_worker,
                                 initargs=(core_tree.data,)) as pool:
            results = list(pool.map(_stretch_job, jobs,
                                    chunksize=max(1, len(jobs) // (4 * n_workers))))
    else:
        results = [_stretch_job(job, core_tree) for job in jobs]
    for pose, coords in zip(todo, results):
        if coords is not None and float(core_tree.query(coords)[0].min()) >= cutoff:
            pose.variant = pose.add_variant(coords, origin="stretched", first=True)
            done += 1
    return done
