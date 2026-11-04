from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..conformers import ConformerLibrary
from ..core import BindingSite
from ..utils.rotation import rotation_about_axis


def site_normals(sites: Sequence[BindingSite]) -> np.ndarray:
    """Unit normal of every site's plane: (n_sites, 3)."""
    return np.array([np.asarray(s.plane, dtype=float) / np.linalg.norm(s.plane) for s in sites])


def mean_normal(sites: Sequence[BindingSite]) -> np.ndarray:
    """Unit mean of the sites' normals."""
    n = site_normals(sites).mean(axis=0)
    return n / np.linalg.norm(n)


@dataclass
class Fit:
    """
    Rigid placement of one conformer onto a constellation.

    Coordinates are ``(x - center) @ rotation.T + pivot``, then turned by ``theta`` about
    ``axis`` through ``pivot`` when the fit leaves that spin free (two anchors, or anchors
    on one line). Other poses keep their coordinates in ``xyz``. ``origin`` says where a
    fit came from: "fit", "stretched" (a template with its chains stood up), "refined",
    "unstuck" (chains stretched just until they clear the core) or "embedded".
    """
    conformer: int
    rotation: np.ndarray
    center: np.ndarray
    pivot: np.ndarray
    axis: Optional[np.ndarray]
    theta: float
    rmsd: float
    alignment: float
    footprint: float
    energy: float
    cost: float
    xyz: Optional[np.ndarray] = None
    origin: str = "fit"

    def coords(self, lib: ConformerLibrary) -> np.ndarray:
        if self.xyz is not None:
            return self.xyz.copy()
        x = (lib.coords[self.conformer] - self.center) @ self.rotation.T
        if self.axis is not None and self.theta != 0.0:
            x = x @ rotation_about_axis(self.axis, self.theta).T
        return x + self.pivot


@dataclass
class Candidate:
    """
    A constellation accepted for one ligand, with its feasible fits, best first.
    """
    sites: List[BindingSite]
    targets: np.ndarray
    fits: List[Fit]

    @property
    def atom_indices(self) -> List[int]:
        return [i for s in self.sites for i in s.atom_indices]


def fit_conformers(
    lib: ConformerLibrary,
    targets: np.ndarray,
    normals: np.ndarray,
    sigma_rmsd: float = 0.25,
    energy_scale: float = 5.0,
    w_align: float = 2.0,
    w_foot: float = 1.0,
    foot_scale: float = 2.0,
    n_spin: int = 72,
) -> Dict[str, np.ndarray]:
    """
    Kabsch-fit every conformer's anchors to `targets` at once.

    The cost of a conformer is ``fit + orient``, with
    ``fit = (rmsd / sigma_rmsd)^2 + energy / energy_scale`` and
    ``orient = w_align * (1 - alignment) + w_foot * (footprint / foot_scale)^2``.
    Alignment is the mean cosine between motif directions and site normals; footprint is
    the spread (Å, RMS) of the body atoms across the mean normal, small when the ligand
    stands up rather than lying on the surface. When the fit leaves a spin free, the
    spin minimizing ``orient`` is taken.

    Returns:
        Dict of per-conformer arrays: rotation (C, 3, 3), center (C, 3), rmsd, alignment,
        footprint, fit, cost (C,), plus the pivot (3,) and spin axis (3,) or None.
    """
    P = lib.anchors
    center = P.mean(axis=1)
    pivot = targets.mean(axis=0)
    Pc = P - center[:, None]
    Tc = targets - pivot

    H = np.einsum("cni,nj->cij", Pc, Tc)
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ np.transpose(U, (0, 2, 1))))
    D = np.tile(np.eye(3), (len(P), 1, 1))
    D[:, 2, 2] = np.where(d == 0, 1.0, d)
    R = V @ D @ np.transpose(U, (0, 2, 1))

    mapped = Pc @ R.transpose(0, 2, 1)
    rmsd = np.sqrt(np.mean(np.sum((mapped - Tc) ** 2, axis=-1), axis=1))

    sv = np.linalg.svd(Tc, compute_uv=False)
    axis = None
    if len(targets) == 2 or sv[1] < 0.05 * sv[0]:
        axis = Tc[-1] - Tc[0] if len(targets) == 2 else np.linalg.svd(Tc)[2][0]
        axis = axis / np.linalg.norm(axis)

    has_dir = np.linalg.norm(lib.directions, axis=-1) > 0          # (C, n)
    n_dir = np.maximum(has_dir.sum(axis=1), 1)
    dirs = lib.directions @ R.transpose(0, 2, 1)

    # lateral spread across the mean normal n: trace(cov) - n.cov.n of the body atoms
    nbar = normals.mean(axis=0)
    nbar = nbar / np.linalg.norm(nbar)
    body = (lib.coords[:, lib.body] - center[:, None]) @ R.transpose(0, 2, 1)
    body -= body.mean(axis=1, keepdims=True)
    cov = (body.transpose(0, 2, 1) @ body) / body.shape[1]
    spread = np.trace(cov, axis1=1, axis2=2)

    if axis is not None:
        thetas = np.linspace(0.0, 2.0 * np.pi, n_spin, endpoint=False)
        spins = rotation_about_axis(axis, thetas)                    # (S, 3, 3)
        # cos[c, s, n] = normal_n . (spin_s @ dir_cn), with the normals turned back instead
        back = normals @ spins                                        # (S, n, 3)
        cos = (dirs.transpose(1, 0, 2) @ back.transpose(1, 2, 0)).transpose(1, 2, 0)   # (C, S, n)
        align = cos.sum(axis=-1) / n_dir[:, None]
        u = np.einsum("sji,j->si", spins, nbar)                       # spin^T n
        lat2 = spread[:, None] - np.einsum("si,cij,sj->cs", u, cov, u)
        orient = w_align * (1.0 - align) + w_foot * lat2 / foot_scale ** 2
        best = np.argmin(orient, axis=1)
        idx = np.arange(len(P))
        R = spins[best] @ R
        alignment, lat2, orient = align[idx, best], lat2[idx, best], orient[idx, best]
    else:
        alignment = np.einsum("cnj,nj->cn", dirs, normals).sum(axis=-1) / n_dir
        lat2 = spread - np.einsum("i,cij,j->c", nbar, cov, nbar)
        orient = w_align * (1.0 - alignment) + w_foot * lat2 / foot_scale ** 2

    fit = (rmsd / sigma_rmsd) ** 2 + lib.energies / energy_scale
    return dict(rotation=R, center=center, pivot=pivot, axis=axis, rmsd=rmsd,
                alignment=alignment, footprint=np.sqrt(np.maximum(lat2, 0.0)),
                fit=fit, cost=fit + orient)
