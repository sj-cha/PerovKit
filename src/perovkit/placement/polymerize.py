from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import GetUFFBondStretchParams

from .kernels import _closest
from .pack import Contacts
from .refine import CONTACT_MARGINS, refine_joint

K_LINK = 160.0         # per Å^2, pull of a cap's bond point onto the partner carbon
LINK_TOLERANCE = 0.1   # Å, a bond whose points end farther from their partners is not made
LINK_STOP = 0.02       # Å, the refinement making bonds stops once every point is this close
NEAR = 8.0             # Å, bonds this short are made first, longer ones only between the ends left
WHOLE = 20             # bonded units refined whole when no more; a larger network only around new bonds,
HOPS = 2               # those within this many bonds of them moving, the next ones held


class _Sets:
    """Union-find over hashable keys."""

    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b) -> None:
        self.parent[self.find(a)] = self.find(b)


def _components(links: List[tuple]) -> List[List[tuple]]:
    """The links grouped by the poses they join."""
    sets = _Sets()
    for (l, _), (m, _) in links:
        sets.union(l, m)
    groups: Dict[int, List[tuple]] = {}
    for link in links:
        groups.setdefault(sets.find(link[0][0]), []).append(link)
    return list(groups.values())


def _candidates(poses: Sequence, coords_list: List[np.ndarray], box, reach: float, made: List[tuple],
                tried: Set[tuple]) -> List[tuple]:
    """
    Bonds to try, tail of a unit to head of another ligand's, nearest first, within `reach`
    and not `tried`: between the ends the `made` bonds left, each end once, and no chain
    closing on itself. A unit is (pose, reactive group).
    """
    sets, tails, heads, links = _Sets(), set(), set(), []
    for u, w in made:
        tails.add(u)
        heads.add(w)
        sets.union(u, w)
    units = [(l, g) for l, p in enumerate(poses) for g in range(len(p.ligand.reactive_groups))]
    free_tails = [u for u in units if u not in tails]
    free_heads = [w for w in units if w not in heads]
    tail_x = np.reshape([coords_list[l][poses[l].ligand.reactive_groups[g][1]] for l, g in free_tails], (-1, 3))
    head_x = np.reshape([coords_list[l][poses[l].ligand.reactive_groups[g][0]] for l, g in free_heads], (-1, 3))
    d = np.linalg.norm(box.mic(head_x[None] - tail_x[:, None]), axis=2)
    pairs = sorted((float(d[i, j]), free_tails[i], free_heads[j]) for i, j in zip(*np.nonzero(d <= reach))
                   if free_tails[i][0] != free_heads[j][0] and (free_tails[i], free_heads[j]) not in tried)
    for _, u, w in pairs:
        if u in tails or w in heads or sets.find(u) == sets.find(w):
            continue
        tails.add(u)
        heads.add(w)
        sets.union(u, w)
        links.append((u, w))
    return links


def _rest_length(tail_ligand, g: int, head_ligand, h: int) -> float:
    """
    UFF rest length (Å) of the bond the tail of group `g` of `tail_ligand` makes with the
    head of group `h` of `head_ligand`, read off the two units joined; both still have their
    caps, which follow every other atom.
    """
    first = tail_ligand.mol
    tail, tail_cap = tail_ligand.reactive_groups[g][1], tail_ligand.reactive_groups[g][3]
    head, head_cap = head_ligand.reactive_groups[h][0], head_ligand.reactive_groups[h][2]
    n = first.GetNumAtoms()
    pair = Chem.RWMol(Chem.CombineMols(first, head_ligand.mol))
    pair.AddBond(tail, n + head, Chem.BondType.SINGLE)
    pair.RemoveAtom(n + head_cap)
    pair.RemoveAtom(tail_cap)   # the second unit's atoms move down one
    pair = pair.GetMol()
    Chem.SanitizeMol(pair)
    params = GetUFFBondStretchParams(pair, tail, n - 1 + head)
    if params is None:
        raise ValueError(f"UFF has no parameters for the bond between {tail_ligand.name!r} and "
                         f"{head_ligand.name!r}; give their ReactiveMotif a bond_length")
    return float(params[1])


def _image(box, frm: np.ndarray, to: np.ndarray) -> int:
    """Index in box.shifts of the image of `to` nearest `frm`."""
    return int(np.argmin(np.linalg.norm(to + box.shifts - frm, axis=1)))


def bond_units(
    poses: Sequence,
    coords_list: List[np.ndarray],
    contacts: Contacts,
    max_rmsd: float,
    reach: float,
    log: Callable = print,
) -> List[Tuple[int, int, int, int]]:
    """
    Bond the reactive groups of neighboring ligands into chains, the tail carbon of a unit to
    the head carbon of another, each pair in place of their caps: bonds up to NEAR Å first,
    then up to `reach` between the ends left, the free ends paired nearest first again and
    again. The units new bonds join are refined with all units bonded to them, or within a
    network larger than WHOLE only those up to HOPS bonds away, the next ones held, each
    cap's point a bond length out pulled onto the partner carbon; the new bonds stay when
    every point ends within LINK_TOLERANCE, no contact but a bond is under the cutoff and
    every anchor is within `max_rmsd` (RMS, Å) of its sites, else the worst-fitting one is
    given up and the rest tried again. A bond once made stays. Updates `coords_list`.

    Returns:
        The bonds, as (pose, carbon, pose, carbon).
    """
    box, cutoff = contacts.box, contacts.cutoff
    groups = [p.ligand.reactive_groups for p in poses]
    removed: Dict[int, Set[int]] = {}   # the caps of the bonds made
    made: List[tuple] = []
    bonded_to: Dict[int, Set[int]] = {l: set() for l in range(len(poses))}   # by the bonds made
    lengths: Dict[tuple, float] = {}
    no_shift = int(np.flatnonzero(~box.shifts.any(axis=1))[0])

    def bond_length(link) -> float:
        """The motifs' bond length, or else the UFF rest length, once per kind of pair."""
        (l, g), (m, h) = link
        tail, head = poses[l].ligand, poses[m].ligand
        given = tail.reactive_motifs[g].bond_length or head.reactive_motifs[h].bond_length
        if given:
            return given
        key = (tail.smiles, g, head.smiles, h)
        if key not in lengths:
            lengths[key] = _rest_length(tail, g, head, h)
        return lengths[key]

    def ends(link) -> tuple:
        """(tail pose, tail carbon, its cap, head pose, head carbon, its cap) of a bond."""
        (l, g), (m, h) = link
        return l, groups[l][g][1], groups[l][g][3], m, groups[m][h][0], groups[m][h][2]

    def link_arrays(links, start, coords, loose_caps):
        """
        The rows of System.links, one per direction of each bond, their scales and K_LINK; and a
        row that pulls nothing for each of `loose_caps`, (pose, cap) of bonds to units left
        out, so that those caps touch nothing either.
        """
        cap, own, partner, image, scale = [], [], [], [], []
        for l, c in loose_caps:
            carbon = start[l] + next(o for h, t, hc, tc in groups[l] for o, cc in ((h, hc), (t, tc)) if cc == c)
            cap.append(start[l] + c)
            own.append(carbon)
            partner.append(carbon)
            image.append(no_shift)
            scale.append(0.0)
        for link in links:
            l, tail, tail_cap, m, head, head_cap = ends(link)
            i = _image(box, coords[l][tail], coords[m][head])
            j = _image(box, coords[m][head], coords[l][tail])
            length = bond_length(link)
            for c, o, q, k, a, b in [(tail_cap, tail, head, i, l, m), (head_cap, head, tail, j, m, l)]:
                cap.append(start[a] + c)
                own.append(start[a] + o)
                partner.append(start[b] + q)
                image.append(k)
                scale.append(length / float(np.linalg.norm(coords[a][c] - coords[a][o])))
        return np.stack([cap, own, partner, image], axis=1).astype(np.int64), np.array(scale), K_LINK

    def attempt(new_bonds) -> Tuple[bool, np.ndarray, Dict[int, np.ndarray]]:
        """
        Refine around `new_bonds`. Returns whether they stay, how far each one's points ended
        from their partners, and the coordinates of the units that moved.
        """
        joined = {u[0] for u, _ in new_bonds} | {w[0] for _, w in new_bonds}
        component, edge = set(joined), set(joined)
        while edge:
            edge = {n for l in edge for n in bonded_to[l]} - component
            component |= edge
        if len(component) <= WHOLE:
            moving = component
        else:
            moving = set(joined)
            for _ in range(HOPS):
                moving |= {n for l in moving for n in bonded_to[l]}
        held = {n for l in moving for n in bonded_to[l]} - moving
        members = sorted(moving | held)
        start = {l: s for l, s in zip(members, np.cumsum([0] + [len(coords_list[l]) for l in members]))}
        links = new_bonds + [b for b in made if b[0][0] in start and b[1][0] in start]
        covered = set()   # the caps of these bonds
        for link in links:
            l, _, tail_cap, m, _, head_cap = ends(link)
            covered |= {(l, tail_cap), (m, head_cap)}
        loose = sorted((l, c) for l in held for c in removed.get(l, ()) if (l, c) not in covered)
        others = contacts.around([np.delete(coords_list[l], sorted(removed.get(l, ())), axis=0)
                                  for l in range(len(poses)) if l not in start])
        arrays = link_arrays(links, start, coords_list, loose)
        results = refine_joint(
            [poses[l].ligand.torsion_model() for l in members], [coords_list[l] for l in members],
            [poses[l].ligand.binding_groups for l in members], [poses[l].targets for l in members],
            [contacts.core_cells[0], others[0]], cutoff + CONTACT_MARGINS[0], box.shifts,
            stop_at=cutoff + CONTACT_MARGINS[0] / 2, stop_rmsd=max_rmsd, stop_link=LINK_STOP,
            spin_axes=[poses[l].spin_axis for l in members], interior=contacts.interior, links=arrays,
            held_poses=[l in held for l in members])
        new = {l: r.coords for l, r in zip(members, results) if l in moving}
        x = {l: new.get(l, coords_list[l]) for l in members}

        # how far the bond points ended from their partners, per bond (the loose rows come first)
        X = np.concatenate([x[l] for l in members])
        (cap, own, partner, image), scale = arrays[0][len(loose):].T, arrays[1][len(loose):]
        points = X[own] + scale[:, None] * (X[cap] - X[own])
        miss = np.linalg.norm(points - X[partner] - box.shifts[image], axis=1).reshape(-1, 2).max(axis=1)
        worst_anchor = max(r.anchor_rmsd for l, r in zip(members, results) if l in moving)
        if miss.max() > LINK_TOLERANCE or worst_anchor > max_rmsd:
            return False, miss[:len(new_bonds)], new

        # contacts of the units that moved, without the caps bonds replace, bonded carbons excepted
        gone = {l: set(removed.get(l, ())) for l in members}
        bonded = set()
        for link in links:
            l, tail, tail_cap, m, head, head_cap = ends(link)
            gone[l].add(tail_cap)
            gone[m].add(head_cap)
            bonded |= {((l, tail), (m, head)), ((m, head), (l, tail))}
        kept = {l: np.delete(np.arange(len(x[l])), sorted(gone[l])) for l in members}
        y = np.concatenate([x[l][kept[l]] for l in members])
        owner = np.concatenate([np.full(len(kept[l]), l) for l in members])
        atom = np.concatenate([kept[l] for l in members])
        if _closest(y[np.isin(owner, list(moving))], contacts.core_cells, others, contacts.reach) < cutoff:
            return False, miss[:len(new_bonds)], new
        tree, image_of = box.tree(y)
        for a, near in enumerate(tree.query_ball_point(y, cutoff)):
            if owner[a] not in moving:
                continue
            for b in near:
                q = image_of[b]
                if owner[q] != owner[a] and ((owner[a], atom[a]), (owner[q], atom[q])) not in bonded:
                    return False, miss[:len(new_bonds)], new
        return True, miss[:len(new_bonds)], new

    tried: Set[tuple] = set()
    for bound in sorted({min(NEAR, reach), reach}):
        while True:
            fresh = _candidates(poses, coords_list, box, bound, made, tried)
            if not fresh:
                break
            tried.update(fresh)
            pending = _components(fresh)   # new bonds sharing units, tried together
            while pending:
                new_bonds = pending.pop()
                ok, miss, new = attempt(new_bonds)
                if not ok:
                    worst = int(np.argmax(miss))
                    pending += _components(new_bonds[:worst] + new_bonds[worst + 1:])
                    continue
                for l, x in new.items():
                    coords_list[l] = x
                for link in new_bonds:
                    l, _, tail_cap, m, _, head_cap = ends(link)
                    removed.setdefault(l, set()).add(tail_cap)
                    removed.setdefault(m, set()).add(head_cap)
                    bonded_to[l].add(m)
                    bonded_to[m].add(l)
                made += new_bonds

    if tried:
        log(f"[Log] Bonded {len(made)} pairs of reactive groups within {reach:g} Å ({len(tried)} tried) "
            f"into {len(_components(made))} chains.")
    return [(l, tail, m, head) for l, tail, _, m, head, _ in map(ends, made)]
