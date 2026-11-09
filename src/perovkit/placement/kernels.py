from collections import namedtuple
import math

import numba
import numpy as np


SKIN = 1.0   # Å, Verlet list skin

System = namedtuple("System", [
    "X0",             # (N, 3) starting atoms of all poses
    "atom_start",     # (L + 1) atoms of each pose
    "param_start",    # (L + 1) parameters of each pose: translation (3), Euler angles (3), torsions
    "torsion_start",  # (L + 1) bonds of each pose
    "bonds",          # (B, 2) rotatable bonds, deepest first in each pose
    "moving_start",   # (B + 1) atoms each bond moves
    "moving",
    "center",         # (L, 3) the Euler angles turn about these
    "frame",          # (L, 3, 3) and about the rows of these
    "framed",         # (L) whether the frame is not the identity
    "group_pose",     # (G) the pose of each motif
    "group_start",    # (G + 1) binding atoms of each motif
    "group_atoms",
    "targets",        # (G, 3) anchor targets
    "n_groups",       # (L) motifs of each pose
    "pose_model",     # (L) the ligand type of each pose
    "far_start",      # (types + 1) pairs of atoms each type keeps apart
    "far_a",          # (F) atoms of each pair, numbered within the pose
    "far_b",
    "far_limit",      # (F) their closest distance
    "owner",          # (N) the pose of each atom
    "fixed",          # cells of the fixed atoms
    "shifts",         # (S, 3) periodic images
    "contact_distance", "k_anchor", "k_contact",
    "k_intra",        # weight of a pair of a pose within its limit
    "k_inside",       # weight of an atom inside the box lo..hi
    "w_torsion",
    "confined",       # whether atoms are kept out of the box lo..hi
    "lo", "hi",
    "pull_atoms",     # atoms pulled along pull_direction by k_pull per Å
    "pull_direction", "k_pull",
    "links",          # (K, 4) bonds being made between poses, a row per direction: the cap the
                      #     bond replaces, its carbon, the partner carbon and the partner's image
    "link_scale",     # (K) the point this far from the carbon to the cap is pulled onto the
    "k_link",         #     partner; caps touch nothing, and the two carbons not each other
])


@numba.njit(cache=True)
def doubled(a):
    """`a` in an array twice as long, for lists that grow in compiled code."""
    b = np.empty(2 * a.shape[0], dtype=a.dtype)
    for i in range(a.shape[0]):
        b[i] = a[i]
    return b


@numba.njit(cache=True)
def _capped(a, links):
    """Whether atom `a` is the cap of a bond being made, in no contact then."""
    for t in range(links.shape[0]):
        if links[t, 0] == a:
            return True
    return False


@numba.njit(cache=True)
def _bonding(p, q, links):
    """Whether atoms `p` and `q` are the carbons of a bond being made."""
    for t in range(links.shape[0]):
        if links[t, 1] == p and links[t, 2] == q:
            return True
    return False


@numba.njit(cache=True)
def _images(x, shifts):
    """Every image of the atoms x (N, 3) at `shifts` (S, 3): (N * S, 3), shift after shift."""
    N, S = x.shape[0], shifts.shape[0]
    images = np.empty((N * S, 3))
    for i in range(S):
        for a in range(N):
            images[i * N + a, 0] = x[a, 0] + shifts[i, 0]
            images[i * N + a, 1] = x[a, 1] + shifts[i, 1]
            images[i * N + a, 2] = x[a, 2] + shifts[i, 2]
    return images


@numba.njit(cache=True)
def make_cells(points, h):
    """
    `points` (N, 3) sorted into cells at least `h` Å wide, as the compiled searches take
    them: every point within `h` of a position lies in the 27 cells around it. Returns
    (points, lo, h, shape, head, following): the first point of each cell, and the next
    point in the cell of each point.
    """
    lo, span = np.zeros(3), np.zeros(3)
    if points.shape[0]:
        for ax in range(3):
            lo[ax] = points[:, ax].min()
            span[ax] = points[:, ax].max() - lo[ax]
    h = max(h, span.max() / 128.0)   # a stray point must not ask for a huge grid
    shape = np.empty(3, dtype=np.int64)
    for ax in range(3):
        shape[ax] = int(span[ax] / h) + 1
    head = np.full(shape[0] * shape[1] * shape[2], -1, dtype=np.int64)
    following = np.empty(points.shape[0], dtype=np.int64)
    for p in range(points.shape[0]):
        c0 = int((points[p, 0] - lo[0]) / h)
        c1 = int((points[p, 1] - lo[1]) / h)
        c2 = int((points[p, 2] - lo[2]) / h)
        c = (c0 * shape[1] + c1) * shape[2] + c2
        following[p] = head[c]
        head[c] = p
    return points, lo, h, shape, head, following


@numba.njit(cache=True)
def closest_squared(x, y, z, closest, grid):
    """The smaller of `closest` and the squared distance of (x, y, z) to the points of `grid`."""
    points, lo, h, shape, head, following = grid
    c0, c1, c2 = int(np.floor((x - lo[0]) / h)), int(np.floor((y - lo[1]) / h)), int(np.floor((z - lo[2]) / h))
    for i in range(max(c0 - 1, 0), min(c0 + 2, shape[0])):
        for j in range(max(c1 - 1, 0), min(c1 + 2, shape[1])):
            for k in range(max(c2 - 1, 0), min(c2 + 2, shape[2])):
                a = head[(i * shape[1] + j) * shape[2] + k]
                while a >= 0:
                    dx, dy, dz = x - points[a, 0], y - points[a, 1], z - points[a, 2]
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < closest:
                        closest = d2
                    a = following[a]
    return closest


@numba.njit(cache=True)
def _turn(ux, uy, uz, phi):
    """The rows of the rotation by `phi` about the unit axis (ux, uy, uz), element by element."""
    c, si = math.cos(phi), math.sin(phi)
    t = 1.0 - c
    return (c + t * ux * ux, t * ux * uy - si * uz, t * ux * uz + si * uy,
            t * ux * uy + si * uz, c + t * uy * uy, t * uy * uz - si * ux,
            t * ux * uz - si * uy, t * uy * uz + si * ux, c + t * uz * uz)


@numba.njit(cache=True)
def _best_spin(x0, center, axis, thetas, reference, follow, best, core, neighbors, bound):
    """
    Among spins of x0 about `axis` through `center`, the first that keeps the closest contact
    with `core` and `neighbors` farthest, if beyond `best`: (spin, contact), a contact being
    `bound` at most. A spin stops at the first atom that rules it out. The spins are
    `reference` + `thetas`, the reference following the best spin found if `follow`.
    """
    ux, uy, uz = axis[0], axis[1], axis[2]
    rel = x0 - center
    best2, spin = best * best if best > 0.0 else -1.0, reference
    for s in range(thetas.shape[0]):
        theta = (reference + thetas[s]) % (2.0 * math.pi)
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = _turn(ux, uy, uz, theta)
        closest = bound * bound
        for a in range(rel.shape[0]):
            vx, vy, vz = rel[a, 0], rel[a, 1], rel[a, 2]
            x = r00 * vx + r01 * vy + r02 * vz + center[0]
            y = r10 * vx + r11 * vy + r12 * vz + center[1]
            z = r20 * vx + r21 * vy + r22 * vz + center[2]
            closest = closest_squared(x, y, z, closest, core)
            closest = closest_squared(x, y, z, closest, neighbors)
            if closest <= best2:
                break
        if closest > best2:
            best2, spin = closest, theta
            if follow:
                reference = theta
    return spin, math.sqrt(best2) if best2 > 0.0 else -math.inf


@numba.njit(cache=True)
def _closest(x, core, neighbors, bound):
    closest = bound * bound
    for a in range(x.shape[0]):
        closest = closest_squared(x[a, 0], x[a, 1], x[a, 2], closest, core)
        closest = closest_squared(x[a, 0], x[a, 1], x[a, 2], closest, neighbors)
    return math.sqrt(closest)


@numba.njit(cache=True)
def _nearest(x, grid, bound):
    """Distance of each atom of `x` to the closest point of `grid`, inf beyond `bound`."""
    out = np.empty(x.shape[0])
    for a in range(x.shape[0]):
        closest = closest_squared(x[a, 0], x[a, 1], x[a, 2], bound * bound, grid)
        out[a] = math.sqrt(closest) if closest < bound * bound else np.inf
    return out


@numba.njit(cache=True)
def _min_distance(x, n_poses, grid, image_of, blocked, bound):
    """
    Per pose, `x` holding `n_poses` of them one after another: the distance to the closest
    point of `grid` whose original (`image_of`) is not `blocked`, inf beyond `bound`.
    """
    points, lo, h, shape, head, following = grid
    m = x.shape[0] // n_poses
    out = np.empty(n_poses)
    for s in range(n_poses):
        closest = bound * bound
        for p in range(s * m, (s + 1) * m):
            c0 = int(np.floor((x[p, 0] - lo[0]) / h))
            c1 = int(np.floor((x[p, 1] - lo[1]) / h))
            c2 = int(np.floor((x[p, 2] - lo[2]) / h))
            for i in range(max(c0 - 1, 0), min(c0 + 2, shape[0])):
                for j in range(max(c1 - 1, 0), min(c1 + 2, shape[1])):
                    for k in range(max(c2 - 1, 0), min(c2 + 2, shape[2])):
                        a = head[(i * shape[1] + j) * shape[2] + k]
                        while a >= 0:
                            if not blocked[image_of[a]]:
                                dx, dy, dz = x[p, 0] - points[a, 0], x[p, 1] - points[a, 1], x[p, 2] - points[a, 2]
                                d2 = dx * dx + dy * dy + dz * dz
                                if d2 < closest:
                                    closest = d2
                            a = following[a]
        out[s] = math.sqrt(closest) if closest < bound * bound else np.inf
    return out


@numba.njit(cache=True)
def _contacts(x, owner, searched, n_ligands, core, shifts, cutoff, reach):
    """
    Closest contact of every ligand in `searched` with `core` and with the other ligands,
    their images at `shifts` included, and the ligands of each atom pair closer than `cutoff`.
    Contacts count up to `reach`; a ligand not searched gets inf. x (N, 3) holds all atoms,
    `owner` the ligand of each.
    """
    N = x.shape[0]
    points, lo, h, shape, head, following = make_cells(_images(x, shifts), reach)
    closest = np.full(n_ligands, np.inf)
    for l in range(n_ligands):
        if searched[l]:
            closest[l] = reach * reach
    cutoff2 = cutoff * cutoff
    n = 0
    first, second = np.empty(1024, dtype=np.int64), np.empty(1024, dtype=np.int64)
    for p in range(N):
        op = owner[p]
        if not searched[op]:
            continue
        ax, ay, az = x[p, 0], x[p, 1], x[p, 2]
        best = closest_squared(ax, ay, az, closest[op], core)
        c0, c1, c2 = int(np.floor((ax - lo[0]) / h)), int(np.floor((ay - lo[1]) / h)), int(np.floor((az - lo[2]) / h))
        for i in range(max(c0 - 1, 0), min(c0 + 2, shape[0])):
            for j in range(max(c1 - 1, 0), min(c1 + 2, shape[1])):
                for k in range(max(c2 - 1, 0), min(c2 + 2, shape[2])):
                    b = head[(i * shape[1] + j) * shape[2] + k]
                    while b >= 0:
                        oq = owner[b % N]
                        if oq != op:
                            dx, dy, dz = ax - points[b, 0], ay - points[b, 1], az - points[b, 2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 < best:
                                best = d2
                            if d2 < cutoff2:
                                if n == first.shape[0]:
                                    first, second = doubled(first), doubled(second)
                                first[n], second[n] = op, oq
                                n += 1
                        b = following[b]
        closest[op] = best
    for l in range(n_ligands):
        if searched[l]:
            closest[l] = math.sqrt(closest[l])
    return closest, first[:n], second[:n]


@numba.njit(cache=True)
def _over_sites(x, sites, normals, radii, out):
    """
    out[s]: whether pose s of x (S, m, 3) lies over any of `sites` (K, 3): an atom on the
    outer side of the site's plane (unit `normals`) within its radius of the site laterally.
    """
    for s in range(x.shape[0]):
        hit = False
        for k in range(sites.shape[0]):
            for a in range(x.shape[1]):
                dx = x[s, a, 0] - sites[k, 0]
                dy = x[s, a, 1] - sites[k, 1]
                dz = x[s, a, 2] - sites[k, 2]
                h = dx * normals[k, 0] + dy * normals[k, 1] + dz * normals[k, 2]
                if h <= 0.0:
                    continue
                lx = dx - h * normals[k, 0]
                ly = dy - h * normals[k, 1]
                lz = dz - h * normals[k, 2]
                if lx * lx + ly * ly + lz * lz < radii[a] * radii[a]:
                    hit = True
                    break
            if hit:
                break
        out[s] = hit


@numba.njit(cache=True)
def _moved(X, ref, limit):
    """Atoms farther than `limit` from their reference, and whether there are any."""
    stale = np.empty(X.shape[0], dtype=np.bool_)
    limit2 = limit * limit
    found = False
    for a in range(X.shape[0]):
        dx, dy, dz = X[a, 0] - ref[a, 0], X[a, 1] - ref[a, 1], X[a, 2] - ref[a, 2]
        stale[a] = dx * dx + dy * dy + dz * dz > limit2
        found = found or stale[a]
    return stale, found


@numba.njit(cache=True)
def _update_fixed(X, s, ref, atom, partner):
    """
    Verlet list of the fixed atoms within the contact distance + SKIN of each atom, searched
    again for an atom once it has moved SKIN: only it moves.
    """
    stale, found = _moved(X, ref, SKIN)
    if not found:
        return ref, atom, partner
    n = 0
    size = max(2 * atom.shape[0], 256)
    new_atom, new_partner = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64)
    for t in range(atom.shape[0]):
        if not stale[atom[t]]:
            new_atom[n], new_partner[n] = atom[t], partner[t]
            n += 1
    points, lo, h, shape, head, following = s.fixed
    reach = s.contact_distance + SKIN
    reach2 = reach * reach
    for a in range(X.shape[0]):
        if not stale[a]:
            continue
        x, y, z = X[a, 0], X[a, 1], X[a, 2]
        ref[a, 0], ref[a, 1], ref[a, 2] = x, y, z
        if _capped(a, s.links):
            continue
        c0, c1, c2 = int(np.floor((x - lo[0]) / h)), int(np.floor((y - lo[1]) / h)), int(np.floor((z - lo[2]) / h))
        for i in range(max(c0 - 1, 0), min(c0 + 2, shape[0])):
            for j in range(max(c1 - 1, 0), min(c1 + 2, shape[1])):
                for k in range(max(c2 - 1, 0), min(c2 + 2, shape[2])):
                    b = head[(i * shape[1] + j) * shape[2] + k]
                    while b >= 0:
                        dx, dy, dz = x - points[b, 0], y - points[b, 1], z - points[b, 2]
                        if dx * dx + dy * dy + dz * dz <= reach2:
                            if n == new_atom.shape[0]:
                                new_atom, new_partner = doubled(new_atom), doubled(new_partner)
                            new_atom[n], new_partner[n] = a, b
                            n += 1
                        b = following[b]
    return ref, new_atom[:n], new_partner[:n]


@numba.njit(cache=True)
def _update_own(X, s, ref, pa, pb, limit):
    """
    Verlet lists of the pairs of each pose four or more bonds apart within their limit +
    SKIN, searched again for a pose once one of its atoms has moved SKIN / 2.
    """
    stale_atom, found = _moved(X, ref, 0.5 * SKIN)
    if not found:
        return ref, pa, pb, limit
    L = s.atom_start.shape[0] - 1
    stale = np.zeros(L, dtype=np.bool_)
    for a in range(X.shape[0]):
        if stale_atom[a]:
            stale[s.owner[a]] = True
    m = 0
    size = max(2 * pa.shape[0], 256)
    new_pa, new_pb, new_limit = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64), np.empty(size)
    for t in range(pa.shape[0]):
        if not stale[s.owner[pa[t]]]:
            new_pa[m], new_pb[m], new_limit[m] = pa[t], pb[t], limit[t]
            m += 1
    for l in range(L):
        if not stale[l]:
            continue
        a0 = s.atom_start[l]
        for a in range(a0, s.atom_start[l + 1]):
            ref[a, 0], ref[a, 1], ref[a, 2] = X[a, 0], X[a, 1], X[a, 2]
        model = s.pose_model[l]
        for t in range(s.far_start[model], s.far_start[model + 1]):
            a, b = a0 + s.far_a[t], a0 + s.far_b[t]
            dx, dy, dz = X[a, 0] - X[b, 0], X[a, 1] - X[b, 1], X[a, 2] - X[b, 2]
            r = s.far_limit[t] + SKIN
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < r * r and d2 > 1e-18:
                if m == new_pa.shape[0]:
                    new_pa, new_pb, new_limit = doubled(new_pa), doubled(new_pb), doubled(new_limit)
                new_pa[m], new_pb[m], new_limit[m] = a, b, s.far_limit[t]
                m += 1
    return ref, new_pa[:m], new_pb[:m], new_limit[:m]


@numba.njit(cache=True)
def _update_mutual(X, s, ref, mp, mq, image):
    """
    Verlet list of the contacts between atoms of different poses, the first against the
    image in `shifts` of the second, searched again for an atom once it has moved SKIN / 3:
    the other may have moved as far on each side of its own reference since.
    """
    if s.atom_start.shape[0] < 3:
        return ref, mp, mq, image
    stale, found = _moved(X, ref, SKIN / 3.0)
    if not found:
        return ref, mp, mq, image

    N = X.shape[0]
    n = 0
    size = max(2 * mp.shape[0], 256)
    new_p, new_q, new_image = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64)
    for t in range(mp.shape[0]):
        if not stale[mp[t]] and not stale[mq[t]]:
            new_p[n], new_q[n], new_image[n] = mp[t], mq[t], image[t]
            n += 1

    reach = s.contact_distance + SKIN
    points, lo, h, shape, head, following = make_cells(_images(X, s.shifts), reach)
    reach2 = reach * reach
    for p in range(N):
        if not stale[p]:
            continue
        op = s.owner[p]
        x, y, z = X[p, 0], X[p, 1], X[p, 2]
        ref[p, 0], ref[p, 1], ref[p, 2] = x, y, z
        if _capped(p, s.links):
            continue
        c0, c1, c2 = int(np.floor((x - lo[0]) / h)), int(np.floor((y - lo[1]) / h)), int(np.floor((z - lo[2]) / h))
        for i in range(max(c0 - 1, 0), min(c0 + 2, shape[0])):
            for j in range(max(c1 - 1, 0), min(c1 + 2, shape[1])):
                for k in range(max(c2 - 1, 0), min(c2 + 2, shape[2])):
                    b = head[(i * shape[1] + j) * shape[2] + k]
                    while b >= 0:
                        q = b % N
                        # a pair of two searched atoms is found from both sides: keep it once
                        if op != s.owner[q] and (not stale[q] or p < q) and not _capped(q, s.links) \
                                and not _bonding(p, q, s.links):
                            dx, dy, dz = x - points[b, 0], y - points[b, 1], z - points[b, 2]
                            if dx * dx + dy * dy + dz * dz <= reach2:
                                if n == new_p.shape[0]:
                                    new_p, new_q, new_image = doubled(new_p), doubled(new_q), doubled(new_image)
                                new_p[n], new_q[n], new_image[n] = p, q, b // N
                                n += 1
                        b = following[b]
    return ref, new_p[:n], new_q[:n], new_image[:n]


@numba.njit(cache=True)
def _fixed_terms(x, atom, partner, points, contact_distance, k, G):
    """Soft-sphere penalty of the listed (atom, fixed point) pairs within `contact_distance`;
    adds its gradient to G. Returns (E, closest contact)."""
    E = 0.0
    min_distance = np.inf
    for t in range(atom.shape[0]):
        a, b = atom[t], partner[t]
        dx = x[a, 0] - points[b, 0]
        dy = x[a, 1] - points[b, 1]
        dz = x[a, 2] - points[b, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d < contact_distance and d > 1e-9:
            min_distance = min(min_distance, d)
            over = contact_distance - d
            E += k * over * over
            c = -2.0 * k * over / d
            G[a, 0] += c * dx
            G[a, 1] += c * dy
            G[a, 2] += c * dz
    return E, min_distance


@numba.njit(cache=True)
def _own_terms(x, pa, pb, limit, k, G):
    """Soft-sphere penalty of the listed pairs of one pose within their `limit`; adds its
    gradient to G. Returns (E, how many are within it)."""
    E = 0.0
    hits = 0
    for t in range(pa.shape[0]):
        a, b = pa[t], pb[t]
        dx = x[a, 0] - x[b, 0]
        dy = x[a, 1] - x[b, 1]
        dz = x[a, 2] - x[b, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d < limit[t] and d > 1e-9:
            hits += 1
            over = limit[t] - d
            E += k * over * over
            c = -2.0 * k * over / d
            G[a, 0] += c * dx
            G[a, 1] += c * dy
            G[a, 2] += c * dz
            G[b, 0] -= c * dx
            G[b, 1] -= c * dy
            G[b, 2] -= c * dz
    return E, hits


@numba.njit(cache=True)
def _mutual_terms(x, mp, mq, image, shifts, contact_distance, k, G):
    """Soft-sphere penalty of the listed pairs of different poses, within `contact_distance`;
    adds its gradient to G. Returns (E, closest contact)."""
    E = 0.0
    min_distance = np.inf
    for t in range(mp.shape[0]):
        a, b, i = mp[t], mq[t], image[t]
        dx = x[a, 0] - x[b, 0] - shifts[i, 0]
        dy = x[a, 1] - x[b, 1] - shifts[i, 1]
        dz = x[a, 2] - x[b, 2] - shifts[i, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d < contact_distance and d > 1e-9:
            min_distance = min(min_distance, d)
            over = contact_distance - d
            E += k * over * over
            c = -2.0 * k * over / d
            G[a, 0] += c * dx
            G[a, 1] += c * dy
            G[a, 2] += c * dz
            G[b, 0] -= c * dx
            G[b, 1] -= c * dy
            G[b, 2] -= c * dz
    return E, min_distance


@numba.njit(cache=True)
def _inside_terms(x, lo, hi, k, links, G):
    """Penalty of the atoms inside the box from `lo` to `hi`, but the caps of `links`, each
    pushed out through its nearest face; adds its gradient to G. Returns E."""
    E = 0.0
    for a in range(x.shape[0]):
        if _capped(a, links):
            continue
        depth, axis, sign = np.inf, 0, 1.0
        for ax in range(3):
            below, above = x[a, ax] - lo[ax], hi[ax] - x[a, ax]
            if below < depth:
                depth, axis, sign = below, ax, 1.0
            if above < depth:
                depth, axis, sign = above, ax, -1.0
        if depth > 0.0:
            E += k * depth * depth
            G[a, axis] += 2.0 * k * depth * sign
    return E


@numba.njit(cache=True)
def _times(a, b):
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]
    return out


@numba.njit(cache=True)
def _matrix(a, b, c, d, e, f, g, h, i):
    m = np.empty((3, 3))
    m[0, 0], m[0, 1], m[0, 2] = a, b, c
    m[1, 0], m[1, 1], m[1, 2] = d, e, f
    m[2, 0], m[2, 1], m[2, 2] = g, h, i
    return m


@numba.njit(cache=True)
def _rotation(a, b, c):
    """R = Rz(c) Ry(b) Rx(a) and its derivatives with respect to a, b, c, each (3, 3)."""
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = _matrix(1.0, 0.0, 0.0, 0.0, ca, -sa, 0.0, sa, ca)
    Ry = _matrix(cb, 0.0, sb, 0.0, 1.0, 0.0, -sb, 0.0, cb)
    Rz = _matrix(cc, -sc, 0.0, sc, cc, 0.0, 0.0, 0.0, 1.0)
    dRx = _matrix(0.0, 0.0, 0.0, 0.0, -sa, -ca, 0.0, ca, -sa)
    dRy = _matrix(-sb, 0.0, cb, 0.0, 0.0, 0.0, -cb, 0.0, -sb)
    dRz = _matrix(-sc, -cc, 0.0, cc, -sc, 0.0, 0.0, 0.0, 0.0)
    RzRy = _times(Rz, Ry)
    return _times(RzRy, Rx), (_times(RzRy, dRx), _times(_times(Rz, dRy), Rx), _times(_times(dRz, Ry), Rx))


@numba.njit(cache=True)
def forward(P, s):
    """
    Coordinates X (N, 3) for parameters P, with the atoms relative to their center before
    the rigid move, Xt, and the derivatives of each pose's rotation, dR (L, 3, 3, 3).

    Torsions apply deepest first: each later (shallower) one moves its side, deeper bonds
    included, rigidly, so the derivative of every torsion is a rotation about its bond as
    it lies in the final coordinates. The Euler angles turn about the lab axes, or about
    the rows of the pose's frame.
    """
    X = s.X0.copy()
    Xt = np.empty_like(X)
    L = s.atom_start.shape[0] - 1
    dR = np.empty((L, 3, 3, 3))
    for l in range(L):
        p0, t0 = s.param_start[l], s.torsion_start[l]
        for b in range(t0, s.torsion_start[l + 1]):
            phi = P[p0 + 6 + b - t0]
            if phi == 0.0:
                continue
            i, j = s.bonds[b, 0], s.bonds[b, 1]
            px, py, pz = X[i, 0], X[i, 1], X[i, 2]
            ux, uy, uz = X[j, 0] - px, X[j, 1] - py, X[j, 2] - pz
            r = math.sqrt(ux * ux + uy * uy + uz * uz)
            ux, uy, uz = ux / r, uy / r, uz / r
            r00, r01, r02, r10, r11, r12, r20, r21, r22 = _turn(ux, uy, uz, phi)
            for m in range(s.moving_start[b], s.moving_start[b + 1]):
                a = s.moving[m]
                vx, vy, vz = X[a, 0] - px, X[a, 1] - py, X[a, 2] - pz
                X[a, 0] = r00 * vx + r01 * vy + r02 * vz + px
                X[a, 1] = r10 * vx + r11 * vy + r12 * vz + py
                X[a, 2] = r20 * vx + r21 * vy + r22 * vz + pz
        R, d = _rotation(P[p0 + 3], P[p0 + 4], P[p0 + 5])
        if s.framed[l]:
            F = s.frame[l]
            Ft = F.T
            R = _times(_times(Ft, R), F)
            d = _times(_times(Ft, d[0]), F), _times(_times(Ft, d[1]), F), _times(_times(Ft, d[2]), F)
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    dR[l, k, i, j] = d[k][i, j]
        cx, cy, cz = s.center[l, 0], s.center[l, 1], s.center[l, 2]
        tx, ty, tz = cx + P[p0], cy + P[p0 + 1], cz + P[p0 + 2]
        for a in range(s.atom_start[l], s.atom_start[l + 1]):
            vx, vy, vz = X[a, 0] - cx, X[a, 1] - cy, X[a, 2] - cz
            Xt[a, 0], Xt[a, 1], Xt[a, 2] = vx, vy, vz
            X[a, 0] = R[0, 0] * vx + R[0, 1] * vy + R[0, 2] * vz + tx
            X[a, 1] = R[1, 0] * vx + R[1, 1] * vy + R[1, 2] * vz + ty
            X[a, 2] = R[2, 0] * vx + R[2, 1] * vy + R[2, 2] * vz + tz
    return X, Xt, dR


@numba.njit(cache=True)
def _gradient(P, s, X, Xt, dR, G):
    """Chain rule from the Cartesian gradient G (N, 3) to the parameters; adds the torsion term."""
    g = np.zeros(P.shape[0])
    for l in range(s.atom_start.shape[0] - 1):
        p0, t0 = s.param_start[l], s.torsion_start[l]
        for a in range(s.atom_start[l], s.atom_start[l + 1]):
            gx, gy, gz = G[a, 0], G[a, 1], G[a, 2]
            vx, vy, vz = Xt[a, 0], Xt[a, 1], Xt[a, 2]
            g[p0] += gx
            g[p0 + 1] += gy
            g[p0 + 2] += gz
            for k in range(3):
                g[p0 + 3 + k] += (gx * (dR[l, k, 0, 0] * vx + dR[l, k, 0, 1] * vy + dR[l, k, 0, 2] * vz)
                                  + gy * (dR[l, k, 1, 0] * vx + dR[l, k, 1, 1] * vy + dR[l, k, 1, 2] * vz)
                                  + gz * (dR[l, k, 2, 0] * vx + dR[l, k, 2, 1] * vy + dR[l, k, 2, 2] * vz))
        for b in range(t0, s.torsion_start[l + 1]):
            i, j = s.bonds[b, 0], s.bonds[b, 1]
            px, py, pz = X[i, 0], X[i, 1], X[i, 2]
            ux, uy, uz = X[j, 0] - px, X[j, 1] - py, X[j, 2] - pz
            r = math.sqrt(ux * ux + uy * uy + uz * uz)
            tx = ty = tz = 0.0
            for m in range(s.moving_start[b], s.moving_start[b + 1]):
                a = s.moving[m]
                rx, ry, rz = X[a, 0] - px, X[a, 1] - py, X[a, 2] - pz
                gx, gy, gz = G[a, 0], G[a, 1], G[a, 2]
                tx += ry * gz - rz * gy
                ty += rz * gx - rx * gz
                tz += rx * gy - ry * gx
            q = p0 + 6 + b - t0
            g[q] = (ux * tx + uy * ty + uz * tz) / r + s.w_torsion * math.sin(P[q])
    return g


@numba.njit(cache=True)
def evaluate(P, s, lists):
    """
    Energy and gradient at parameters P: anchor, contact, intramolecular, inside-the-box,
    pull and link terms, and a mild pull of the torsions to the conformer. `lists` are the
    Verlet lists of the last call, returned updated. Returns (E, gradient, closest contact
    with fixed atoms and other poses, whether any pair of a pose is within its limit, worst
    anchor RMSD of a pose, farthest link point from its partner, lists).
    """
    X, Xt, dR = forward(P, s)
    ref_fixed, atom, partner, ref_own, pa, pb, limit, ref_mutual, mp, mq, image = lists
    ref_fixed, atom, partner = _update_fixed(X, s, ref_fixed, atom, partner)
    ref_own, pa, pb, limit = _update_own(X, s, ref_own, pa, pb, limit)
    ref_mutual, mp, mq, image = _update_mutual(X, s, ref_mutual, mp, mq, image)

    G = np.zeros_like(X)
    L = s.atom_start.shape[0] - 1
    sq = np.zeros(L)
    E = 0.0
    for g in range(s.group_pose.shape[0]):
        a0, a1 = s.group_start[g], s.group_start[g + 1]
        n = a1 - a0
        mx = my = mz = 0.0
        for t in range(a0, a1):
            a = s.group_atoms[t]
            mx += X[a, 0]
            my += X[a, 1]
            mz += X[a, 2]
        dx, dy, dz = mx / n - s.targets[g, 0], my / n - s.targets[g, 1], mz / n - s.targets[g, 2]
        d2 = dx * dx + dy * dy + dz * dz
        E += s.k_anchor * d2
        sq[s.group_pose[g]] += d2
        c = 2.0 * s.k_anchor / n
        for t in range(a0, a1):
            a = s.group_atoms[t]
            G[a, 0] += c * dx
            G[a, 1] += c * dy
            G[a, 2] += c * dz

    e, min_distance = _fixed_terms(X, atom, partner, s.fixed[0], s.contact_distance, s.k_contact, G)
    E += e
    e, hits = _own_terms(X, pa, pb, limit, s.k_intra, G)
    E += e
    e, d = _mutual_terms(X, mp, mq, image, s.shifts, s.contact_distance, s.k_contact, G)
    E += e
    min_distance = min(min_distance, d)
    if s.confined:
        E += _inside_terms(X, s.lo, s.hi, s.k_inside, s.links, G)
    for t in range(s.pull_atoms.shape[0]):
        a = s.pull_atoms[t]
        E -= s.k_pull * ((X[a, 0] - s.center[0, 0]) * s.pull_direction[0]
                         + (X[a, 1] - s.center[0, 1]) * s.pull_direction[1]
                         + (X[a, 2] - s.center[0, 2]) * s.pull_direction[2])
        G[a, 0] -= s.k_pull * s.pull_direction[0]
        G[a, 1] -= s.k_pull * s.pull_direction[1]
        G[a, 2] -= s.k_pull * s.pull_direction[2]
    link_miss = 0.0
    for t in range(s.links.shape[0]):
        c, o, q, i, w = s.links[t, 0], s.links[t, 1], s.links[t, 2], s.links[t, 3], s.link_scale[t]
        dx = (1.0 - w) * X[o, 0] + w * X[c, 0] - X[q, 0] - s.shifts[i, 0]
        dy = (1.0 - w) * X[o, 1] + w * X[c, 1] - X[q, 1] - s.shifts[i, 1]
        dz = (1.0 - w) * X[o, 2] + w * X[c, 2] - X[q, 2] - s.shifts[i, 2]
        d2 = dx * dx + dy * dy + dz * dz
        E += s.k_link * d2
        link_miss = max(link_miss, math.sqrt(d2))
        g = 2.0 * s.k_link
        G[o, 0] += g * (1.0 - w) * dx
        G[o, 1] += g * (1.0 - w) * dy
        G[o, 2] += g * (1.0 - w) * dz
        G[c, 0] += g * w * dx
        G[c, 1] += g * w * dy
        G[c, 2] += g * w * dz
        G[q, 0] -= g * dx
        G[q, 1] -= g * dy
        G[q, 2] -= g * dz

    grad = _gradient(P, s, X, Xt, dR, G)
    worst = 0.0
    for l in range(L):
        for q in range(s.param_start[l] + 6, s.param_start[l + 1]):
            E += s.w_torsion * (1.0 - math.cos(P[q]))
        worst = max(worst, math.sqrt(sq[l] / s.n_groups[l]))
    return (E, grad, min_distance, hits > 0, worst, link_miss,
            (ref_fixed, atom, partner, ref_own, pa, pb, limit, ref_mutual, mp, mq, image))
