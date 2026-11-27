"""Knot-balanced rectangular patches with a common spline on each seam."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from . import capi
from .bspline import BSplinePatches


def _axes(face):
    a = face // 2
    axes = ((a + 1) % 3, (a + 2) % 3)
    return axes[::-1] if face % 2 else axes


def _locate(knots, frame):
    """Inverse of patch_reference, including the caps of the axial chart."""
    if frame is None:
        cube = knots / np.abs(knots).max(axis=1)[:, None]
    else:
        north, tangent, bitangent = frame[:3], frame[3:6], frame[6:9]
        inv, rn, rs = frame[9:]
        xy = knots @ tangent + 1j * (knots @ bitangent)
        z = knots @ north
        # Two equivalent homogeneous stereographic formulas avoid cancellation
        # at the south pole. They also cover exactly antipodal clusters.
        w = np.where(z >= 0, xy, 1-z).astype(complex)
        d = np.where(z >= 0, 1+z, np.conj(xy)) - inv*w
        cube = np.empty_like(knots)
        wd = w * np.conj(d)
        scale = np.maximum(abs(wd.real), abs(wd.imag))
        top = (abs(d) > 0) & (scale <= rn*abs(d)**2)
        bottom = (~top) & (abs(w) > 0) & (scale <= rs*abs(w)**2)
        side = ~(top | bottom)
        for mask, coord, height in ((top, w[top]/d[top]/rn, 1),
                                    (bottom, np.conj(d[bottom]/w[bottom])/rs, -1)):
            cube[mask] = np.column_stack((coord.real, coord.imag, np.full(len(coord), height)))
        radius = w[side]/d[side]
        direction = radius / np.maximum(abs(radius.real), abs(radius.imag))
        r = abs(direction)
        z = 2*(np.log(abs(radius))+np.log(rs*r))/(np.log(rn*rs)+2*np.log(r))-1
        cube[side] = np.column_stack((direction.real, direction.imag, z))
    axis = np.argmax(abs(cube), axis=1)
    faces = 2*axis + (cube[np.arange(len(cube)), axis] < 0)
    uv = np.empty((len(knots), 2))
    for f in range(6):
        uv[faces == f] = (cube[faces == f][:, _axes(f)] + 1)/2
    return faces, np.clip(uv, 0, 1)


def _full(breaks):
    return np.r_[[breaks[0]]*3, breaks, [breaks[-1]]*3]


def _grid(u, v):
    return np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1)


def _probes(breaks, divisions):
    return np.r_[(breaks[:-1, None] + np.diff(breaks)[:, None] *
                  np.arange(divisions)/divisions).ravel(), breaks[-1]]


def _midpoints(breaks):
    # Snap to the same dyadic grid on parallel seams. Otherwise merging the
    # opposite boundaries' knots interleaves two nearly identical sequences.
    lengths = np.diff(breaks)
    step = np.exp2(np.floor(np.log2(lengths)) - 1)
    return np.rint((breaks[:-1] + lengths/2) / step) * step


def _augmented(breaks):
    h = 1e-4 * np.diff(breaks)[[0, -1]]
    return np.r_[breaks[0]+h[0], breaks, breaks[-1]-h[1]], h


def _derivatives(values, h):
    values[0] = (values[0] - values[1])/h[0]
    values[-1] = (values[-2] - values[-1])/h[1]


@dataclass
class Tile:
    face: int
    box: np.ndarray  # [[u0,u1],[v0,v1]], in increasing cube-axis coordinates
    nodes: np.ndarray
    refinement_depth: int = 0

    def split(self, axis, *, refine=False):
        lo, hi = self.box[axis]
        fraction = .5
        if len(self.nodes) > 1:
            fraction = (np.median(self.nodes[:, axis])-lo)/(hi-lo)
        # Dyadic cuts keep opposite edges and their T-junctions exactly addressable.
        width = hi - lo
        step = 2**(np.floor(np.log2(width)) - 3)
        cut = np.clip(np.rint((lo + fraction*width)/step)*step, lo+width/4, hi-width/4)
        if cut == lo or cut == hi:
            raise ValueError("Cannot resolve coincident reference nodes")
        lower, upper = self.box.copy(), self.box.copy()
        lower[axis, 1] = upper[axis, 0] = cut
        left = self.nodes[:, axis] < cut
        depth = self.refinement_depth + int(refine)
        return Tile(self.face, lower, self.nodes[left], depth), Tile(self.face, upper, self.nodes[~left], depth)


def _can_merge(tile, sample, tolerance):
    """Try a parent with 25 poles (two bicubic children need at least 32)."""
    breaks = [np.linspace(a, b, 3) for a, b in tile.box]
    (u, hu), (v, hv) = map(_augmented, breaks)
    data = sample(tile.face, _grid(u, v))
    _derivatives(data, hu)
    _derivatives(data.swapaxes(0, 1), hv)
    poles = capi.interpolate_cubic(breaks[0], data)
    poles = capi.interpolate_cubic(breaks[1], poles.swapaxes(0, 1)).swapaxes(0, 1)
    u, v = (_probes(t, 8) for t in breaks)
    error = _tensor_eval(poles, breaks, u, v) - sample(tile.face, _grid(u, v))
    return np.linalg.norm(error, axis=-1).max() <= .25*tolerance


def _partition(faces, uv, target, sample, tolerance):
    tiles = []
    stack = [Tile(f, np.array([[0., 1.], [0., 1.]]), uv[faces == f]) for f in reversed(range(6))]
    while stack:
        tile = stack.pop()
        if (len(tile.nodes) <= target or
                (len(tile.nodes) <= 2*target and _can_merge(tile, sample, tolerance))):
            tiles.append(tile)
        else:
            axis = np.argmax(np.diff(tile.box, axis=1))
            stack.extend(reversed(tile.split(axis)))
    return tiles


def _tensor_eval(poles, breaks, u, v):
    a = BSpline.design_matrix(u, _full(breaks[0]), 3)
    b = BSpline.design_matrix(v, _full(breaks[1]), 3)
    return np.stack([a @ p @ b.T for p in poles.transpose(2, 0, 1)], axis=-1)


class _ResolutionLimit(ValueError):
    """The requested position tolerance exceeded the available subdivisions."""


def _fit_curve(sample, breaks, tol, limit):
    """One clamped spline for a connected seam, interpolating all T-junctions."""
    while True:
        t, h = _augmented(breaks)
        data = sample(t)
        _derivatives(data, h)
        spline = BSpline(_full(breaks), capi.interpolate_cubic(breaks, data), 3)
        check = _probes(breaks, 8)
        error = np.linalg.norm(spline(check) - sample(check), axis=1)
        bad = error[:-1].reshape(-1, 8).max(axis=1) > tol
        if not bad.any():
            return breaks, spline
        if len(breaks)-1 + bad.sum() > limit:
            raise _ResolutionLimit(f"Sampled boundary error {error.max():g} exceeds tolerance {tol:g}")
        breaks = np.sort(np.r_[breaks, _midpoints(breaks)[bad]])


def _seams(tiles, sample, tol, max_spans, hints):
    # Group edges by the varying Cartesian cube axis and the two fixed coordinates.
    lines = defaultdict(list)
    per_tile = [[None]*4 for _ in tiles]
    for i, tile in enumerate(tiles):
        u, v = _axes(tile.face)
        for varying, axis in enumerate((u, v)):
            for end in (0, 1):
                corner = np.zeros(3)
                corner[tile.face//2] = 1 - tile.face % 2
                corner[(v, u)[varying]] = tile.box[1-varying, end]
                lines[(axis, *np.delete(corner, axis))].append(
                    (*tile.box[varying], i, varying, end))
    groups = []
    for key, entries in lines.items():
        group = []
        stop = -np.inf
        for entry in sorted(entries):
            if entry[0] > stop and group:
                groups.append((key, group))
                group = []
            group.append(entry)
            stop = max(stop, entry[1])
        groups.append((key, group))
    for key, group in groups:
        endpoints = np.unique([x for edge in group for x in edge[:2]])
        _, _, i, varying, end = group[0]
        tile = tiles[i]
        fixed = tile.box[1-varying, end]

        def line(t):
            uv = np.empty((len(t), 2))
            uv[:, varying] = t
            uv[:, 1-varying] = fixed
            return sample(tile.face, uv)

        extra = [x for x in hints.get(key, ()) if endpoints[0] < x < endpoints[-1]]
        initial = np.union1d(endpoints, extra)
        curve = _fit_curve(line, initial, tol, max_spans*max(8, len(endpoints)-1))
        for lo, hi, i, varying, end in group:
            per_tile[i][2*varying+end] = (*curve, key)
    return per_tile, groups


def _fit_tile(tile, edges, sample, tolerance, max_spans):
    breaks = []
    for axis in (0, 1):
        lo, hi = tile.box[axis]
        knots = np.union1d(edges[2*axis][0], edges[2*axis+1][0])
        breaks.append(knots[(knots >= lo) & (knots <= hi)])
    while True:
        counts = [len(t)-1 for t in breaks]
        if max(counts) > max_spans:
            return None, int(np.argmax(counts))
        (u, hu), (v, hv) = map(_augmented, breaks)
        data = sample(tile.face, _grid(u, v))
        _derivatives(data, hu)
        _derivatives(data.swapaxes(0, 1), hv)
        # Restrict each common curve to this rectangle. Its knots are already
        # present, so interpolation reproduces that restriction exactly.
        for axis, t in enumerate(breaks):
            rows = data if axis == 1 else data.swapaxes(0, 1)
            for end, row in ((0, 1), (1, -2)):
                spline = edges[2*axis+end][1]
                rows[row, 1:-1] = spline(t)
                rows[row, [0, -1]] = spline(t[[0, -1]], nu=1)
        poles = capi.interpolate_cubic(breaks[0], data)
        poles = capi.interpolate_cubic(breaks[1], poles.swapaxes(0, 1)).swapaxes(0, 1)
        u, v = (_probes(t, 6) for t in breaks)
        error = np.linalg.norm(_tensor_eval(poles, breaks, u, v) -
                               sample(tile.face, _grid(u, v)), axis=2)
        peak = error.max()
        # Check the actual reference knots as well as independent off-knot samples.
        if len(tile.nodes):
            bu, bv = [BSpline.design_matrix(tile.nodes[:, d], _full(t), 3).toarray()
                      for d, t in enumerate(breaks)]
            at_nodes = np.einsum("ni,nj,ijd->nd", bu, bv, poles)
            node_error = np.linalg.norm(at_nodes - sample(tile.face, tile.nodes), axis=1)
            peak = max(peak, node_error.max())
        if not np.isfinite(peak):
            raise ValueError("Nonfinite adaptive patch error")
        if peak <= tolerance:
            return (poles, breaks, peak, np.sum(error**2), error.size), None
        # Errors along interpolation lines isolate the coordinate needing detail.
        marked = [error[:-1, ::6].reshape(counts[0], 6, -1).max(axis=(1,2)) > .45*tolerance,
                  error[::6, :-1].reshape(-1, counts[1], 6).max(axis=(0,2)) > .45*tolerance]
        if not any(np.any(m) for m in marked):
            cells = error[:-1,:-1].reshape(counts[0],6,counts[1],6).max(axis=(1,3)) > .45*tolerance
            marked = [cells.any(axis=1), cells.any(axis=0)]
        if len(tile.nodes):
            for axis, t in enumerate(breaks):
                bad = tile.nodes[node_error > tolerance, axis]
                marked[axis][np.clip(np.searchsorted(t, bad)-1, 0, len(t)-2)] = True
        grow = [count + mask.sum() for count, mask in zip(counts, marked)]
        if max(grow) > max_spans:
            return None, int(np.argmax(grow))
        for d, t in enumerate(breaks):
            breaks[d] = np.sort(np.r_[t, _midpoints(t)[marked[d]]])


def _edge_normal(result, axis, end, t):
    poles, breaks = result[:2]
    p = poles if axis == 0 else poles.swapaxes(0, 1)
    along = _full(breaks[axis])
    row = 0 if end == 0 else -1
    tangent = BSpline(along, p[:, row], 3)(t, nu=1)
    across = p[:, 1]-p[:, 0] if end == 0 else p[:, -1]-p[:, -2]
    transverse = BSpline(along, across, 3)(t)
    normal = np.cross(tangent, transverse) if axis == 0 else np.cross(transverse, tangent)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    if not np.all(length > 0):
        raise ValueError("Singular normal on adaptive patch seam")
    return normal / length


def _check_normals(results, groups, edges, hints):
    largest = 0.
    def refine(i, axis, end, points):
        breaks, _, key = edges[i][2*axis+end]
        ids = np.clip(np.searchsorted(breaks, points, side="right")-1, 0, len(breaks)-2)
        hints[key].update(_midpoints(breaks)[ids])
        hints[key].update(breaks)

    for _, group in groups:
        for k, (a,b,i,axis,end) in enumerate(group):
            for c,d,j,other,side in group[k+1:]:
                lo, hi = max(a,c), min(b,d)
                if hi <= lo:
                    continue
                t = np.union1d(results[i][1][axis], results[j][1][other])
                check = _probes(t[(t >= lo) & (t <= hi)], 8)
                n = _edge_normal(results[i], axis, end, check)
                m = _edge_normal(results[j], other, side, check)
                angle = np.arctan2(np.linalg.norm(np.cross(n,m), axis=1), (n*m).sum(axis=1))
                largest = max(largest, angle.max())
                bad = check[angle > np.radians(.24)]
                if len(bad):
                    refine(i, axis, end, bad)
                    # At a corner the transverse derivative is prescribed by
                    # the perpendicular seam, so refine that curve there too.
                    for face, along, side_index in ((i, axis, end), (j, other, side)):
                        coord = results[face][1][along]
                        cross_coord = results[face][1][1-along][0 if side_index == 0 else -1]
                        for extremity in (0, 1):
                            if coord[0 if extremity == 0 else -1] in bad:
                                refine(face, 1-along, extremity, [cross_coord])
    return np.degrees(largest)


def _fit_layout(tiles, sample, tolerance, max_spans, hints):
    # The span limit applies separately to each tile. Allow up to six extra
    # binary cuts for unresolved geometry, including regions with no knots.
    while True:
        edges, groups = _seams(tiles, sample, .25*tolerance, max_spans, hints)
        results = []
        next_tiles = []
        for tile, boundary in zip(tiles, edges):
            result, axis = _fit_tile(tile, boundary, sample, .85*tolerance, max_spans)
            if result is None:
                if tile.refinement_depth >= 6 or max_spans < 4:
                    raise _ResolutionLimit(f"Sampled adaptive patch error exceeds tolerance {tolerance:g} "
                                     f"at {max_spans} spans per axis")
                next_tiles.extend(tile.split(axis, refine=True))
            else:
                next_tiles.append(tile)
                results.append(result)
        if len(next_tiles) != len(tiles):
            tiles = next_tiles
            continue
        return tiles, results, groups, edges


def adaptive_fit(model, chart, tolerance, max_spans, target, faces, uv):
    frame = None if chart is None else chart[0]

    def sample(face, coords):
        q = capi.patch_reference(face, coords.reshape(-1, 2), frame)
        return model.map3(q).reshape((*coords.shape[:-1], 3))

    tiles = _partition(faces, uv, target, sample, tolerance)
    hints = defaultdict(set)
    best = None
    normal_passes = 0
    while True:
        try:
            tiles, results, groups, edges = _fit_layout(tiles, sample, tolerance, max_spans, hints)
        except _ResolutionLimit:
            if best is None:
                raise
            break  # A previous layout already satisfied the requested position tolerance.
        angle = _check_normals(results, groups, edges, hints)
        if best is None or angle < best[0]:
            best = angle, results, tiles
        if angle <= .25 or normal_passes >= 5:
            break
        normal_passes += 1
    angle, results, tiles = best
    knots = [tuple((t-t[0])/(t[-1]-t[0]) for t in r[1]) for r in results]
    return BSplinePatches(tuple(r[0] for r in results), max(r[2] for r in results),
                          np.sqrt(sum(r[3] for r in results)/sum(r[4] for r in results)),
                          angle, knots,
                          [(t.face, t.box, len(t.nodes)) for t in tiles], frame)
