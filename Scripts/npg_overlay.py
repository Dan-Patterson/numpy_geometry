# -*- coding: utf-8 -*-
r"""
-----------
npg_overlay
-----------

Overlay tool. Working with two sets of geometries.  The functions are largely
confined to polygon and polyline objects.

----

Script :
    npg_overlay.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2020-01-13

Purpose
-------
Functions for ...

See Also
--------
None

Notes
-----
p0s = s00[:-1]
p1s = s00[1:]
p2s = c[:-1]
p3s = c[1:]
n = np.arange(len(p0s))
m = np.arange(len(p2s))
z0 = [[intersects(p0s[i], p1s[i], p2s[j], p3s[j]) for i in n] for j in m]
z1 = [[_intersect_(p0s[i], p1s[i], p2s[j], p3s[j]) for i in n] for j in m]
z2 = poly_intersects_poly(s00, c)
z3 = p_intr_p(s00, c)
z4  p_ints_p(s00, c)  **

Timing
z0  150 µs ± 1.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
z1  149 µs ± 1.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
z2  173 µs ± 12.1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
z3  73.4 µs ± 10.6 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
z4  65.3 µs ± 4.08 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

References
----------
`Paul Bourke geometry
<http://paulbourke.net/geometry/pointlineplane/>`_.

"""
# pycodestyle D205 gets rid of that one blank line thing
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0402,E0611,E1136,E1121,R0904,R0914,
# pylint: disable=W0201,W0212,W0221,W0612,W0621,W0105
# pylint: disable=R0902


import sys
from textwrap import dedent
import numpy as np

import npgeom as npg
from npgeom.npg_helpers import (
    compare_geom, line_crosses, radial_sort, crossing_num, pnts_in_poly
)

# ---- optional imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

# noqa: E501
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]  # print this should you need to locate the script

# FLOATS = np.typecodes['AllFloat']
# INTS = np.typecodes['AllInteger']
# NUMS = FLOATS + INTS
# TwoPI = np.pi * 2.0

__all__ = [
    '_in_LBRT_', 'p_ints_p', 'batch_p_int_p',
    '_intersect_', 'intersects',
    'clip_', 'dissolve', 'append_', 'merge_', '_union_', 'union_'
]  # '_poly_intersect_poly',


# ----------------------------------------------------------------------------
# ---- (1) helpers/mini functions
#
def flat(l):
    """Flatten input. Basic flattening but doesn't yield where things are"""
    def _flat(l, r):
        """Recursive flattener."""
        if not isinstance(l[0], (list, np.ndarray, tuple)):  # added [0]
            r.append(l)
        else:
            for i in l:
                r = r + flat(i)
        return r
    return _flat(l, [])


def _in_extent_(pnts, ext):
    """Return points in, or on the line of an extent. See `_in_LBRT_` also.

    Parameters
    ----------
    pnts : array
        An Nx2 array representing point objects.
    extent : array-like
        A 2x2 array, as [[x0, y0], [x1, y1] where the first pair is the
        left-bottom and the second pair is the right-top coordinate.
    """
    LB, RT = ext
    comp = np.logical_and(LB <= pnts, pnts <= RT)  # using <= and <=
    idx = np.logical_and(comp[..., 0], comp[..., 1])
    return pnts[idx]


def _in_LBRT_(pnts, extent):
    """Return points in, or on the line of an extent. See `_in_extent_` also.

    Parameters
    ----------
    pnts : array
        An Nx2 array representing point objects.
    extent : array-like
        A 1x4, as [x0, y0, x1, y1] where the first tw is the left-bottom
        and the second two are the right-top coordinate.
    """
    # if hasattr(pnts, 'XY'):
    if pnts.__class__.__name__ == "Geo":
        pnts = pnts.XY
    LB, RT = extent[:2], extent[2:]
    return np.any(LB < pnts) & np.any(pnts <= RT)

# c0 = np.logical_and(yp[i] <= y, y < yp[j])
# c1 = np.logical_and(yp[j] <= y, y < yp[i]) 
# np.logical_or(c0, c1)
# ((yp[j] <= y) && (y < yp[i]))) &&
#             (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
#           c = !c;

# ----------------------------------------------------------------------------
# ---- (2) clip, intersect geometry
#  `p_ints_p` is the main function
#  `batch_p_int_p` use this to batch intersect multiple polygons as input
#  and intersectors.

def dist_to_segment(x1, y1, x2, y2, x3, y3): # x3,y3 is the point
    """
    `<https://stackoverflow.com/questions/849211/shortest-distance-between
    -a-point-and-a-line-segment/2233538#2233538>`_.
    """
    px = x2-x1
    py = y2-y1
    norm = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = (dx*dx + dy*dy)**.5
    return dist


def ray_tracing_numpy(x, y, poly):
    """
    `<https://stackoverflow.com/questions/36399381/whats-the-fastest-way
    -of-checking-if-a-point-is-inside-a-polygon-in-python>`_.
    """
    n = len(poly)
    inside = np.zeros(len(x),np.bool_)
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        idx = np.nonzero((y > min(p1y, p2y)) & (y <= max(p1y,p2y))
                          & (x <= max(p1x,p2x)))[0]
        if p1y != p2y:
            xints = (y[idx]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
        if p1x == p2x:
            inside[idx] = ~inside[idx]
        else:
            idxx = idx[x[idx] <= xints]
            inside[idxx] = ~inside[idxx]    

        p1x,p1y = p2x,p2y
    return inside    

def ray_tracing_mult(x, y,poly):
    return [ray_tracing_numpy(xi, yi, poly[:-1,:]) for xi, yi in zip(x, y)]


def points_in_polygons(pnts, geo):
    """Points in polygon test

    Bourkes implementation.
    [[p3.bits, 2, 'red', '.', True ], [psrt, 0, 'black', 'o', False]]
    plot_mixed(data, title="Points in Polygons", invert_y=False, ax_lbls=None)
    """
    def _pip_(x, y, poly):
        """Point in polygon."""
        n = len(poly)
        inside = False
        # cnt = 0
        p1x, p1y = poly[0]
        if [x, y] in poly:
            return True
        cnt = 0
        for i in range(n+1):
            p2x, p2y = poly[i % 2]
            if y >= min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x)/(p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                            # cnt += 1
            p1x, p1y = p2x, p2y
        return inside
        # if cnt % 2 == 0:
        #     return False
        # return True

    def in_extent(pnts, bit_ext):
        """Return pnts within a bit extent"""
        LB = bit_ext[:2]
        RT = bit_ext[2:]
        comp = np.logical_and(LB < pnts, pnts <= RT)  # using <= and <=
        idx = np.logical_and(comp[..., 0], comp[..., 1])
        return idx
    # ----
    # Work section.
    final = []
    in_outs = []
    ids = np.arange(pnts.shape[0])
    polys = geo.bits
    bit_exts = geo.extents('bit')
    #in_ext = []
    for poly in polys:
        #idx = in_extent(pnts, bit_exts[i])
        #good = ids[idx]
        #in_ext.append(good)
        #p_in = pnts[idx]
        in_out = [_pip_(i[0], i[1], poly) for i in pnts]  # p_in]
        #pnts = p_out
        in_outs.append(ids[in_out])
        final.append(pnts[in_out])  # p_in[in_out])
    return in_outs, final  # in_ext,
#     return pnts[in_out]


def p_ints_p(poly0, poly1):
    """Intersect two polygons.  Used in clipping.

    Parameters
    ----------
    poly0, poly1 : ndarrays
        Two polygons, The one with more vertices first. the order is switched
        if it is not.

    Returns
    -------
    Points of intersection or None.

    Notes
    -----
    Using Paul Bourke`s notation.

    Intersection point of two line segments in 2 dimensions, 1989

    `<http://paulbourke.net/geometry/pointlineplane/>`_.
    `<http://paulbourke.net/geometry/polygonmesh/>`_.

    | line a : p0-->p1
    | line b : p2-->p3

    >>> d_nom = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    >>> a_num = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2) ==>
    >>> b_num = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2) ==> u_b
    >>> u_a = a_num/d_nom  # if d_nom != 0
    >>> u_b = b_num/d_nom

    if 0 <= u_a, u_b <=1 then the intersection is on both segments
    """
    if poly1.shape[0] > poly0.shape[0]:
        poly0, poly1 = poly1, poly0
    p01 = poly0[1:] - poly0[:-1]
    p01_x, p01_y = p01.T
    p23 = poly1[1:] - poly1[:-1]
    p23_x, p23_y = p23.T
    p02 = poly0[:-1] - poly1[:-1][:, None]
    d_nom = (p23_y[:, None] * p01_x) - (p23_x[:, None] * p01_y)
    b_num = p01_x * p02[..., 1] - p01_y * p02[..., 0]
    a_num = p23_x[:, None] * p02[..., 1] - p23_y[:, None] * p02[..., 0]
    with np.errstate(all='ignore'):  # divide='ignore', invalid='ignore'):
        a_n = a_num/d_nom
        b_n = b_num/d_nom
        z0 = np.logical_and(0. <= a_n, a_n <= 1.)
        z1 = np.logical_and(0. <= b_n, b_n <= 1.)
        bth = z0 & z1
        xs = a_n * p01_x + poly0[:-1][:, 0]
        ys = a_n * p01_y + poly0[:-1][:, 1]
    xs = xs[bth]
    ys = ys[bth]
    if xs.size > 0:
        final = np.array(list(zip(xs, ys)))
        return np.unique(final, axis=0)
    return


def clip_(geo, c):
    """Do the work"""
    out = []
    bounds = dissolve(geo)
    in_0 = points_in_polygon(bounds, c)
    in_1 = points_in_polygon(c, bounds)
    x_sect0 = p_ints_p(bounds, c)
    # x_sect1 = p_ints_p(bounds, c[::-1])
    final = [np.atleast_2d(i) for i in (in_0, in_1, x_sect0) if i.size > 0]
    final = radial_sort(np.vstack(final))
    return in_0, in_1, x_sect0, final  # x_sect1,


def batch_p_int_p(polys, overlays):
    """Batch `p_ints_p`.

    Parameters
    ----------
    polys : list of ndarrays or a Geo Array
        The geometry to intersect.
    overlay : polygon/polyline
        The intersecting geometry which is used to examine all the input polys.
    p_ispolygon, o_ispolygon : boolean
        True indicates that the polys or overlay features are polygons.  If
        False, than either can be a polyline with at least 2 points.

    Returns the points of intersection between the input poly features and the
    overlay feature(s).
    """
    overlays = np.asarray(overlays)
    polys = np.asarray(polys)
    if overlays.ndim == 2:
        overlays = [overlays]
    elif overlays.dtype.kind == 'O':
        overlays = overlays
    if polys.dtype.kind == 'O':
        polys = polys
    elif polys.ndim == 3:
        polys = [i for i in polys]
    # ----
    empty_array = np.array([])
    output = []
    for ov in overlays:
        clip_extent = np.concatenate(
            (np.min(ov, axis=0), np.max(ov, axis=0))
        )
        for p in polys:
            if _in_LBRT_(p, clip_extent):
                result = p_ints_p(p, ov)
                if result is not None:
                    output.append(result)
    return output


# ---- clip is above
#
def _intersect_(p0, p1, p2, p3):
    """Return the intersection of two segments (p0-->p1) and (p0-->p2).

    All calculations are made whether the segments intersect or not.

    Notes
    -----
    Checks section.
    # d_gt0 = d_nom > 0
    # t1 = d_nom == 0.0
    # t2 = (b_num < 0) == d_gt0
    # t3 = (a_num < 0) == d_gt0
    # t4 = np.logical_or((b_num > d_nom) == d_gt0, (a_num > d_nom) == d_gt0)
    # good = ~(t1 + t2 + t3 + t4)

    >>> denom   = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    >>> s_numer = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
    >>> t_numer = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    >>> x = x0 + t * (x1 - x0)
    >>> y = y0 + t * (y1 - y0)

    `<http://paulbourke.net/geometry/pointlineplane/>`_.
    `<https://en.wikipedia.org/wiki/Intersection_(Euclidean_geometry)>`_.
    """
    null_pnt = np.array([np.nan, np.nan])
    x0, y0, x1, y1, x2, y2, x3, y3 = (*p0, *p1, *p2, *p3)
    p01_x, p01_y = p1 - p0
    p02_x, p02_y = p0 - p2
    p23_x, p23_y = p3 - p2
    # ---- denom = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    denom = (p23_y * p01_x - p23_x * p01_y)  # np.cross(p0-p1, p2-p3)
    if denom == 0.0:
        return null_pnt
    d_gt0 = denom > 0.
    s_numer = p01_x * p02_y - p01_y * p02_x
    if (s_numer < 0) == d_gt0:
        return null_pnt
    t_numer = p23_x * p02_y - p23_y * p02_x
    if (t_numer < 0) == d_gt0:
        return null_pnt
    # ---- are s_numer and t_numer between 0 and 1 test
#    if ((s_numer > denom) == d_gt0) and ((t_numer > denom) == d_gt0):
#        return null_pnt  # ---- change to and from or in line above
    t = t_numer / denom
    x = x0 + t * p01_x
    y = y0 + t * p01_y
#    t1 = s_numer /denom
#    x2 = x0 + t1 * p01_x
#    y2 = y0 + t1 * p01_y
#     print("inside _intersection_ {}, {}".format([x, y], [x1, y1]))
    return np.array([x, y])  # , np.array([x2, y2])


def intersects(*args):
    """Line segment intersection check. **Largely kept for documentation**.

    Two lines or 4 points that form the lines.  This does not extrapolate to
    find the intersection, they either intersect or they don't

    Parameters
    ----------
    args : array-like
        Two lines with two points each:  intersects(line0, line1).

        Four points, two points for each: intersects(p0, p1, p2, p3).

    Returns
    -------
    boolean, if the segments do intersect

    >>> a = np.array([[0, 0], [10, 10]])
    >>> b = np.array([[0, 10], [10, 0]])
    >>> intersects(*args)  # True

    Examples
    --------
    ::

        c = np.array([[0, 0], [0, 90], [90, 90], [60, 60], [20, 20], [0, 0]])
        segs = [np.array([c[i-1], c[i]]) for i in range(1, len(c))]
        ln = np.array([[50, -10], [50, 100]])
        print("line {}".format(ln.ravel()))
        for i, j in enumerate(segs):
            r = intersects(ln, j)
            print("{}..{}".format(j.ravel(), r))
        ...
        line [ 50 -10  50 100]
        [ 0  0  0 90]..(False, 'collinear/parallel')
        [ 0 90 90 90]..(True, (50.0, 90.0))
        [90 90 60 60]..(False, 'numerator(s) check')
        [60 60 20 20]..(True, (50.0, 49.99999999999999))
        [20 20  0  0]..(False, 's_num -3300 den 2200 cross(p1-p0, p0-p2) = 0')

    References
    ----------
    `<https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
    line-segments-intersect#565282>`_.

    """
    if len(args) == 2:    # two lines
        p0, p1, p2, p3 = *args[0], *args[1]
    elif len(args) == 4:  # four points
        p0, p1, p2, p3 = args
    else:
        raise AttributeError("Pass 2, 2-pnt lines or 4 points to the function")
    #
    x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3  # points to xs and ys
    #
    # ---- First check ----   np.cross(p1-p0, p3-p2)
    denom = (x1 - x0) * (y3 - y2) - (y1 - y0) * (x3 - x2)
    t1 = denom == 0.0
    if t1:  # collinear
        return (False, "collinear/parallel")
    #
    # ---- Second check ----  np.cross(p1-p0, p0-p2)
    denom_gt0 = denom > 0  # denominator greater than zero
    s_numer = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
    t2 = (s_numer < 0) == denom_gt0
    if t2:
        msg = "s_num {} den {} cross(p1-p0, p0-p2) = 0".format(s_numer, denom)
        return (False, msg)
    #
    # ---- Third check ----  np.cross(p3-p2, p0-p2)
    t_numer = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    t3 = (t_numer < 0) == denom_gt0
    if t3:
        msg = "t_num {} den {} cross(p3-p2, p0-p2) = 0".format(t_numer, denom)
        return (False, msg)
    #
    # ---- Fourth check ----
    t4 = np.logical_or((s_numer > denom) == denom_gt0,
                       (t_numer > denom) == denom_gt0
                       )
    if t4:
        return (False, "numerator(s) check")
    #
    # ---- check to see if the intersection point is one of the input points
    # substitute p0 in the equation  These are the intersection points
    t = t_numer / denom
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    # be careful that you are comparing tuples to tuples, lists to lists
    if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
        return (True, "intersects at an input point {}, {}".format(x, y))
    return (True, (x, y))


# ----------------------------------------------------------------------------
# ---- (3) dissolve shared boundaries
#
def dissolve(g, plot=False):
    """Dissolves shared boundaries in geometries within a set of data.

    Parameters
    ----------
    g : Geo array
        The array to dissolve.
    plot : boolean
        True, to produce a plot of the segments forming the new geometry.

    Returns
    -------
    A new Geo array.

    """
    if plot is True:
        from npgeom.npg_plots import plot_polygons, plot_2d
    if g.__class__.__name__ != "Geo":
        print("\nGeo array required for `to_this`\n")
        return
    bit_segs = g.od_pairs()
    fr_to = np.concatenate(bit_segs, axis=0)
    to_fr = np.hstack([fr_to[:, -2:], fr_to[:, :2]])
    out = compare_geom(fr_to, to_fr, unique=False, invert=True,
                       return_idx=False
                       )
    pairs = np.array(list(zip(out[:, :2], out[:, -2:])))
    seq = np.concatenate(pairs, axis=0)
    uni, idx, inv = np.unique(seq, True, True, axis=0)
    uni_ordered = uni[inv]
    srted = radial_sort(seq, close_poly=True, clockwise=True)
    if plot:
        plot_polygons([srted])
    return srted


# ----------------------------------------------------------------------------
# ---- (4) append geometry
#
def append_(this, to_this):
    """
    Append `this` geometry `to_this` geometry.

    Parameters
    ----------
    this : array(s) or a Geo array
        The geometry to append to the existing geometry (`to_this`).
        `this` can be a single array, a list of arrays or a Geo array.
        If you want to append object array(s) (dtype= 'O'), then convert to a
        list of arrays or a list of lists first.
    to_this : Geo array
        The Geo array to receive the new geometry

    Returns
    -------
    A new Geo array.

    a = np.array([[0, 10.],[5., 15.], [5., 0.], [0., 10]])
    b = a + [5, 0]
    this = [a, b]
    to_this = s0
    """
    # if not hasattr(to_this, 'IFT'):
    if to_this.__class__.__name__ != "Geo":
        print("\nGeo array required for `to_this`\n")
        return
    if this.__class__.__name__ == "Geo":
        a_stack = this.XY
        IFT = this.IFT
        if this.K != to_this.K:
            print("\nGeo array `kind` is not the same,\n")
            return
    else:
        a_stack, IFT, extent = npg.array_IFT(this)
    last = to_this.IFT[-1, :]
    add_ = []
    for i, row in enumerate(IFT, 1):
        add_.append([last[0] + i, last[2] + row[1],
                     last[2] + row[2]] + list(row[3:]))
    add_ = np.atleast_2d(add_)
    new_ift = np.vstack((to_this.IFT, add_))
    xys = np.vstack((to_this.XY, a_stack))
    kind = to_this.K
    sr = to_this.SR
    out = npg.Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
    return out


# ----------------------------------------------------------------------------
# ---- (5) merge geometry
#
def merge_(a, to_b):
    """
    Merge `this` geometry `to_this` geometry

    Parameters
    ----------
    this : array(s) or a Geo array
        The geometry to merge to the existing geometry (`to_this`).
    to_this : Geo array
        The Geo array to receive the new geometry

    Notes
    -----
    The `this` array can be a single array, a list of arrays or a Geo array.
    If you want to append object array(s) (dtype= 'O'), then convert to a
        list of arrays or a list of lists first.

    During the merged operation, overlapping geometries are not intersected.

    Returns
    -------
    A new Geo array.

    a = np.array([[0, 8.],[5., 13.], [5., 8.], [0., 8]])
    b = a + [5, 2]
    this = [a, b]
    to_this = s0
    """
    if ('Geo' in str(type(to_b))) & (issubclass(to_b.__class__, np.ndarray)):
        a_stack = to_b.XY
        IFT = to_b.IFT
    else:
        to_b = npg.arrays_to_Geo([to_b])
    if ('Geo' in str(type(a))) & (issubclass(a.__class__, np.ndarray)):
        if a.K != to_b.K:
            print("\nGeo array `kind` is not the same,\n")
            return
    else:
        a = np.asarray(a)
        if a.ndim == 2:
            a = [a]
        # mins = np.min(np.vstack(([np.min(i, axis=0) for i in this])), axis=0)
        a_stack, IFT, extent = npg.array_IFT(a)
        a_stack = a_stack + extent[0]

    last = to_b.IFT[-1, :]
    add_ = []
    for i, row in enumerate(IFT, 1):
        add_.append([last[0] + i, last[2] + row[1],
                     last[2] + row[2]] + list(row[3:]))
    add_ = np.atleast_2d(add_)
    new_ift = np.vstack((to_b.IFT, add_))
    xys = np.vstack((to_b.XY, a_stack))
    kind = to_b.K
    sr = to_b.SR
    out = npg.Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
    return out


# ----------------------------------------------------------------------------
# ---- (6) union geometry
#
def _union_(poly, clipper, is_polygon=True):
    """Union polygon features.

    Parameters
    ----------
    poly, clipper : ndarray
        The two polygon arrays to union

    Returns
    -------
    Unioned polygon
    """
    from npgeom.npg_plots import plot_2d, plot_polygons
    # plot_2d([b0, clipper], True, True)

    def _radsrt_(a, dup_first=True):
        """Worker for radial sort."""
        uniq = np.unique(a, axis=0)
        cent = np.mean(uniq, axis=0)
        dxdy = uniq - cent
        angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])
        idx = angles.argsort()
        srted = uniq[idx]
        if dup_first:
            srted =np.concatenate((srted, [srted[0]]), axis=0)[::-1]
        return srted

    # ----
    def _clockwise_check_(a):
        """Mini e_area, used by areas and centroids."""
        x0, y1 = (a.T)[:, 1:]
        x1, y0 = (a.T)[:, :-1]
        e0 = np.einsum('...i,...i->...i', x0, y0)
        e1 = np.einsum('...i,...i->...i', x1, y1)
        return np.sum((e0 - e1)*0.5) > 0
    # ----
    empty_array = np.array([])
    clone = np.copy(poly)  # Don't modify original list.
    poly = np.copy(poly[1:])        # normally poly[:-1])
    clipper = np.asarray(clipper[1:])  # normally clipper[:-1]
    result = []
    strt = clipper[-1]         # Start with first vertex in clip polygon.
    msg = "p0, p1, strt, end, c\n  {} {} {} {}\n - {} ...{}"
    for end in clipper:
        inputList = poly
        output = []
        if len(inputList) == 0:
            break
        p0 = inputList[-1]  # Previous vertex.
        is_first = True
        for p1 in inputList:
            a, b = intersects(p0, p1, strt, end)
            if a is True:
                output.append(b)
            p0 = np.copy(p1)  # remember vertex for next edge.
        strt = np.copy(end)  # remember clip vertex for next edge.
        if len(output) > 0:  # form a closed-loop if poly is a polygon
            result.append(np.array(output))
    inside = npg.pnts_in_poly(clone, clipper)
    outside = npg.remove_geom(clone, inside)
    if outside is not None:
        print("outside {}".format(outside))
        outside = [np.atleast_2d(i) for i in outside if np.array(i).size > 0]
        result.extend(outside)
    # clipper = np.vstack(clipper)
    result.extend(clipper)
    result.extend(clone)
    # print(result)
    result = np.vstack(result)
    result = _radsrt_(result, dup_first=True)  # return CW
    if not _clockwise_check_(result):
        result = result[::-1]
    return result


def union_(poly, clipper, is_polygon=True):
    """Return. Test main program"""
    if poly.__class__.__name__ != "Geo":
        print("\nGeo array required for `to_this`\n")
        return
    cw = poly.CW
    ccw = np.zeros(cw.shape, np.bool_)
    ccw[:] = cw
    ccw = ~ccw
    cw_idx = np.where(poly.CW == 1)[0]
    cw_bits = [poly.bits[i] for i in cw_idx]
    ccw_idx = np.where(poly.CW == 0)[0]
    ccw_bits = [poly.bits[i] for i in ccw_idx]
    final = [_union_(p, clipper, is_polygon=True) for p in cw_bits]
    # final = np.vstack((final))
    # ccw_bits = np.vstack((ccw_bits))
    # final = np.concatenate((final, ccw_bits), axis=0)
    return final
# ---- keep for now ----
# ---- _poly_intersect_poly
# def _poly_intersect_poly(p, c):
#     """Intersect the points of intersection between two polygons.

#     Parameters
#     ----------
#     p, c : ndarrays
#         Polygons represented by (n, 2) arrays. `p` is the subject polygon and
#         `c` is the clipping polygon.

#     Returns
#     -------
#     Intersection points or None.
#     """
#     p0s = p[:-1]
#     p1s = p[1:]
#     p2s = c[:-1]
#     p3s = c[1:]
#     n = np.arange(len(p0s))
#     m = np.arange(len(p2s))
#     z = [[_intersect_(p0s[i], p1s[i], p2s[j], p3s[j])
#           for i in n] for j in m]
#     # inside = [crossing_num(c, i) for i in p.bits]
#     z0 = np.vstack(z)
#     good = ~np.isnan(z0)[:, 0]
#     z1 = radial_sort(z0[good])
#     return z1


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polylines2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygon2pnts"
