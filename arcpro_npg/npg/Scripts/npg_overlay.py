# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
-----------
npg_overlay
-----------

Overlay tools. Working with two sets of geometries.  The functions are largely
confined to polygon and polyline objects.

----

Script :
    npg_overlay.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2020-10-23

Purpose
-------
Functions for overlay analysis.

See Also
--------
None


References
----------
`Paul Bourke geometry
<http://paulbourke.net/geometry/pointlineplane/>`_.

**Dissolve**

Folder path::

    C:\arc_pro\Resources\ArcToolBox\Scripts\Weights.py
    C:\arc_pro\Resources\ArcToolBox\Toolboxes\
    GeoAnalytics Desktop Tools.tbx\DissolveBoundaries.tool

Web link

`<https://pro.arcgis.com/en/pro-app/tool-reference/geoanalytics-desktop
/dissolve-boundaries.htm>`_.

"""

# pycodestyle D205 gets rid of that one blank line thing
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0402,E0611,E1136,E1121,R0904,R0914,
# pylint: disable=W0201,W0212,W0221,W0612,W0621,W0105
# pylint: disable=R0902


import sys
# from textwrap import dedent
import numpy as np

from npGeo import Geo, array_IFT, arrays_to_Geo, roll_coords
from npg_pip import np_wn
from npg_helpers import (_to_lists_, _bit_check_, _in_LBRT_,
                         remove_geom, radial_sort)

# ---- optional imports
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

# noqa: E501
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 7.3f}'.format})

script = sys.argv[0]  # print this should you need to locate the script

# FLOATS = np.typecodes['AllFloat']
# INTS = np.typecodes['AllInteger']
# NUMS = FLOATS + INTS
# TwoPI = np.pi * 2.0

__all__ = [
    'pnt_segments',
    'p_ints_p', 'intersections', 'clip_',
    'intersects',
    'adjacent', 'adjacency_matrix',
    'append_',
    'merge_',
    'union_'
    ]  # 'dissolve',
__helpers__ = ['_intersect_', '_adj_']

# ----------------------------------------------------------------------------
# ---- (1) helpers/mini functions
#

# c0 = np.logical_and(yp[i] <= y, y < yp[j])
# c1 = np.logical_and(yp[j] <= y, y < yp[i])
# np.logical_or(c0, c1)
# ((yp[j] <= y) && (y < yp[i]))) &&
#             (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
#           c = !c;


def pnt_segments(g):
    """Return point segmentation information.

    Notes
    -----
    p_ids : Nx2 array
        The first column is the point number and the second column is the
        feature they belong to.  The point IDs are in the sequence order of
        the poly feature construction.  This information is used to determine
        where duplicates occur and to what features they belong.
    """
    uni, idx, inv, cnts = np.unique(g, True, True, True, axis=0)
    # index = sorted(idx)
    p_ids = g.pnt_indices()  # get the point IDs, they will not be sorted
    out = []
    uni_cnts = np.unique(cnts).tolist()
    for i in uni_cnts:
        sub0 = uni[cnts == i]
        sub1 = idx[cnts == i]
        sub2 = p_ids[sub1]
        out.append([i, sub0, sub1, sub2])
    return out


# ----------------------------------------------------------------------------
# ---- (3) clip, intersect geometry
#  `p_ints_p` is the main function
#  `intersections` uses this to batch intersect multiple polygons as input
#  and intersectors.

def p_ints_p(poly0, poly1):
    """Intersect two polygons.  Used in clipping.

    Parameters
    ----------
    poly0, poly1 : ndarrays
        Two polygons/polylines. poly0, feature to intersect using poly1 as the
        clipping feature.

    Returns
    -------
    Points of intersection or None.

    Notes
    -----
    Using Paul Bourke`s notation.

    Intersection point of two line segments in 2 dimensions, 1989.

    `<http://paulbourke.net/geometry/pointlineplane/>`_.

    `<http://paulbourke.net/geometry/polygonmesh/>`_.

    | line a : p0-->p1
    | line b : p2-->p3

    >>> d_nom = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    >>> a_num = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)  # ==> u_a
    >>> b_num = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)  # ==> u_b
    >>> u_a = a_num/d_nom  # if d_nom != 0
    >>> u_b = b_num/d_nom

    if 0 <= u_a, u_b <=1 then the intersection is on both segments
    """
    poly0, poly1 = [i.XY if hasattr(i, "IFT") else i for i in [poly0, poly1]]
    p01 = poly0[1:] - poly0[:-1]
    p01_x, p01_y = p01.T
    p23 = poly1[1:] - poly1[:-1]
    p23_x, p23_y = p23.T
    p02 = poly0[:-1] - poly1[:-1][:, None]
    d_nom = (p23_y[:, None] * p01_x) - (p23_x[:, None] * p01_y)
    b_num = p01_x * p02[..., 1] - p01_y * p02[..., 0]
    a_num = p23_x[:, None] * p02[..., 1] - p23_y[:, None] * p02[..., 0]
    with np.errstate(all='ignore'):  # divide='ignore', invalid='ignore'):
        u_a = a_num/d_nom
        u_b = b_num/d_nom
        z0 = np.logical_and(0. <= u_a, u_a <= 1.)
        z1 = np.logical_and(0. <= u_b, u_b <= 1.)
        bth = z0 & z1
        xs = u_a * p01_x + poly0[:-1][:, 0]
        ys = u_a * p01_y + poly0[:-1][:, 1]
        # *** np.any(bth, axis=1)
        # yields the segment on the clipper that the points are on
    xs = xs[bth]
    ys = ys[bth]
    if xs.size > 0:
        final = np.zeros((len(xs), 2))
        final[:, 0] = xs
        final[:, 1] = ys
        return final  # np.unique(final, axis=0)
    return None


def intersections(polys, overlays, outer_only=True, stacked=False):
    """Batch `p_ints_p`.

    Parameters
    ----------
    polys : list of ndarrays or a Geo Array
        The geometry to intersect.
    overlay : polygon/polyline
        The intersecting geometry which is used to examine all the input polys.
    outer_only : boolean
        True, uses Geo array inner and outer rings.  False, for outer rings.
        This only applies to Geo arrays.
    stacked : boolean
        True, returns an Nx2 array of the intersection points.  False, returns
        the intersections on a per-shape basis.

    Requires
    --------
    ``_to_lists_`` from npg_helpers

    Notes
    -----
    Make sure you process holes separately otherwise you will get phantom
    intersection points where the inner and outer rings of a polygon connect.

    Returns
    -------
    - The points of intersection between the input poly features and the
      overlay feature(s).
    - The id values for the clip feature and polygon for each intersection.
    """
    #
    polys = _to_lists_(polys, outer_only)
    overlays = _to_lists_(overlays, outer_only)
    # ----
    output = []
    cl_info = []
    for i, ov in enumerate(overlays):
        clip_extent = np.concatenate((np.min(ov, axis=0), np.max(ov, axis=0)))
        for j, p in enumerate(polys):
            if _in_LBRT_(p, clip_extent):
                result = p_ints_p(p, ov)
                if result is not None:
                    output.append(result)
                    cl_info.append([i, j])
    if stacked:
        output = np.vstack(output)
    return output, cl_info


def clip_(geo, c):
    """Do the work.

    https://github.com/fonttools/pyclipper/tree/master/pyclipper

    For testing use `sq` and
    c = np.array([[0, 10.],[11., 13.], [12., 7.], [8., 2], [0., 10]])
    """
    def _e_2d_(a, p):
        """Array points to point distance."""
        diff = a - p[None, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    if not hasattr(geo, 'IFT'):
        geo = arrays_to_Geo(geo)
    bounds = dissolve(geo).XY
    in_0 = np_wn(bounds, c)  # points_in_polygon(bounds, c)  # 30 micro
    in_1 = np_wn(c, bounds)  # points_in_polygon(c, bounds)
    x_sect0 = p_ints_p(bounds[::-1], c)  # clips points in clockwise order
    # x_sect1 = p_ints_p(bounds, c[::-1])
    test = [in_0, in_1, x_sect0]
    final = []
    for i in test:
        if hasattr(i, 'size'):
            if i.size > 0:
                final.append(np.atleast_2d(i))
    if final:
        # return radial_sort(np.vstack(final))
        final = np.vstack(final)
        min_f = np.array([np.min(final[:, 0]), np.mean(final[:, 1])])
        dxdy = np.subtract(final, np.atleast_2d(min_f))
        ang = np.degrees(np.arctan2(dxdy[:, 1], dxdy[:, 0]))  # 2.0
        dist = _e_2d_(final, min_f)
        ang_dist = np.vstack((ang, dist)).T  # 4.7
        keys = np.argsort(uts(ang_dist))    # 17.7
        rev = keys[::-1]   # works
        return final[rev]
    return None


def _intersect_(p0, p1, p2, p3):
    """Return the intersection of two segments.

    The intersection of the segments, (p0-->p1) and (p0-->p2) or
    the extrapolation point if they don't cross.

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
    >>> t = t_numer / denom
    >>> x = x0 + t * (x1 - x0)
    >>> y = y0 + t * (y1 - y0)

    `<http://paulbourke.net/geometry/pointlineplane/>`_.

    `<https://en.wikipedia.org/wiki/Intersection_(Euclidean_geometry)>`_.
    """
    null_pnt = np.array([np.nan, np.nan])
    x0, y0 = p0
    p01_x, p01_y = np.subtract(p1, p0)
    p02_x, p02_y = np.subtract(p0, p2)
    p23_x, p23_y = np.subtract(p3, p2)
    # ---- denom = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    denom = (p23_y * p01_x - p23_x * p01_y)  # np.cross(p0-p1, p2-p3)
    if denom == 0.0:
        return null_pnt
    d_gt0 = denom > 0.
    s_numer = p01_x * p02_y - p01_y * p02_x
    if (s_numer < 0.0) == d_gt0:
        return null_pnt
    t_numer = p23_x * p02_y - p23_y * p02_x
    if (t_numer < 0.0) == d_gt0:
        return null_pnt
    # ---- are s_numer and t_numer between 0 and 1 test
    if ((s_numer > denom) == d_gt0) and ((t_numer > denom) == d_gt0):
        return null_pnt  # ---- change to and from or in line above
    t = t_numer / denom
    x = x0 + t * p01_x
    y = y0 + t * p01_y
    return np.array([x, y])


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
    if isinstance(args, np.ndarray):
        args = args.tolist()
    if len(args) == 2:
        p0, p1, p2, p3 = *args[0], *args[1]
    elif len(args) == 4:
        p0, p1, p2, p3 = args
    else:
        raise AttributeError("Pass 2, 2-pnt lines or 4 points to the function")
    #
    # ---- First check
    # Given 4 points, if there are < 4 unique, then the segments intersect
    u, cnts = np.unique((p0, p1, p2, p3), return_counts=True, axis=0)
    if len(u) < 4:
        intersection_pnt = u[cnts > 1]
        return True, intersection_pnt
    #
    x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3  # points to xs and ys
    #
    # ---- Second check ----   np.cross(p1-p0, p3-p2)
    denom = (x1 - x0) * (y3 - y2) - (y1 - y0) * (x3 - x2)
    t1 = denom == 0.0
    if t1:  # collinear
        return (False, "collinear/parallel")
    #
    # ---- Third check ----  np.cross(p1-p0, p0-p2)
    denom_gt0 = denom > 0  # denominator greater than zero
    s_numer = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
    t2 = (s_numer < 0) == denom_gt0
    if t2:
        msg = "s_num {} den {} cross(p1-p0, p0-p2) = 0".format(s_numer, denom)
        return (False, msg)
    #
    # ---- Fourth check ----  np.cross(p3-p2, p0-p2)
    t_numer = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    t3 = (t_numer < 0) == denom_gt0
    if t3:
        msg = "t_num {} den {} cross(p3-p2, p0-p2) = 0".format(t_numer, denom)
        return (False, msg)
    #
    # ---- Fifth check ----
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
def _adj_(a, full=False):
    """Determine adjacency for a polygon Geo array's outer rings.

    # Parameters
    ----------
    a : Geo array
    full : boolean
        True, returns a full report of the shapes being compared (f_shp, t_shp)
        and the input point ids.  The common points are returned from t_shp and
        their point and polygon id values.
        False, simply returns the adjacent polygon ids (f_shp, t_shp).

    Returns
    -------
    Two arrays, the adjacency information and the outer_rings.
    """
    out = []
    ids = a.shp_pnt_ids
    fr_to = a.FT
    f_shps = a.IDs
    arr = a.XY
    for i, f_t in enumerate(fr_to[:-1]):
        f_shp = f_shps[i]
        f, t = f_t
        w = (arr[f:t][:, None] == arr[t:]).any(0).all(-1)  # .tolist()
        if np.sum(w) > 0:
            pnt_shp = ids[t:][w]
            t_shp = np.unique(pnt_shp[:, 1])  # t_shp = pnt_shp[0][1]
            if full:
                common = arr[t:][w]
                out.append([[f_shp] + t_shp.tolist(), f_t, common, pnt_shp])
            else:
                out.append(np.asarray([f_shp] + t_shp.tolist()))
    return out


def dissolve(a, asGeo=True):
    r"""Dissolve polygons sharing edges.

    Parameters
    ----------
    a : Geo array
        A Geo array is required. Use ``arrays_to_Geo`` to convert a list of
        lists/arrays or an object array representing geometry.
    asGeo : boolean
        True, returns a Geo array. False returns a list of arrays.

    Requires
    --------
    `_adj_` to determine shared boundaries.

    Notes
    -----
    >>> from npgeom.npg_plots import plot_polygons  # to plot the geometry

    `_isin_2d_`, `find`, `adjacent` equivalent::

        (b0[:, None] == b1).all(-1).any(-1)
    """

    def _adjacent_(a, b):
        """Check adjacency between 2 polygon shapes."""
        s = np.sum((a[:, None] == b).all(-1).any(-1))
        return True if s > 0 else False

    def _cycle_(b0, b1):
        """Cycle through the bits."""
        def _find_(a, b):
            """Find.  Abbreviated form of ``adjacent``, to use for slicing."""
            return (a[:, None] == b).all(-1).any(-1)

        idx01 = _find_(b0, b1)
        if idx01.sum() == 0:
            return None
        elif idx01[0] == 1:  # you can't split between the first and last pnt.
            b0, b1 = b1, b0
            idx01 = _find_(b0, b1)
        dump = b0[idx01]
        sp0 = np.nonzero(idx01)[0]
        sp1 = np.any(np.isin(b1, dump, invert=True), axis=1)
        z0 = np.array_split(b0, sp0[1:])
        z1 = b1[sp1]
        return np.concatenate((z0[0], z1, z0[-1]), axis=0)

    def _combine_(r, shps):
        """Combine the shapes."""
        missed = []
        processed = False
        for i, shp in enumerate(shps):
            adj = _adjacent_(r, shp[1:-1])  # shp[1:-1])
            if adj:
                new = _cycle_(r, shp[:-1])  # shp[:-1])  ** today
                r = new
                processed = True
            else:
                missed.append(shp)
        if len(shps) == 2 and not processed:
            missed.append(r)
        return r, missed  # done

    # --- check for appropriate Geo array.
    if not hasattr(a, "IFT") or a.is_multipart():
        msg = """function : dissolve
        A `Singlepart` Geo array is required. Use ``arrays_to_Geo`` to convert
        arrays to a Geo array and use ``multipart_to_singlepart`` if needed.
        """
        print(msg)
        return None
    # --- get the outer rings, roll the coordinates and run ``_combine_``.
    a = a.outer_rings(True)
    a = a.roll_shapes()
    a.IFT[:, 0] = np.arange(len(a.IFT))
    out = []
    ids = a.IDs
    shps = a.get_shapes(ids, False)
    r = shps[0]
    missed = shps
    N = len(shps)
    cnt = 0
    while cnt <= N:
        r1, missed1 = _combine_(r, missed[1:])
        if r1 is not None and N >= 0:
            out.append(r1)
        if len(missed1) == 0:
            N = 0
        else:
            N = len(missed1)
            r = missed1[0]
            missed = missed1
        cnt += 1
    # final kick at the can
    if len(out) > 1:
        r, missed = _combine_(out[0], out[1:])
        if missed is not None:
            out = [r] + missed
        else:
            out = r
    if asGeo:
        out = arrays_to_Geo(out, 2, "dissolved", False)
        out = roll_coords(out)
    return out  # , missed


# ---- (4) adjacency
#
def adjacent(a, b):
    """Check adjacency between 2 polygon shapes.

    Parameters
    ----------
    a, b : ndarrays
        The arrays of coordinates to check for polygon adjacency.
        The duplicate first/last point is removed so that a count will not
        flag a point as being a meeting point between polygons.

    Note
    ----
    Adjacency is defined as meeting at least 1 point.  Further checks will be
    needed to assess whether two shapes meet at 1 point or 2 non-consequtive
    points.
    """
    s = np.sum((a[:, None] == b).all(-1).any(-1))
    return True if s > 0 else False


def adjacency_matrix(a, prn=False):
    """Construct an adjacency matrix from an input polygon geometry.

    Parameters
    ----------
    a : array_like
        A Geo array, list of lists/arrays or an object array representing
        polygon geometry.

    Returns
    -------
    An nxn array adjacency for polygons and a id-keys to convert row-column
    indices to their original ID values.

    The diagonal of the output is assigned the shape ID. Adjacent to a cell
    is denoted by assigning the shape ID.  Non-adjacent shapes are assigned -1

    Example::

        ad =adjacency_matrix(a)             # ---- 5 polygons shapes
        array([[ 0, -1, -1,  3,  4],
               [-1,  1, -1, -1, -1],
               [-1, -1,  2, -1, -1],
               [ 0, -1, -1,  3,  4],
               [ 0, -1, -1,  3,  4]])
        ad >= 0                             # ---- where there are links
        array([[1, 0, 0, 1, 1],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [1, 0, 0, 1, 1],
               [1, 0, 0, 1, 1]])
        row_sums = np.sum(ad >= 0, axis=1)  # sum the rows
        w = np.where(row_sums > 1)[0]       # find out where sum is > 1
        np.unique(ad[w], axis=0)            # get the unique combinations
        array([[ 0, -1, -1,  3,  4]])       # of the adjacency test if needed

    Polygons 0, 3 and 4 are connected. Polygons 1 and 2 are not connected to
    other geometry objects.
    """

    def recl(arr, ids):
        """Reclass non-sequential id values in an array."""
        u = np.unique(ids)
        d = np.arange(len(u))
        ud = np.vstack((u, d)).T
        du = ud[::-1]
        for i in du:
            if i[0] - i[1] != 0:
                arr[arr == i[1]] = i[0]
            # else:
            #     arr[arr == i[0]] = i[0]
        return arr
    #
    rings = _bit_check_(a, just_outer=True)  # get the outer rings
    ids = a.IDs[a.CW == 1]                   # and their associated ID values
    n = len(rings)
    N = np.arange(n)
    z = np.full((n, n), -1)
    np.fill_diagonal(z, N)  # better with ids but will reclass later
    for i in N:
        for j in N:
            if j > i:  # changed from j != i
                ij = adjacent(rings[i], rings[j])
                if ij:
                    z[i, j] = j
                    z[j, i] = i  # added the flop
    if np.sum(ids - N) > 0:
        z = recl(z, ids)  # reclass the initial values using the actual ids
    if prn:
        out = "\n".join(["{} : {}".format(i, row[row != -1])
                         for i, row in enumerate(z)])
        print(out)
        return None
    return z

    """
    https://stackoverflow.com/questions/14263284/create-non-intersecting
    -polygon-passing-through-all-given-points/20623817#20623817

    see the code there

    find the left most and right-most points, sort those above the line into
    A and B
    sort the points above by ascending X, those below by decending X
    use left=hand/right hand rule to order the points
    """


# ----------------------------------------------------------------------------
# ---- (5) append geometry
#
def append_(this, to_this):
    """Append `this` geometry `to_this` geometry.

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
    if not hasattr(to_this, "IFT"):
        print("\nGeo array required for `to_this`\n")
        return
    if hasattr(this, "IFT"):
        a_stack = this.XY
        IFT = this.IFT
        if this.K != to_this.K:
            print("\nGeo array `kind` is not the same,\n")
            return
    else:
        a_stack, IFT, extent = array_IFT(this)
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
    out = Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
    return out


# ----------------------------------------------------------------------------
# ---- (6) merge geometry
#
def merge_(this, to_this):
    """
    Merge `this` geometry and `to_this` geometry.  The direction is important.

    Parameters
    ----------
    this : array(s) or a Geo array
        The geometry to merge to the existing geometry (`to_this`).
    to_this : Geo array
        The Geo array to receive the new geometry.

    Notes
    -----
    The `this` array can be a single array, a list of arrays or a Geo array.
    If you want to append object array(s) (dtype= 'O'), then convert to a
    list of arrays or a list of lists first.

    During the merge operation, overlapping geometries are not intersected.

    Returns
    -------
    A new Geo array.

    this = np.array([[0, 8.], [5., 13.], [5., 8.], [0., 8]])
    b = this + [5, 2]
    this = [a, b]
    to_this = s0
    """
    a = this      # --- rename to simplify the input names
    b = to_this   # merge a to b, or this to_this
    if not hasattr(b, 'IFT'):
        b = arrays_to_Geo(b)
    b_XY = b.XY
    b_IFT = b.IFT
    if hasattr(this, 'IFT'):
        if a.K != b.K:
            print("\nGeo array `kind` is not the same.\n")
            return None
        a_XY = a.XY
        a_IFT = a.IFT
    else:
        a = np.asarray(a)
        if a.ndim == 2:
            a = [a]
        a_XY, a_IFT, extent = array_IFT(a)
        a_XY = a_XY + extent[0]
    last = b.IFT[-1, :]
    add_ = []
    for i, row in enumerate(a_IFT, 1):
        add_.append([last[0] + i, last[2] + row[1],
                     last[2] + row[2]] + list(row[3:]))
    add_ = np.atleast_2d(add_)
    new_ift = np.vstack((b_IFT, add_))
    xys = np.vstack((b_XY, a_XY))
    kind = b.K
    sr = b.SR
    out = Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
    return out


# ----------------------------------------------------------------------------
# ---- (7) union geometry
#
def union_(a, b, is_polygon=True):
    """Union polyline/polygon features.

    Parameters
    ----------
    a, b : ndarray
        The two polygon arrays to union.  Holes not supported as yet.

    Requires
    --------
    `npg_helpers.radial_sort`
        to close polygons and/or sort coordinates in cw or ccw order.
    `npg_pip.np_wn`
        point in polygon using winding number.

    Returns
    -------
    Unioned polygon.
    """
    if hasattr(a, "IFT"):
        a = a.outer_rings()
    if hasattr(b, "IFT"):
        b = b.outer_rings()
    # ---- get the intersection points
    x_sect = p_ints_p(a, b)
    if not x_sect:
        return np.asarray([a, b], dtype="O")
    a_in_b = np_wn(a, b)
    b_in_a = np_wn(b, a)
    out_ab = remove_geom(a, a_in_b)
    out_ba = remove_geom(b, b_in_a)
    if out_ab is None:
        out_ab = a
    if out_ba is None:
        out_ba = b
    stack = np.vstack([np.atleast_2d(i)
                       for i in (x_sect, out_ab, out_ba)
                       if i is not None]
                      )
    srt_ = radial_sort(stack, close_poly=True, clockwise=True)
    return srt_


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
#     in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polylines2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygon2pnts"
# python
