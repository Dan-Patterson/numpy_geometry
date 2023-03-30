# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
-----------
npg_overlay
-----------

Overlay tools. Working with two sets of geometries.  The functions are largely
confined to polygon and polyline objects.

**Main functions**::

- intersect, intersections
- dissolve
- adjacent, adjacency_matrix
- append
- merge
- union
- crossing, crossings

----

Script :
    npg_overlay.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2022-09-21

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

`<https://stackoverflow.com/questions/26782038/how-to-eliminate-the-extra
-minus-sign-when-rounding-negative-numbers-towards-zer>`_.
"""

# pycodestyle D205 gets rid of that one blank line thing
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0402,E0611,E1136,E1121,R0904,R0914,
# pylint: disable=W0201,W0212,W0221,W0612,W0621,W0105
# pylint: disable=R0902


import sys
# from textwrap import dedent
import numpy as np

# if 'npg' not in list(locals().keys()):
#     import npg
# import npGeo
# from npGeo import array_IFT, arrays_to_Geo, roll_coords
from npg import npGeo
from npg.npg_pip import np_wn
# import npg_helpers
from npg.npg_helpers import (_to_lists_, _bit_check_, _in_LBRT_,
                             remove_geom, radial_sort)

# -- optional imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
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
    'pnt_segment_info',
    'p_ints_p', 'intersections',
    'intersects',
    'dissolve',
    'adjacent', 'adjacency_matrix',
    'append_',
    'merge_',
    'union_as_one',
    'line_crosses', 'in_out_crosses', 'crossings'
]
__helpers__ = ['_intersect_', '_adj_']

__all__ = __helpers__ + __all__


# ----------------------------------------------------------------------------
# ---- (1) helpers/mini functions
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
    >>> s_num = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
    >>> t_num = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    >>> t = t_num / denom
    >>> x = x0 + t * (x1 - x0)
    >>> y = y0 + t * (y1 - y0)

    `<http://paulbourke.net/geometry/pointlineplane/>`_.

    `<https://en.wikipedia.org/wiki/Intersection_(Euclidean_geometry)>`_.
    """
    null_pnt = np.array([np.nan, np.nan])
    x0, y0 = p0
    p10_x, p10_y = np.subtract(p1, p0)
    p02_x, p02_y = np.subtract(p0, p2)
    p32_x, p32_y = np.subtract(p3, p2)
    # -- denom = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    denom = (p32_y * p10_x - p32_x * p10_y)  # np.cross(p0-p1, p2-p3)
    if denom == 0.0:
        return (False, null_pnt)
    d_gt0 = denom > 0.
    s_num = p10_x * p02_y - p10_y * p02_x
    if (s_num < 0.0) == d_gt0:
        return (False, null_pnt)
    t_num = p32_x * p02_y - p32_y * p02_x
    if (t_num < 0.0) == d_gt0:
        return (False, null_pnt)
    # -- are s_num and t_num between 0 and 1 test
    if ((s_num > denom) == d_gt0) and ((t_num > denom) == d_gt0):
        return null_pnt  # -- change to and from or in line above
    t = t_num / denom
    x = x0 + t * p10_x
    y = y0 + t * p10_y
    return (True, np.array([x, y]))


def pnt_segment_info(arr):
    """Return point segmentation information.

    Notes
    -----
    p_ids : Nx2 array
        The first column is the point number and the second column is the
        feature they belong to.  The point IDs are in the sequence order of
        the poly feature construction.  This information is used to determine
        where duplicates occur and to what features they belong.
    """
    uni, idx, inv, cnts = np.unique(arr, True, True, True, axis=0)
    # index = sorted(idx)
    p_ids = arr.pnt_indices()  # get the point IDs, they will not be sorted
    out = []
    uni_cnts = np.unique(cnts).tolist()
    for i in uni_cnts:
        sub0 = uni[cnts == i]
        sub1 = idx[cnts == i]
        sub2 = p_ids[sub1]
        out.append([i, sub0, sub1, sub2])
    return out


# ----------------------------------------------------------------------------
# ---- (2) intersect geometry
#  `p_ints_p` is the main function
#  `intersections` uses this to batch intersect multiple polygons as input
#  and intersectors.
#
def p_ints_p(poly0, poly1):
    r"""Intersect two polygons.  Used in clipping.

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
    >>>       = (y3[:, None] - y2) * (x1[:, None] - x0) -
    ...         (x3[:, None] - x2) * (y1[:, None] - x0)
    >>> a_num = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)  # ==> u_a
    >>> b_num = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)  # ==> u_b
    >>> u_a = a_num/d_nom  # if d_nom != 0
    >>> u_b = b_num/d_nom

    if 0 <= u_a, u_b <=1 then the intersection is on both segments
    """
    poly0, poly1 = [i.XY if hasattr(i, "IFT") else i for i in [poly0, poly1]]
    p10 = poly0[1:] - poly0[:-1]
    p32 = poly1[1:] - poly1[:-1]
    p10_x, p10_y = p10.T
    p32_x, p32_y = p32.T
    p02 = poly0[:-1] - poly1[:-1][:, None]
    d_nom = (p32_y[:, None] * p10_x) - (p32_x[:, None] * p10_y)
    a_num = p32_x[:, None] * p02[..., 1] - p32_y[:, None] * p02[..., 0]
    b_num = p10_x * p02[..., 1] - p10_y * p02[..., 0]
    #
    with np.errstate(all='ignore'):  # divide='ignore', invalid='ignore'):
        u_a = a_num / d_nom
        u_b = b_num / d_nom
        z0 = np.logical_and(u_a >= 0., u_a <= 1.)
        z1 = np.logical_and(u_b >= 0., u_b <= 1.)
        both = z0 & z1
        xs = u_a * p10_x + poly0[:-1][:, 0]
        ys = u_a * p10_y + poly0[:-1][:, 1]
        # *** np.any(both, axis=1)
        # yields the segment on the clipper that the points are on
        # *** np.sum(bth, axis=1)  how many intersections on clipper
        #     np.sum(both, axis=0)  intersections on the polygon
    xs = xs[both]
    ys = ys[both]
    if xs.size > 0:
        final = np.zeros((len(xs), 2))
        final[:, 0] = xs
        final[:, 1] = ys
        return final  # z0, z1, both  # np.unique(final, axis=0)
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
    # --
    output = []
    cl_info = []
    for i, ov in enumerate(overlays):
        clip_extent = np.concatenate((np.min(ov, axis=0), np.max(ov, axis=0)))
        for j, p in enumerate(polys):
            if _in_LBRT_(p, clip_extent):
                result = p_ints_p(p, ov)  # call to p_ints_p
                if result is not None:
                    output.append(result)
                    cl_info.append([i, j])
    if stacked:
        output = np.vstack(output)
    return output, cl_info


def intersects(*args):
    r"""Line segment intersection check. **Largely kept for documentation**.

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

    `<https://scicomp.stackexchange.com/questions/8895/vertical-and-horizontal
    -segments-intersection-line-sweep>`_.
    """
    msg = "A list/array representing 4 points required"
    if len(args) == 2:
        a, b = args
        if len(a) == 2 and len(b) == 2:
            p0, p1, p2, p3 = *a, *b
        else:
            raise AttributeError(msg)
    elif len(args) == 4:
        p0, p1, p2, p3 = args
    else:
        raise AttributeError(msg)
    #
    # -- First check, but it is expensive, so omit
    # Given 4 points, if there are < 4 unique, then the segments intersect
    # u, cnts = np.unique((p0, p1, p2, p3), return_counts=True, axis=0)
    # if len(u) < 4:
    #     intersection_pnt = u[cnts > 1]
    #     return True, intersection_pnt
    #
    # x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3  # pnts to xs and ys
    #
    x0, y0 = p0
    x10, y10 = p1 - p0
    x32, y32 = p3 - p2
    x02, y02 = p0 - p2
    # -- Second check ----   denom = np.cross(p1-p0, p3-p2)
    # denom = (x1 - x0) * (y3 - y2) - (y1 - y0) * (x3 - x2)
    denom = x10 * y32 - y10 * x32
    if denom == 0.0:  # "(1) denom = 0 ... collinear/parallel"
        return (False, None)
    #
    # -- Third check ----  s_num = np.cross(p1-p0, p0-p2)
    denom_gt0 = denom > 0  # denominator greater than zero
    # s_num = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
    s_num = x10 * y02 - y10 * x02
    if (s_num < 0) == denom_gt0:  # "(2) (s_n < 0) == (denom > 0) : False"
        return (False, None)
    #
    # -- Fourth check ----  np.cross(p3-p2, p0-p2)
    # t_num = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    t_num = x32 * y02 - y32 * x02
    if (t_num < 0) == denom_gt0:  # "(3) (t_n < 0) == (denom > 0) : False"
        return (False, None)
    #
    # -- Fifth check ----
    t4 = np.logical_or(
        (s_num > denom) == denom_gt0, (t_num > denom) == denom_gt0)
    if t4:  # "(4) numerator checks fail"
        return (False, None)
    #
    # -- check to see if the intersection point is one of the input points
    # substitute p0 in the equation  These are the intersection points
    t = t_num / denom
    x = x0 + t * x10  # (x1 - x0)
    y = y0 + t * y10  # (y1 - y0)
    #
    # be careful that you are comparing tuples to tuples, lists to lists
    if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
        # return (True, "(5) intersects at an input point {}, {}".format(x, y))
        return (True, (x, y))
    return (True, (x, y))


# ----------------------------------------------------------------------------
# ---- (3) dissolve shared boundaries
#
def dissolve(a, asGeo=True):
    """Dissolve polygons sharing edges.

    Parameters
    ----------
    a : Geo array
        A Geo array is required. Use ``arrays_to_Geo`` to convert a list of
        lists/arrays or an object array representing geometry.
    asGeo : boolean
        True, returns a Geo array. False returns a list of arrays.

    Notes
    -----
    >>> from npgeom.npg_plots import plot_polygons  # to plot the geometry

    `_isin_2d_`, `find`, `adjacent` equivalent::

        (b0[:, None] == b1).all(-1).any(-1)
    """

    def _adjacent_(a, b):
        """Check adjacency between 2 polygon shapes."""
        s = np.sum((a[:, None] == b).all(-1).any(-1))
        if s > 0:
            return True
        return False

    def _cycle_(b0, b1):
        """Cycle through the bits."""

        def _find_(a, b):
            """Find.  Abbreviated form of ``adjacent``, to use for slicing."""
            return (a[:, None] == b).all(-1).any(-1)

        idx01 = _find_(b0, b1)
        idx10 = _find_(b1, b0)
        if idx01.sum() == 0:
            return None
        if idx01[0] == 1:  # you can't split between the first and last pnt.
            b0, b1 = b1, b0
            idx01 = _find_(b0, b1)
        dump = b0[idx01]
        # dump1 = b1[idx10]
        sp0 = np.nonzero(idx01)[0]
        sp1 = np.any(np.isin(b1, dump, invert=True), axis=1)
        z0 = np.array_split(b0, sp0[1:])
        # direction check
        # if not (dump[0] == dump1[0]).all(-1):
        #     sp1 = np.nonzero(~idx10)[0][::-1]
        sp1 = np.nonzero(sp1)[0]
        if sp1[0] + 1 == sp1[1]:  # added the chunk section 2023-03-12
            print("split equal {}".format(sp1))
            chunk = b1[sp1]
        else:
            sp2 = np.nonzero(idx10)[0]
            print("split not equal {} using sp2 {}".format(sp1, sp2))
            chunks = np.array_split(b1, sp2[1:])
            chunk = np.concatenate(chunks[::-1])
        return np.concatenate((z0[0], chunk, z0[-1]), axis=0)

    def _combine_(r, shps):
        """Combine the shapes."""
        missed = []
        processed = False
        for i, shp in enumerate(shps):
            adj = _adjacent_(r, shp[1:-1])  # shp[1:-1])
            if adj:
                new = _cycle_(r, shp[:-1])  # or shp)
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
    # -- try sorting by  y coordinate rather than id, doesn't work
    # cent = a.centers()
    # s_ort = np.argsort(cent[:, 1])  # sort by y
    # shps = a.get_shapes(s_ort, False)
    # -- original, uncomment below
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
        out = npGeo.arrays_to_Geo(out, 2, "dissolved", False)
        out = npGeo.roll_coords(out)
    return out  # , missed


# ----------------------------------------------------------------------------
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
    if s > 0:
        return True
    return False


def adjacency_matrix(a, prn=False):
    """Construct an adjacency matrix from an input polygon geometry.

    Parameters
    ----------
    a : array-like
        A Geo array, list of lists/arrays or an object array representing
        polygon geometry.

    Returns
    -------
    An nxn array adjacency for polygons and a id-keys to convert row-column
    indices to their original ID values.

    The diagonal of the output is assigned the shape ID. Adjacent to a cell
    is denoted by assigning the shape ID.  Non-adjacent shapes are assigned -1

    Example::

        ad =adjacency_matrix(a)             # -- 5 polygons shapes
        array([[ 0, -1, -1,  3,  4],
               [-1,  1, -1, -1, -1],
               [-1, -1,  2, -1, -1],
               [ 0, -1, -1,  3,  4],
               [ 0, -1, -1,  3,  4]])
        ad >= 0                             # -- where there are links
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
                arr[arr == i[1]] = i[0]  # reclass back to original values
            # else:
            #     arr[arr == i[0]] = i[0]
        return arr
    #
    rings = _bit_check_(a, just_outer=True)  # get the outer rings
    ids = a.IDs[a.CL == 1]                   # and their associated ID values
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
        print("Adjacency matrix\n  n : poly ids\n" + "-" * 14)
        row_frmt = "{:>3.0f} : {!s:<15}"
        out = "\n".join([row_frmt.format(i, row[row != -1])
                         for i, row in enumerate(z)])
        print(out)
        return None
    return z


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
        return None
    if hasattr(this, "IFT"):
        a_stack = this.XY
        IFT = this.IFT
        if this.K != to_this.K:
            print("\nGeo array `kind` is not the same,\n")
            return None
    else:
        a_stack, IFT, extent = npGeo.array_IFT(this)
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
    out = npGeo.Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
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
        b = npGeo.arrays_to_Geo(b)
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
        a_XY, a_IFT, extent = npGeo.array_IFT(a)
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
    out = npGeo.Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
    return out


# ----------------------------------------------------------------------------
# ---- (7) union geometry
#
def union_as_one(a, b):
    """Union polygon features with a dissolve of internal shared edges.

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
    # -- get the intersection points
    x_sect = p_ints_p(a, b)
    if x_sect is None:
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


# ----------------------------------------------------------------------------
# ---- (8) `crossing` and related methods ------------------------------------
# related functions
# See : line_crosses, in_out_crosses
#  pnt_right_side : single point relative to the line
#  line_crosses   : checks both segment points relative to the line
#  in_out_crosses # a variant of the above, with a different return signature

def line_crosses(p0, p1, p2, p3, as_integers=False):
    """Determine if a line is `inside` another line segment.

    Parameters
    ----------
    p0, p1, p2, p3 : array-like
        X,Y coordinates of the subject (p0-->p1) and clipping (p2-->p3) lines.

    Returns
    -------
    The result indicates which points, if any, are on the inward bound side of
    a polygon (aka, right side). The clip edge (p2-->p3) is for clockwise
    oriented polygons and its segments. If `a` and `b` are True, then both are
    inside.  False for both means that they are on the outside of the clipping
    segment.
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3
    dc_x = x3 - x2
    dc_y = y3 - y2
    # -- check p0 and p1 separately and return the result
    a = (y0 - y2) * dc_x <= (x0 - x2) * dc_y
    b = (y1 - y2) * dc_x <= (x1 - x2) * dc_y
    if as_integers:
        return a * 1, b * 1
    return a, b


def in_out_crosses(*args):
    """Return whether two line segments cross.

    Line segment (p0-->p1) is crossed by a cutting/clipping
    segment (p2-->p3).  `inside` effectively means `right side` for clockwise
    oriented polygons.

    Parameters
    ----------
    p0p1, p2p3 : line segments
        Line segments with their identified start-end points, as below
    p0, p1, p2, p3 : array-like
        X,Y coordinates of the subject (p0-->p1) and clipping (p2-->p3) lines.

    Requires
    --------
    `_line_crosses_` method.

    Returns
    -------
    -  2 both segment points are inside the clipping segment.
    -  1 start point is inside.
    -  0 both points are outside.
    - -1 end point is inside.

    """
    msg = "\nPass 2, 2-pnt lines or 4 points to the function.\n"
    args = np.asarray(args)
    if np.size(args) == 8:
        if len(args) == 2:  # two lines
            p0, p1, p2, p3 = *args[0], *args[1]
        elif len(args) == 4:  # four points
            p0, p1, p2, p3 = args
        else:
            print(msg)
            return None
    else:
        print(msg)
        return None
    # --
    a, b = line_crosses(p0, p1, p2, p3)
    a = a * 1
    b = b * 1
    if (a == 1) and (b == 1):    # both on right
        return 2
    elif (a == 1) and (b == 0):  # start on right
        return 1
    elif (a == 0) and (b == 0):  # both on left
        return 0
    elif (a == 0) and (b == 1):  # end on right
        return -1
    else:
        return -999


def crossings(geo, clipper):
    """Determine if lines cross. multiline implementation of above."""
    if hasattr(geo, "IFT"):
        bounds = dissolve(geo)  # **** need to fix dissolve
    else:
        bounds = geo
    p0s = bounds[:-1]
    p1s = bounds[1:]
    p2s = clipper[:-1]
    p3s = clipper[1:]
    n = len(p0s)
    m = len(p2s)
    crosses_ = []
    for j in range(m):
        p2, p3 = p2s[j], p3s[j]
        for i in range(n):
            p0, p1 = p0s[i], p1s[i]
            crosses_.append(intersects(p0, p1, p2, p3))  # this seems to work
    return crosses_


# ----------------------------------------------------------------------------
# ---- (9) polygon from points
#
def left_right_pnts(a):
    """Return the two points that contain the min and max ``X`` coordinate.

    Notes
    -----
    These points are used to form a line.  This line can be used by
    ``line_side`` to classify points with respect to it.  Classification is
    obviously based on the direction the line points.
    If there are duplicate x values, then the first is taken.  In either case,
    an array of (2,) is returned.  This could be altered to take the higher y
    value or the y average for that x.
    """
    srted = np.sort(a[:, 0])
    min_x, max_x = srted[[0, -1]]
    lft = a[np.where(a[:, 0] == min_x)[0]]
    rght = a[np.where(a[:, 0] == max_x)[0]]
    return np.array([lft[0], rght[0]])


def line_side(pnts, line=None):
    """Return the side of a line that the points lie on.

    Parameters
    ----------
    pnts : array-like
        The points to examine as an Nx2 array.
    line : array-like or None
        If None, then the left, right-most points are used to construct it.

    References
    ----------
    `<https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point
    -is-to-the-right-or-left-side-of-a-line>`_.

    Notes
    -----
    - Above the line (+ve) is left.
    - Below the line is right (-ve).
    - Zero (0) is on the line.
    A-B is the line.    X,Y is the point.  This is vectorized by numpy using.

    >>> sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
    """
    if line is None:
        A, B = line = np.array(left_right_pnts(pnts))
    else:
        A, B = line
    BAx, BAy = line[1] - line[0]
    XAx = pnts[:, 0] - A[0]
    YAy = pnts[:, 1] - A[1]
    return np.sign(np.int32((BAx * YAy - BAy * XAx) + 0.0))  # -- ref 2


def _line_crossing_(clip_, poly):
    """Determine if a line is `inside` another line segment.

    Notes
    -----
    The original ***
    See _line_cross_ in npg_clip ***

    Multi-line implementation of line_crosses.
    Used by ``z``.

    points inside clipper
    >>> w0.all(0).nonzero()[0]         # both on right of all segments
    >>> (w1 == 1).any(0).nonzero()[0]  # start on right
    >>> (w2 == 1).any(0).nonzero()[0]  # end on right
    >>> (w3 == 1).any(0).nonzero()[0]  # both on left of all segments
    """
    x0s, y0s = poly[:-1].T   # x0s = poly[:-1, 0], y0s = poly[:-1, 1]
    x1s, y1s = poly[1:]. T   # x1s = poly[1:, 0],  y1s = poly[1:, 1]
    x2s, y2s = clip_[:-1].T  # x2s = clip_[:-1, 0], y2s = clip_[:-1, 1]
    x3s, y3s = clip_[1:].T   # x3s = clip_[1:, 0],  y3s = clip_[1:, 1]
    dcx = x3s - x2s
    dcy = y3s - y2s
    a_0 = (y0s - y2s[:, None]) * dcx[:, None]
    a_1 = (x0s - x2s[:, None]) * dcy[:, None]
    b_0 = (y1s - y2s[:, None]) * dcx[:, None]
    b_1 = (x1s - x2s[:, None]) * dcy[:, None]
    a = (a_0 <= a_1)
    b = (b_0 <= b_1)
    w0 = np.logical_and(a == 1, b == 1)   # both on right
    z0 = np.where(w0 == 1, 2, 0)          # 2
    w1 = np.logical_and(a == 1, b == 0)   # start on right
    z1 = np.where(w1 == 1, 1, 0)          # 1
    w2 = np.logical_and(a == 0, b == 1)   # end on right
    z2 = np.where(w2 == 1, -1, 0)         # -1
    w3 = np.logical_and(a == 0, b == 0)   # both on left
    z3 = np.where(w3 == 1, -2, 0)         # -2
    z = z0 + z1 + z2 + z3
    return z, a, b


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
#     in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polylines2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygon2pnts"
# python
