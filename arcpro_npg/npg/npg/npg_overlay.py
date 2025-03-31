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
    2025-02-17

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

`Web link on intersections
`<https://stackoverflow.com/questions/79409653
/how-to-efficiently-check-if-two-line-segments-intersect>`_.

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
# from npg import npGeo
from npg.npg_bool_hlp import p_ints_p
from npg.npg_bool_ops import union_adj
# from npg.npg_pip import np_wn
# import npg_geom_hlp
from npg.npg_geom_hlp import _in_LBRT_
from npg.npg_helpers import _to_lists_

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


__all__ = [
    'intersections',
    'intersects',
    'line_crosses', 'in_out_crosses', 'crossings',
    'left_right_pnts', 'line_side', '_line_crossing_'
]
__helpers__ = ['_intersect_']


# ---- ---------------------------
# ---- (1) helpers/mini functions
#

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

    `<https://stackoverflow.com/questions/79409653
    /how-to-efficiently-check-if-two-line-segments-intersect>`_.

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


# ---- ---------------------------
# ---- (2) intersect geometry
#  `p_ints_p` is the main function
#  `intersections` uses this to batch intersect multiple polygons as input
#  and intersectors.
#


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
    ``_to_lists_`` from npg_geom_hlp

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


def intersects(args):
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
            r = intersects([ln, j])
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

    `<https://stackoverflow.com/questions/79409653
    /how-to-efficiently-check-if-two-line-segments-intersect>`_.

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


# ---- ---------------------------
# ---- (3) `crossing` and related methods ------------------------------------
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
    msg = "\nPass 2, 2-pnt lines, 4 points or 8 coordinates to the function.\n"
    args = np.asarray(args)
    if np.size(args) == 8:
        if len(args) == 2:  # two lines
            p0, p1, p2, p3 = *args[0], *args[1]
        elif len(args) == 4:  # four points
            p0, p1, p2, p3 = args
        elif len(args) == 8:
            p0, p1, p2, p3 = args.reshape(4, 2)
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
        bounds = union_adj(geo)  # **** need to fix dissolve
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
            crosses_.append(intersects([p0, p1, p2, p3]))  # this seems to work
    return crosses_


# ---- ---------------------------
# ---- (4) polygon from points
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
