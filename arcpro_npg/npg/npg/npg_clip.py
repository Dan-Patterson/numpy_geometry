# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
--------
npg_clip
--------

----

Script :
    npg_clip.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2021-01-10

Purpose
-------
Functions for clipping polygons.
"""

import sys
# from textwrap import dedent
import numpy as np

# -- optional numpy imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

if 'npg' not in list(locals().keys()):
    import npg
from npg_helpers import _to_lists_
from npg_plots import plot_polygons

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['clip_polygons']


# ----------------------------------------------------------------------------
# ---- (1) clip geometry
# `wp_wn` for point in polygon
# `_p_c_p_` and `_clip_` for poly intersections
# `clip_polygons` for multiple clips

def split_line_at_point(a, b, c, as_pairs=True):
    """Split line segment (a->b) at point (c).

    Parameters
    ----------
    a, b, c : points (x, y)
        Their x, y coordinates in planar units.
    as_pairs : boolean
        True, returns two point pairs::

          [[xa, ya], [xc, yc]]  # start point to intersection point
          [[xc, yc], [xb, yb]]  # intersection point to end point

        False returns the segments as raveled coordinates::

          [xa, ya, xc, yc]  # start point to intersection point
          [xc, yc, xb, yb]  # intersection point to end point.

        `None` is returned in both cases if there is no intersection.
    """
    out = []
    if is_on(a, b, c):
        out = np.asarray([[a, c], [c, b]])
        if as_pairs:
            return out
        else:
            return out.reshape(2, 4)
    return None


def is_on(a, b, c):
    """Determine whether a point (c) lies on a line segment (a->b).

    Parameters
    ----------
    a, b, c : points
        The x, y coordinates of their locations.

    Requires
    --------
    ``collinear`` and ``within``.  A builtin tolerance of 1.0e-12 to avoid
    floating point issues.

    Returns
    -------
    True, if `c` intersects the segment `a` to `b`

    Example
    -------
    >>> ft_p = np.array([  5.000,  15.000,   7.000,  14.000])
    >>> pnt = nparray([  6.350,  14.325])
    >>> is_on(ft_p[ply, :2], ft_p[ply, -2:], pnt)
    ... True
    """
    # (or the degenerate case that all 3 points are coincident)
    return (collinear(a, b, c)
            and (within(a[0], c[0], b[0]) if a[0] != b[0] else
                 within(a[1], c[1], b[1])))


def collinear(a, b, c):
    """Return True if all points lie on the same line."""
    epsilon = 1.0e-12
    abc = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    return abs(abc) < epsilon


def within(p, q, r):
    """Return true if `q` is between `p` and `r` (inclusive)."""
    return p <= q <= r or r <= q <= p


def between(strt, end, pnt):
    """Return whether `pnt` is between the start and end point of a line.

    Returns
    -------
    -  1 if strt=>end is ascending in x
    - -1 if descending
    -  0 otherwise
    """
    if strt[0] != end[0]:
        up_ = np.logical_and(strt[0] <= pnt[0], pnt[0] <= end[0])
        if up_:
            return 1
        down_ = np.logical_and(strt[0] >= pnt[0], pnt[0] >= end[0])
        if down_:
            return -1
        if not (up_ or down_):
            return 0
    else:
        up_ = np.logical_and(strt[1] <= pnt[1],  pnt[1] <= end[1])
        if up_:
            return 1
        down_ = np.logical_and(strt[1] >= pnt[1], pnt[1] >= end[1])
        if down_:
            return -1
        if not (up_ or down_):
            return 0


# -- from npg_overlay
def p_c_p(clipper, poly):
    """Intersect two polygons.  Abbreviated from ``p_ints_p``.

    Parameters
    ----------
    clipper, poly : polygon
         Clockwise-ordered sequence of points (x, y) with the first and last
         point being the same. The `clipper` is the polygon cutting the `poly`
         geometry.

         - Polygons must not be self-intersecting.
         - Holes are ignored.

    Notes
    -----
    ::

               col       row
               on poly,  on clipper  clipper has 4 segments
        array([[0, 1, 1, 0, 0, 0],   polygon has 6
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1],
               [1, 0, 0, 0, 0, 1]])
        if 0 <= u_a, u_b <= 1 then the intersection is on both segments

    Notes
    -----
    - np.any(both, axis=1)
    - yields the segment on the clipper that the points are on
    - np.sum(both, axis=1)  how many intersections on clipper
    - np.sum(both, axis=0)  intersections on the polygon
    """
    p_cl, c_cl = [i.XY if hasattr(i, "IFT") else i for i in [poly, clipper]]
    p10_x, p10_y = (p_cl[1:] - p_cl[:-1]).T
    dc = c_cl[1:] - c_cl[:-1]
    dc_x = dc[:, 0][:, None]
    dc_y = dc[:, 1][:, None]
    pc02 = p_cl[:-1] - c_cl[:-1][:, None]
    pc02_x = pc02[..., 0]
    pc02_y = pc02[..., 1]
    # --
    d_nom = (p10_x * dc_y) - (p10_y * dc_x)
    a0 = pc02_y * dc_x
    a1 = pc02_x * dc_y
    b0 = p10_x * pc02_y
    b1 = p10_y * pc02_x
    a_num = a0 - a1  # (p02_y * dc_x) - (p02_x * dc_y)
    b_num = b0 - b1  # (p10_x * p02_y) - (p10_y * p02_x)
    with np.errstate(all='ignore'):  # divide='ignore', invalid='ignore'):
        u_a = a_num/d_nom
        u_b = b_num/d_nom
        z0 = np.logical_and(u_a >= 0., u_a <= 1.)
        z1 = np.logical_and(u_b >= 0., u_b <= 1.)
        both = z0 & z1
        xs = u_a * p10_x + p_cl[:-1][:, 0]
        ys = u_a * p10_y + p_cl[:-1][:, 1]
    # xs = xs[both];  ys = ys[both];  whre = np.vstack(both.nonzero()).T
    bothT = both.T
    xs = xs.T[bothT]
    ys = ys.T[bothT]
    poly_clipper_ids = np.vstack(np.nonzero(bothT)).T
    # clipper_poly_ids = np.vstack(np.nonzero(both)).T
    if xs.size > 0:
        final = np.zeros((len(xs), 2))
        final[:, 0] = xs
        final[:, 1] = ys
        return final, bothT, poly_clipper_ids
    return None, bothT, None


def _wn_(pnts, poly, return_winding=True):
    """Return points in polygon using `winding number`.  See npg_pip."""
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    x, y = pnts.T         # point coordinates
    y_y0 = y[:, None] - y0
    x_x0 = x[:, None] - x0
    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0
    chk1 = (y_y0 >= 0.0)  # y values above poly's first y value, per segment
    chk2 = np.less(y[:, None], y1)  # y values above the poly's second point
    chk3 = np.sign(diff_).astype(np.int)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn_vals = pos - neg
    out_ = pnts[np.nonzero(wn_vals)]
    if return_winding:
        return out_, wn_vals  # np.vstack((wn_vals[:-1], wn_vals[1:])).T
    return out_


def _line_cross_(poly, clipper):
    """Determine if a line is `inside` another line segment.

    Multi-line implementation of line_crosses.
    Used by ``z``.

    Returns
    -------
    The result indicates which points, if any, are on the inward bound side of
    a polygon (aka, right side). The clip edge (p2-->p3) is for clockwise
    oriented polygons and its segments. If `a` and `b` are True, then both are
    inside.  False for both means that they are on the outside of the clipping
    segment.

    points inside clipper
    >>> z0 = w0.all(0).nonzero()[0]         # both on right of all segments
    >>> z1 = (w1 == 1).any(0).nonzero()[0]  # start on right
    >>> z2 = (w2 == 1).any(0).nonzero()[0]  # end on right
    >>> z3 = (w3 == 1).any(0).nonzero()[0]  # both on left of all segments
    """
    x0s, y0s = poly[:-1].T     # x0s = poly[:-1, 0], y0s = poly[:-1, 1]
    x1s, y1s = poly[1:]. T     # x1s = poly[1:, 0],  y1s = poly[1:, 1]
    x2s, y2s = clipper[:-1].T  # x2s = clip_[:-1, 0], y2s = clip_[:-1, 1]
    x3s, y3s = clipper[1:].T   # x3s = clip_[1:, 0],  y3s = clip_[1:, 1]
    dcx = x3s - x2s
    dcy = y3s - y2s
    a_0 = (y0s - y2s[:, None]) * dcx[:, None]
    a_1 = (x0s - x2s[:, None]) * dcy[:, None]
    b_0 = (y1s - y2s[:, None]) * dcx[:, None]
    b_1 = (x1s - x2s[:, None]) * dcy[:, None]
    a = (a_0 <= a_1) * 1
    b = (b_0 <= b_1) * 1
    #
    # -- First column is the clipper, second the poly for w0/z0,... w3/z3 etc
    # -- clipper both outside poly
    w0 = np.logical_and(a == 1, b == 1).T
    z0a = w0.all(0).nonzero()[0]  # clip seg outside poly
    z0b = w0.all(1).nonzero()[0]  # poly seg inside clip
    #
    # -- seg start on inside/right or intersects
    w1 = np.logical_and(a == 1, b == 0).T
    z1a = (w1 == 1).any(0).nonzero()[0]  # clip crosses poly, start out=>in?
    z1b = (w1 == 1).any(1).nonzero()[0]  # poly crosses clip, start  in=>outZ
    idx1 = np.vstack(np.nonzero(w1)).T   # summarized here
    #
    # -- seg end on inside/right or intersects
    w2 = np.logical_and(a == 0, b == 1).T
    z2a = (w2 == 1).any(0).nonzero()[0]  # clip crosses poly, start  in=>out?
    z2b = (w2 == 1).any(1).nonzero()[0]  # poly crosses clip, start out=>in
    idx2 = np.vstack(np.nonzero(w2)).T
    #
    # -- both on outside/left
    w3 = np.logical_and(a == 0, b == 0).T
    z3a = (w3 == 1).any(0).nonzero()[0]  # clip seg with poss intx or poly in
    z3b = (w3 == 1).any(1).nonzero()[0]  # poly seg completely outside
    z = [z0a, z0b, z1a, z1b, z2a, z2b, z3a, z3b]
    return z, idx1, idx2


def _clip_(poly, clipper):
    """Return the result of a polygon clip.

    Parameters
    ----------
    poly, clipper : ndarrays
        Arrays representing polygon geometry.  `clip_poly` is the clipping
        feature and `poly` is the feature being clipped.

    Requires
    --------
    ``_wn_`` and ``p_c_p`` check for point inclusion and line crossings type.

    Notes
    -----
    p_in_c, c_in_p : polygons
        Polygon points in clipper and clipper points in polygon.
    wn0, wn1 :  winding number
        Values of 0 are outside and -1 for points inside the clipper, for
        clockwise oriented polygons.
    x_sect, bothT, pl_cl_ids : arrays
        Intersection points, intersection code matrix and polygon-clipper ids

    # ******
    _all = np.vstack((c_in_p, p_in_c, x_sect))
    uni, idx = np.unique(_all, True, axis=0)
    _all[np.sort(idx)]
    _all[np.lexsort((_all[:, 1], _all[:, 0]))]
    """
    p_in_c, wn0 = _wn_(poly, clipper, True)
    c_in_p, wn1 = _wn_(clipper, poly, True)
    p_in_c_ft = np.abs(np.array([wn0[:-1], wn0[1:]]).T)
    # c_in_p_ft = np.abs(np.array([wn1[:-1], wn1[1:]]).T)
    #
    # -- exploring whether wn can be used for point insertion
    # n = np.arange(len(wn0) - 1)[None, :]
    # n0 = np.abs(np.array([wn0[:-1], wn0[1:]]))
    # pc_ft = np.concatenate((n, n0)).T
    # wh = np.nonzero(wn0)[0]  # points that are inside
    #
    # produce from-to pairs for both polygons
    fr_to = [np.concatenate((arr[:-1], arr[1:]), axis=1)
             for arr in [clipper, poly]
             ]
    clip_ft, poly_ft = fr_to
    #
    # -- get the intersection points and their location
    x_sect, bothT, pl_cl_ids = p_c_p(clipper, poly)
    #
    if x_sect is None:
        # print("No intersections.")
        return None
    else:
        x_x, idx = np.unique(x_sect, True, axis=0)
        x_sect = x_sect[np.sort(idx)]
    out = []
    ft_p = np.copy(poly_ft)
    ft_c = np.copy(clip_ft)
    cnt = 0  # intersection counter to extract the intersection points in order
    N = len(p_in_c_ft)  # (pl_cl_ids)  # fixed an issue
    # tested = np.vstack([i for i in [p_in_c, c_in_p, x_sect]
    #                     if i is not None])
    p_prev, c_prev = p_in_c_ft[0]
    for i, o_i in enumerate(p_in_c_ft):  # o_i = p_in_c_ft[0]
        #
        if (i <= N) and (cnt < len(pl_cl_ids)):
            ply, clp = pl_cl_ids[cnt]
            p_strt, p_end = ft_p[ply, :2], ft_p[ply, -2:]
            c_strt, c_end = ft_c[clp, :2], ft_c[clp, -2:]
            pnt = x_sect[cnt]
        else:
            break
        if sum(o_i) == 0:
            chk0 = is_on(p_strt, p_end, pnt)  # pnt on poly?
            chk1 = is_on(c_strt, c_end, pnt)  # pnt on clipper?
            if (chk0 & chk1) & (cnt > 0):
                if clp == pl_cl_ids[cnt - 1][1]:  # previous
                    ft_p[ply, :2] = pnt
                else:
                    ft_p[ply, -2:] = pnt   # ft_c[clp, -2:] = pnt
                out.append(pnt)
                cnt += 1
            elif chk0:                       # ***
                if p_strt[0] <= c_strt[0]:   # ***
                    ft_p[ply, :2] = pnt      # ***
                    out.append(pnt)          # ***
                    cnt += 1                 # ***
                # continue
        elif sum(o_i) == 2:
            out.extend(ft_p[i].reshape(2, 2))  # this has to be `i` not `cnt
        elif (o_i == (0, 1)).all():         # start out end in
            out.append(pnt)
            if (cnt == 0):  # and (i == 0):   # added and (i==0) ****
                out.append(p_end)  # last point
            if (pnt == c_in_p).all(-1).any():  # add end if c_in_p
                if not (pnt == out).all(-1).any():  # ***
                    out.append(pnt)
            if (p_prev == ply) and (cnt > 0):  # renumber if ft_c has > 1 xsect
                ft_c[c_prev, -2:] = pnt
            cnt += 1
        elif (o_i == (1, 0)).all():         # start in end out
            if (cnt == 0):
                out.append(p_strt)
            p_end = pnt
            ft_c[clp, :2] = pnt
            # ft_c[clp, -2:] = pnt
            out.append(pnt)
            if (c_end == c_in_p).all(-1).any():  # add end if c_in_p
                out.append(c_end)
            cnt += 1
        # -- store previous edges
        p_prev, c_prev = ply, clp
    # --
    if out:
        if not (x_sect[-1] == out).all(-1).any():  # needed for b0, c0
            out.append(x_sect[-1])
        out = np.vstack((out, out[0]))
        return out  # , ft_p, ft_c
    return None


# -- main calling script for multiple inputs and clippers
#
def clip_polygons(polys, clippers, full_details=True):
    """Return batch clipped polygons."""
    # -- process
    polys = _to_lists_(polys, True)
    clippers = _to_lists_(clippers, True)
    output = []
    ij = []
    for i, clipp in enumerate(clippers):
        for j, p in enumerate(polys):
            result = _clip_(p, clipp)
            if result is not None:
                ij.append([i, j])
                output.append(result)
    if full_details:
        return output, ij
    return output


# ----------------------------------------------------------------------------
# ---- (2) extra clip helpers
#
def _clip_line_crosses_(p, c, as_integers=False):
    """Determine if a line's points are `inside` another line segment.

    See Also
    --------
    ``line_crosses`` in npg_overlays

    Returns
    -------
    The result indicates which points, if any, are on the inward bound side of
    a polygon (aka, right side). The clip edge (p2-->p3) is for clockwise
    oriented polygons and its segments. If `a` and `b` are True, then both are
    inside.  False for both means that they are on the outside of the clipping
    segment.
    """
    x0, y0, x1, y1 = p  # polygon segment points
    x2, y2, x3, y3 = c  # clipper segment points
    dc_x = x3 - x2
    dc_y = y3 - y2
    # -- check p0 and p1 separately and return the result
    a = (y0 - y2) * dc_x <= (x0 - x2) * dc_y
    b = (y1 - y2) * dc_x <= (x1 - x2) * dc_y
    if as_integers:
        return a*1, b*1
    return a, b


def _clip_intersects_(c, p):
    """Return intersections.  See ``npg_overlays.intersects`` for details.

    >>> c = np.array([  0.000,   7.500,  11.000,  10.500])
    >>> p = np.array([ 10.000,  10.000,   8.000,  10.000])
    >>> _clip_intersects_(c_pairs[0], p_pairs[-1])
    ... (True, (9.166666666666668, 10.0))
    """
    msg = "Each element for c and p requires 4 values... see header"
    if len(c) == 4 and len(p) == 4:
        p0 = c[:2]
        p1 = c[2:]
        p2 = p[:2]
        p3 = p[2:]
    else:
        raise AttributeError(msg)
    #
    x0, y0 = p0
    x10, y10 = p1 - p0
    x32, y32 = p3 - p2
    x02, y02 = p0 - p2
    denom = x10 * y32 - y10 * x32
    if denom == 0.0:              # (1) denom = 0 ... collinear/parallel
        return (False, None)
    denom_gt0 = denom > 0         # denominator greater than zero
    s_num = x10 * y02 - y10 * x02
    if (s_num < 0) == denom_gt0:  # (2) (s_n < 0) == (denom > 0) : False
        return (False, None)
    t_num = x32 * y02 - y32 * x02
    if (t_num < 0) == denom_gt0:  # (3) (t_n < 0) == (denom > 0) : False
        return (False, None)
    t4 = np.logical_or(
        (s_num > denom) == denom_gt0, (t_num > denom) == denom_gt0)
    if t4:                        # (4) numerator checks fail
        return (False, None)
    t = t_num / denom
    x = x0 + t * x10
    y = y0 + t * y10
    #                               (5) check for input point intersection
    if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
        return (True, (x, y))
    return (True, (x, y))


def _clip_in_out_crosses_(c, p):
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
    `__clip_line_crosses__` method

    Returns
    -------
    -  2 both segment points are inside the clipping segment.
    -  1 start point is inside
    -  0 both points are outside
    - -1 end point is inside

    """
    msg = "Each element for c and p requires 4 values... see header"
    if len(c) == 4 and len(p) == 4:
        p0 = c[:2]
        p1 = c[2:]
        p2 = p[:2]
        p3 = p[2:]
    else:
        raise AttributeError(msg)
    # --
    a, b = _clip_line_crosses_(p0, p1, p2, p3)
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


def _which_side_(pnt, line):
    """Return line side that a point is on relative to a polygon segment.

    >>> sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))

    Notes
    -----
    Normally test the start or end point of a line segment to see if it crosses
    a polygon clipping segment.

    negative for inside, positive for outside
    """
    if len(line) == 4:
        line = line.reshape(2, 2)
    A, B = line                   # -- A to B, start to end
    BAx, BAy = line[1] - line[0]  # -- B - A
    XAx = pnt[0] - A[0]
    YAy = pnt[1] - A[1]
    return np.sign(BAx * YAy - BAy * XAx).astype('int')


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
