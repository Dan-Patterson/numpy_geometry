# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
---------
npg_clip
---------

----

Script :
    npg_clip.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2021-03-10

Purpose
-------
Functions for clipping polygons.

Call sequence::

    # c_in, _cut_, _inouton_ and _wn_clip_ can all be used without poly_clip
    # _inouton_ and _wn_clip have no other dependencies.
    poly_clip
    ...   |__ c_in
    ...        |__ _cut_ __
    ...        |          |__ _inouton_
    ...        |__ _wn_clip_

Notes
-----
Examples::

    p0 = [[[0., 8.], [1., 6.]], [[0., 8.], [2., 8]],
          [[0., 8.], [4., 9]], [[0., 8.], [2., 10.]]]
    poly = [np.array(p) for p in p0]
    pnts = np.array(poly).reshape(-1, 2)
    poly = np.array([[0.0, 5.0], [2., 10.], [4., 10.], [2., 5.0], [0., 5.]])
    plot_polygons([pnts, poly])

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
# from npg_helpers import _to_lists_
from npg_plots import plot_polygons

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft)

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['common_extent', 'c_in', 'poly_clip', 'winding_num']
__helpers_ = [
    '_onseg_', '_is_on_', '_side_', '_inouton_', '_cut_', '_wn_clip_'
    ]


# ----------------------------------------------------------------------------
# ---- (1) helpers
#
def common_extent(a, b):
    """Return the extent overlap for two polygons as L, B, R, T or None."""
    ext0 = np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))
    ext1 = np.concatenate((np.min(b, axis=0), np.max(b, axis=0)))
    es = np.vstack((ext0, ext1))
    maxs = np.max(es, axis=0)
    mins = np.min(es, axis=0)
    L, B = maxs[:2]
    R, T = mins[2:]
    if (L <= R) and (B <= T):
        return (True, np.array([L, B, R, T]))  # (x1, y1, x2, y2)
    return (False, None)


def _onseg_(poly, pnts, tol=1.0e-12):
    """Determine whether points (pnts) lies on line segments (poly).

    Parameters
    ----------
    poly : array_like
        Polyline/polygon objects represented as Nx2 arrays of points.  Every
        segment of poly features are tested.
    pnts : array_like
        The x, y coordinates of point locations.
    tol : float
        The tolerance of accepting whether a point is on a segment.

    Requires
    --------
    ``_collinear_`` and ``_within_``.  A default tolerance of 1.0e-12 to avoid
    floating point issues.

    Returns
    -------
    An array of the ids of the ``pnts`` versus ``poly`` segments.

    Example
    -------
    >>> ft_p = np.array([  5.000,  15.000,   7.000,  14.000])
    >>> pnt = np.array([  6.350,  14.325])
    >>> _onseg_(ft_p[:2], ft_p[-2:], pnt)
    ... True

    """
    def _collinear_(a, b, pnts):
        """Return True if all points lie on the same line.

        This is the same ``_side_`` check.
        >>> r = (x1 - x0) * (y[:, None] - y0) - (y1 - y0) * (x[:, None] - x0)
        """
        epsilon = tol
        x0, y0 = a.T
        x1, y1 = b.T
        x, y = pnts.T
        r0 = (x1 - x0) * (y[:, None] - y0)
        r1 = (x[:, None] - x0) * (y1 - y0)
        r01 = r0 - r1
        return abs(r01) < epsilon

    def _within_(a, b, pnts):
        """Return."""
        x = pnts[:, 0][:, None]
        y = pnts[:, 1][:, None]
        x0, y0 = a.T
        x1, y1 = b.T
        z0 = (x0 <= x) & (x <= x1)
        z1 = (y0 <= y) & (y <= y1)
        # or
        z2 = (x1 <= x) & (x <= x0)
        z3 = (y1 <= y) & (y <= y0)
        res = np.logical_or(z0 & z1, z2 & z3)
        return res
    # --
    a = poly[:-1]
    b = poly[1:]
    r0 = _collinear_(a, b, pnts)
    r1 = _within_(a, b, pnts)
    whr = r0 & r1
    return np.array(np.nonzero(whr)).T


def _is_on_(a, b, pnt):
    """Return whether a point, (pnt) lies on a line segment (a->b).

    Or the degenerate case where all 3 points coincide.

    See Also
    --------
    ``_onseg_`` is the multi poly-pnt version.
    """

    def collinear(a, b, pnt):
        """Return True if all points lie on the same line."""
        epsilon = 1.0e-12
        r = (b[0] - a[0]) * (pnt[1] - a[1]) - (pnt[0] - a[0]) * (b[1] - a[1])
        return abs(r) < epsilon

    def within(a, pnt, b):
        """Return true if `pnt` is between `a` and `b` (inclusive)."""
        return a <= pnt <= b or b <= pnt <= a
    # --
    return (collinear(a, b, pnt)
            and (within(a[0], pnt[0], b[0]) if a[0] != b[0] else
                 within(a[1], pnt[1], b[1])))


def _side_(pnts, poly):
    r"""Return points inside, outside or equal/crossing a convex poly feature.

    Notes
    -----
    >>> `r` == diff_ in _wn_ used in chk3
    >>> `r` == t_num = a_0 - a_1 ... in previous equations
    >>> r_lt0 = r < 0, r_gt0 = ~r_lt0, to yield =>  (r_lt0 * -1) - (r_gt0 + 0)
    >>> (r < 0).all(-1)  # just the boolean locations
    ... array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    >>> (r < 0).all(-1).nonzero()[0]  # the index numbers
    ... array([2, 3, 4, 6, 7], dtype=int64)
    """
    if pnts.ndim < 2:
        pnts = np.atleast_2d(pnts)
    x0, y0 = pnts.T
    x2, y2 = poly[:-1].T  # poly segment start points
    x3, y3 = poly[1:].T   # poly segment end points
    r = (x3 - x2) * (y0[:, None] - y2) - (y3 - y2) * (x0[:, None] - x2)
    # -- from _wn_, winding numbers for concave/convex poly
    chk1 = ((y0[:, None] - y2) >= 0.)
    chk2 = (y0[:, None] < y3)
    chk3 = np.sign(r).astype(np.int)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn_vals = pos - neg
    in_ = pnts[np.nonzero(wn_vals)]
    inside = pnts[(r < 0).all(axis=-1)]  # all must be True along row, convex
    outside = pnts[(r > 0).any(-1)]      # any must be True along row
    equal_ = pnts[(r == 0).any(-1)]      # ditto
    return r, in_, inside, outside, equal_


# ---- (2) poly_clip

def _inouton_(pnts, poly):
    r"""Return polygon overlay results.  These are `in`, `out` or `on` points.

    Parameters
    ----------
    pnts : array_like
        The polygon points that are being queried (Nx2 shape).
        These can also be any points that would yield points inside, outside
        the `poly`gon feature or intersecting with its boundary.
    poly : array_like
        The polygon feature with shape Nx2 with the first and last points
        being the same.

    Returns
    -------
    The following ndarrays::

    - x_ings  : the intersection array
    - x_type  : crossing type, inside-outside or outside-inside
    - whr     : segment pairs where intersections occur
    - inside  : points inside the polygon
    - outside : points outside the polygon
    - x_pnts  : intersection/crossing points
    - eq      : points in common to both shapes

    Counterclockwise check for 3 points::

        def ccw(A, B, C):
            '''Tests whether the turn formed by A, B, and C is ccw'''
            return (B.x - A.x) * (C.y - A.y) > (B.y - A.y) * (C.x - A.x)

    Notes
    -----
    Point in/out of polygon uses `winding number` approach.
    If `denom` is 0, then the lines are coincedent or parallel.

    *intersection section*::

        denom = x1_x0 * y3_y2 - y1_y0 * x3_x2
        s_num = x1_x0 * y0_y2 - y1_y0 * x0_x2
        t_num = x3_x2 * y0_y2 - y3_y2 * x0_y2
          none/False
        denom == 0.0
        (s_num < 0.) == (denom > 0.0)
        (t_num < 0.) == (denom > 0.0)
        t4 = np.logical_or((s_num > denom) == (denom > 0.0),
                           (t_num > denom) == (denom > 0.0))
    ----

    """
    pnts, poly = [i.XY if hasattr(i, "IFT") else i for i in [pnts, poly]]
    # -- point equality check, not_eq = ~eq_
    eq_ = (pnts[:, None] == poly).all(-1).any(-1)  # pnts equal those on poly
    # -- pnts
    x0, y0 = pnts[:-1].T
    x1, y1 = pnts[1:]. T
    x2, y2 = poly[:-1].T
    x3, y3 = poly[1:].T
    #
    x1_x0, y1_y0 = (pnts[1:] - pnts[:-1]).T  # deltas for pnts
    x3_x2, y3_y2 = (poly[1:] - poly[:-1]).T  # deltas for poly
    x3_x2 = x3_x2[:, None]      # reshape poly deltas
    y3_y2 = y3_y2[:, None]
    x0_x2 = (x0 - x2[:, None])  # deltas between pnts/poly x and y
    y0_y2 = (y0 - y2[:, None])
    y1_y2 = (y1 - y2[:, None])
    x1_x2 = (x1 - x2[:, None])
    #
    a_0 = y0_y2 * x3_x2  # pc02_y * dc_x
    a_1 = x0_x2 * y3_y2  # pc02_x * dc_y
    b_0 = x1_x0 * y0_y2  # b_0 = p10_x * pc02_y  # y1_y2 * x3_x2 wrong
    b_1 = y1_y0 * x0_x2  # b_1 = p10_y * pc02_x  # x1_x2 * y3_y2 wrong
    c_0 = y1_y2 * x3_x2  # used for x_ings
    c_1 = x1_x2 * y3_y2
    denom = (x1_x0 * y3_y2) - (y1_y0 * x3_x2)
    #
    # winding number section
    # --
    a_num = a_0 - a_1  # t_num
    b_num = b_0 - b_1  # s_num
    # chk01 = np.logical_and(y2[:, None] <= y0, y0 <= y3[:, None])
    chk1 = (y0_y2 >= 0.0)
    chk2 = np.less(y0, y3[:, None])
    chk3 = np.sign(a_num).astype(np.int)  # t_num replaced with a_num
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=0, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=0, dtype=int)
    wn_vals = pos - neg
    inside = pnts[np.nonzero(wn_vals)]       # pnts in poly
    outside = pnts[np.nonzero(wn_vals + 1)]  # pnts outside poly
    x_type = np.vstack((wn_vals[:-1], wn_vals[1:])).T
    #
    # crossings
    # --
    a = a_0 <= a_1  # b = b_0 <= b_1
    c = c_0 <= c_1
    # w0 = np.logical_and(a, c) * 2     # both on right  (T, T)
    w1 = np.logical_and(a, ~c) * 1      # start on right (T, F)
    w2 = np.logical_and(~a, c) * -1     # start on left  (F, T)
    w3 = np.logical_and(~a, ~c) * -2    # both on left   (F, F)
    x_ings = (w1 + w2 + w3)  # whr = np.argwhere(abs(x_ings) == 1)
    #
    # intersections
    # --
    with np.errstate(all="ignore"):     # ignore all errors
        u_a = a_num/denom
        u_b = b_num/denom
        z0 = np.logical_and(u_a >= 0., u_a <= 1.)  # equal to `id_vals`
        z1 = np.logical_and(u_b >= 0., u_b <= 1.)
        both = z0 & z1
        xs = (u_a * x1_x0[None, :] + x0[None, :])[both]
        ys = (u_a * y1_y0[None, :] + y0[None, :])[both]
    x_pnts = []
    if xs.size > 0:
        x_pnts = np.concatenate((xs[:, None], ys[:, None]), axis=1)
    eq = pnts[eq_]
    whr = np.array(np.nonzero(both)).T  # or np.argwhere(abs(x_ings) == 1)
    return x_ings, x_type, whr, inside, outside, x_pnts, eq


def _cut_(splitter, geom):
    """Return the result of a polygon split.

    Parameters
    ----------
    splitter, geom : geometry
        `splitter` is a line which crosses two segments of geom.
        `geom` is a polygon feature.

    Notes
    -----
    .. Note::
        note to self... to clip polylines, just don't close the geometry

    Split into chunks.

    >>> spl = np.array([[ 0.0, 7.5], [10.0, 2.5]])
    >>> sq = np. array([[ 0.0, 0.0], [ 0.0, 10.0], [10.0, 10.0],
    ...                 [10.0, 0.0], [ 0.0,  0.0]])
    >>> result = _c_(spl, sq)
    >>> result
    ... array([[ 0.0, 0.0], [ 0.0, 7.5], [10.0, 2.5],
    ...        [10.0, 0.0], [ 0.0, 0.0]])

    Reassemble an from-to array back to an N-2 array of coordinates. Also,
    split, intersection points into pairs.

    >>> coords = np.concatenate((r0[0, :2][None, :], r0[:, -2:]), axis=0)
    >>> from_to = [o_i[i: i+2] for i in range(0, len(o_i), 2)]
    """
    ft_p = np.concatenate((geom[:-1], geom[1:]), axis=1)
    args = _inouton_(geom, splitter)
    x_ings, x_type, whr, inside, outside, x_pnts, eq = args  # whr = id_vals
    #
    if x_pnts is None:
        return None
    #
    uni, idx, cnts = np.unique(whr[:, 0], True, return_counts=True)
    crossings = uni[cnts >= 2]  # a line has to produce at least 2 x_sections
    #
    polys = []
    is_first = True
    ps = np.copy(ft_p)
    test = []
    for i, seg in enumerate(crossings):  # clipper ids that cross poly
        w = whr[:, 0] == seg
        out_in = whr[w]
        all_pairs = x_pnts[w]
        chunks = [out_in[i: i+2] for i in range(0, len(out_in), 2)]
        pair_chunk = [all_pairs[i: i+2] for i in range(0, len(out_in), 2)]
        sub = []
        for j, o_i in enumerate(chunks):
            pairs = pair_chunk[j]
            if is_first:
                p0, p1 = o_i[:, 1]
                is_first = False  # cancel first
            else:
                o_i = o_i[:, 1] - (ft_p.shape[0] - ps.shape[0])
                p0, p1 = o_i
            if (p1 - p0) >= 2:    # slice out extra rows, but update sp first
                sp = ps[p0: p1 + 1]
                ps = np.concatenate((ps[:p0 + 1], ps[p1:]))
            else:
                sp = ps[p0:p1 + 1]
                ps = np.copy(ps)
            ps[p0, -2:] = pairs[0]
            ps_new = pairs[:2].ravel()
            ps[p0 + 1, :2] = pairs[1]
            #
            pieces = [ps[:(p0 + 1)], ps_new, ps[(p0 + 1):]]
            z0 = [np.atleast_2d(i) for i in pieces]
            z0 = np.concatenate(z0, axis=0)
            z_0 = np.concatenate((z0[0, :2][None, :], z0[:, -2:]))
            #
            ps = np.copy(z0)  # copy if there is more than 1 chunk!!
            #
            if i == len(crossings - 1):
                close_ = z0[0]
                close_[:2] = pairs[0]
                close_ = close_[None, :]
                sp_new = np.concatenate((pairs[1], pairs[0]))
                z1 = np.concatenate((close_, sp, sp_new[None, :]), axis=0)
            else:
                sp[0, :2] = pairs[0]
                sp[-1, -2:] = pairs[1]
                sp_new = np.concatenate((pairs[1], pairs[0]))
                z1 = np.concatenate((sp, sp_new[None, :]), axis=0)
            z_1 = np.concatenate((z1[0, :2][None, :], z1[:, -2:]))
            sub.append([z_0, z_1])
            polys.extend([z_0, z_1])
        test.append(sub)
    return (polys, x_pnts, whr, inside, outside, test)


def _wn_clip_(pnts, poly, return_winding=True):
    """Return points in polygon using `winding number`.  See npg_pip.

    Notes
    -----
    >>> in_ = pnts[np.nonzero(wn_vals)]  # points inside the polygon
    >>> out2_ = pnts[np.nonzero(wn_vals + 1)]  # clever, clever
    """
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
    in_ = pnts[np.nonzero(wn_vals)]
    if return_winding:  # return pnts `in_`side the `poly`.
        return in_, wn_vals  # np.vstack((wn_vals[:-1], wn_vals[1:])).T
    return in_


def c_in(splitter, geom, keep_in=True):  # --- c_in with _cut_
    """Return split parts.  Calls ``_cut_`` for all splitters.

    Returns
    -------
    The split geometry is divided into two parts:  those inside and outside
    the clipping polygon.

    - ``_cut_`` does the actual work of splitting and sorting.
    - ``_wn_clip_`` does checking to ensure geometry are inside or outside.
    """
    vals = _cut_(splitter, geom)
    if vals is None:
        return None
    polys, x_pnts, whr, inside, outside, test = vals
    # x_pnts = np.unique(x_pnts, axis=0)
    if len(polys) == 0:
        return None
    in_ = []
    out_ = []
    N = len(test)
    for i, inout in enumerate(test):
        n = len(inout)
        if i < N:  # redundancy check to see if `outside` is really outside
            z = inout[0][0]
            wn_test = _wn_clip_(z, splitter, False)
            if len(wn_test) == len(z):
                in_.append(z)
        if n == 1:  # for splits that return 2 arrays, check `inside` for True
            # out_.append(inout[0][0])
            z = inout[0][1]
            wn_test = _wn_clip_(z, splitter, False)
            if len(wn_test) == len(z):
                in_.append(z)
            else:
                out_.append(z)
        else:
            for j in range(n):
                in_.append(inout[j][1])
                if j == n - 1:
                    out_.append(inout[j][0])
    return in_, out_


def poly_clip(clippers, polys, inside=True):
    """Return the inside or outside clipped geometry.

    Parameters
    ----------
    clippers, polys : list or array_like
        The geometries representing the clipping features and the polygon
        geometry.  A list or array with ndim > 2 is expected for both
    inside : boolean
        True, keeps the inside geometry.  False for outside, the clip extents.

    Returns
    -------
    The shared_extent between the clipping and poly features.  The geometry
    inside/outside depending on the ``inside`` boolean.

    Example
    -------
    >>> poly_clip([c0], [E])  # lists provided
    """
    shared_extent = []
    out2 = []
    for i, clp in enumerate(clippers):
        for j, poly in enumerate(polys):
            shared, extent = common_extent(clp, poly)
            if shared:
                shared_extent.append([i, j, extent])
                r = c_in(clp, poly, keep_in=inside)
                if r is not None:
                    out2.append(r)
    # remember that shared_extent and out2 should be sliced prior to plotting
    # ie. out2[0] is the result of clipping one polygon by one clipper
    return shared_extent, out2


def winding_num(pnts, poly, batch=False):
    """Point in polygon using winding numbers.

    Parameters
    ----------
    pnts : array
        This is simply an (x, y) point pair of the point in question.
    poly : array
        A clockwise oriented Nx2 array of points, with the first and last
        points being equal.

    Notes
    -----
    Until this can be implemented in a full array of points and full suite of
    polygons, you have to test for all the points in each polygon.

    >>> w = [winding_num(p, e1) for p in g_uni]
    >>> g_uni[np.nonzero(w)]
    array([[ 20.00,  1.00],
    ...    [ 21.00,  0.00]])

    References
    ----------
    `<http://geomalgorithms.com/a03-_inclusion.html#wn_PnPoly()>`_.
    """
    def _is_right_side(p, strt, end):
        """Determine if a point (p) is `inside` a line segment (strt-->end).

        See Also
        --------
        `line_crosses`, `in_out_crosses` in npg_helpers.
        position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
        negative for right of clockwise line, positive for left. So in essence,
        the reverse of _is_left_side with the outcomes reversed ;)
        """
        x, y, x0, y0, x1, y1 = *p, *strt, *end
        return (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)

    def cal_w(p, poly):
        """Do the calculation."""
        w = 0
        y = p[1]
        ys = poly[:, 1]
        for i in range(poly.shape[0]):
            if ys[i-1] <= y:
                if ys[i] > y:
                    if _is_right_side(p, poly[i-1], poly[i]) > 0:
                        w += 1
            elif ys[i] <= y:
                if _is_right_side(p, poly[i-1], poly[i]) < 0:
                    w -= 1
        return w
    if batch:
        w = [cal_w(p, poly) for p in pnts]
        return pnts[np.nonzero(w)], w
    else:
        return cal_w(pnts, poly)


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
