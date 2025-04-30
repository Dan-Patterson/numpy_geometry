# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
------------------------------------------------
npg_geom_ops: Buffering geometry focused methods
------------------------------------------------

**Buffering/offsetting methods that work with Geo arrays or np.ndarrays.**

----

Script :
    npg_buffer.py

Author :
    Dan_Patterson

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-04-27

Purpose
-------
Buffering/offsetting methods that work with Geo arrays or np.ndarrays.
In the case of the former, the methods may be being called from Geo methods
in such things as a list comprehension.

See Also
--------
`npg_geom_ops.py` contains the main documentation.

"""

# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121

import sys
import numpy as np

from npg import npGeo  # noqa

from npg.npg_bool_hlp import _del_seq_pnts_

from npg.npg_geom_hlp import (_area_centroid_, _in_extent_,  # noqa
                              _is_convex_, _bit_length_)  # noqa

from npg.npg_geom_ops import _is_pnt_on_line_  # noqa

from npg.npg_maths import (_angles_3pnt_, circ_circ_intersection,  # noqa
                           line_circ_intersection)  # noqa

from npg.npg_pip import np_wn, _side_  # noqa

from npg.npg_plots import plot_polygons, plot_segments, plot_polylines  # noqa

# from npg.npg_prn import prn_q, prn_tbl

# np.set_printoptions(
#     edgeitems=10, linewidth=100, precision=2, suppress=True, threshold=200,
#     formatter={"bool": lambda x: repr(x.astype(np.int32)),
#                "float_kind": '{: 6.2f}'.format})
# np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# -- See script header

__all__ = [
    'on_line_chk',                     # (1a) distance functions
    'area_buffer',                     # (4) buffer, scale
    'offset_buffer',
    'node_buffer',
    'rounded_buffer'
]

__helpers__ = [
    '_e_2d_',                          # (1) general helpers
    '_x_ings_',
    '_offset_np_'
]

__imports__ = [
    'npGeo',             # npGeo and sub modules
    'npg_geom_hlp',
    'npg_pip',
    'npg.npg_prn'
    'np_wn', '_side_',   # npg.npg_pip
    '_del_seq_pnts_',    # npg.npg_bool_hlp
    '_area_centroid_',   # npg_geom_hlp
    '_bit_area_',
    '_get_base_',
    '_bit_min_max_',
    '_in_extent_',
    '_is_pnt_on_line_',  # npg_geom_ops
    '_angles_3pnt_'      # npg_maths
    'prn_q',             # npg.npg_prn
    'prn_tbl'
]


# ---- ---------------------------
# ---- (1) general helpers
#
def _e_2d_(a, p):
    """Array points, `a`,  to point `p`, distance.

    Use to determine distance of a point to a segment or segments.
    """
    if hasattr(a, 'IFT'):
        a = a.tolist()
    diff = a - p[None, :]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def on_line_chk(start, end, xy, tolerance=1.0e-12):
    """Perform a distance check of whether a point is on a line.

    Parameters
    ----------
    start, end, xy : array_like
        The x,y values for the points.
    tolerance : number
        Acceptable distance tolerance to account for floating point issues.

    Returns
    -------
    A boolean indicating whether the x,y point is on the line and a list of
    values as follows::
        [xy] : if start or end equals xy
        [start, xy, d0] : `xy` is closest to `start` with a distance `d0`.
        [xy, end, d1] : `xy` is closest to `end` with a distance of `d1`.
        [] : the empty list is returned when `xy` is not on the line.

    See Also
    --------
    `npg_geom_ops._is_pnt_on_line_` use if just a boolean check is required.
    """
    #
    def dist(a, b):
        """Actual distance."""
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    #
    # boolean checks for start, end xy equality
    if (start == xy).all():
        return True, [xy]
    if (end == xy).all():
        return True, [xy]
    line_leng = dist(start, end)
    d0, d1 = dist(start, xy), dist(end, xy)
    d = (d0 + d1) - line_leng
    chk = -tolerance <= d <= tolerance
    if chk:  # -- xy is on line
        if d0 <= d1:  # -- closest to start
            return chk, [start, xy, d0]
        return chk, [xy, end, d1]  # -- closest to end
    return chk, []  # -- not on line


# def _wn_(pnts, poly, return_winding=True, extras=False):
#     """Return points in polygon using a winding number algorithm in numpy.

#     See Also
#     --------
#     A direct copy of `npg_pip.np_wn` renamed and used for testing here.

#     `_side_` from `npg_pip` is useful

#     r, in_, inside, outside, equal_ = _side_(pnts, poly)

#     Notes
#     -----
#     The following returns the indices of the points that are equal `w`.  This
#     can be used to extract those points (or None)::

#       w = np.nonzero((poly[:-1] == pnts[:, None]).all(-1).any(-1))[0]
#       eq_ = pnts[w] if w.size != 0 else None

#     Inclusion checks
#     ----------------
#     on the perimeter is deemed `out`
#         chk1 (y_y0 > 0.0)  changed from >=
#         chk2 np.less is ok
#         chk3 leave
#         pos  leave
#         neg  chk3 <= 0  to keep all points inside poly on edge included
#     """
#     x0, y0 = poly[:-1].T  # polygon `from` coordinates
#     x1, y1 = poly[1:].T   # polygon `to` coordinates
#     x, y = pnts.T         # point coordinates
#     y_y0 = y[:, None] - y0
#     y_y1 = y[:, None] - y1
#     x_x0 = x[:, None] - x0
#     # -- diff = np.sign(np.einsum("ikj, kj -> ij", pnts[:, None], poly[:-1]))
#     diff_ = ((x1 - x0) * y_y0 - (y1 - y0) * x_x0) + 0.0  # einsum originally
#     chk1 = (y_y0 > 0.0)  # -- top and bottom point inclusion!   try `>`
#     chk2 = (y_y1 < 0.0)  # was  chk2 = np.less(y[:, None], y1)  try `<`
#     chk3 = np.sign(diff_).astype(np.int32)
#     pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
#     neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)  # -- <= ??
#     wn = pos - neg
#     in_ = pnts[np.nonzero(wn)]
#     if extras:
#         eq_ids = np.isin(pnts, poly).all(-1).nonzero()[0]  # equal
#         extra_info = ["equal pnt ids", eq_ids]
#     if return_winding:
#         if extras:
#             return in_, wn, extra_info
#         return in_, wn
#     return in_


def _x_ings_(args):
    """Return a two line intersection point from 4 points.

    Parameters
    ----------
    args : array_like
        The input shape needs to be one of (8,), (2, 4) or (2, 2, 2).

    Notes
    -----
    `c` is the denominator, and `a` and `b` are ua and ub in Paul Bourkes

    Returns
    -------
    The intersection point if two segments cross, or an extrapolation to a
    point where they would meet.
    """
    #
    if len(args) == 8:
        x0, y0, x1, y1, x2, y2, x3, y3 = args
    elif np.prod(args.shape) == 8:
        args = np.ravel(args)
    else:
        print("\n{}".format(_x_ings_.__doc__))
        return None
    #
    x0, y0, x1, y1, x2, y2, x3, y3 = args
    dx_10, dy_10 = x1 - x0, y1 - y0
    dx_32, dy_32 = x3 - x2, y3 - y2
    #
    a = x0 * y1 - x1 * y0  # 2d cross product `cross2d`
    b = x2 * y3 - x3 * y2
    c = dy_10 * dx_32 - dy_32 * dx_10  # (y1-y0)*(x3-x2) - (y3-y2)*(x1-x0)
    if abs(c) > 1e-12:  # -- return the intersection point
        n1 = (a * dx_32 - b * dx_10) / c
        n2 = (a * dy_32 - b * dy_10) / c
        return (n1, n2)
    # -- if no intersection exists, return the end of the first line
    return None


def _offset_np_(poly, buff_dist=1.0, offset_check=False, as_segments=False):  #
    """Buffer by offsetting lines.

    Parameters
    ----------
    a : array
        The array of polygon points to be buffered.
    buff_dist : number
        Positive number buffers outward, negative for inwards.
    as_segments : boolean
        True returns a polygon.  False returns its segmented representation.

    Notes
    -----
    This is 4x faster than `_offset_` which can be used for lists or arrays
    which delineate a polygon boundary.

    `offset_check` impact in `_offset_np_` vs `_offset_`
      - True  540 μs vs 660 μs
      - False 50 μs vs 220 μs
    For history::

      dxdy = a[1:] - a[:-1]  # `a` is a polygon array of points
      dx, dy = dxdy.T
      hypot_ = np.hypot(dx, dy)
      r = buff_dist / hypot_
      # versus
      r = buff_dist / np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))
    """
    # -- calculate the offsets
    if hasattr(poly, "IFT"):
        poly = poly.XY
    #
    # -- no longer checking for convexity.
    #
    dxdy = poly[1:] - poly[:-1]
    r = buff_dist / np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))
    rr = np.concatenate((r[:, None], -r[:, None]), axis=1)
    dx_dy = dxdy * rr
    dy_dx = dx_dy[:, [1, 0]]  # -- swap order yielding dy, dx
    pnt0 = poly[:-1] + dy_dx  # new start and ...
    pnt1 = poly[1:] + dy_dx   # end points for offset segments
    #
    # -- offset segments
    _offset_orig_ = np.concatenate((pnt0, pnt1), axis=1)
    #
    _offset_ = np.copy(_offset_orig_)
    tmp = np.concatenate((_offset_[:-1], _offset_[1:]), axis=1)
    to_add = np.concatenate(
        (_offset_[-1], _offset_[0]))[None, :]  # atleast_2d
    # wrap last
    args = np.concatenate((tmp, to_add), axis=0)  # 8xN array of x,y values
    #
    # -- get the crossings for the array `args`
    #
    x0, y0, x1, y1, x2, y2, x3, y3 = args.T
    x1_x0, y1_y0 = x1 - x0, y1 - y0
    x3_x2, y3_y2 = x3 - x2, y3 - y2
    #
    a = x0 * y1 - x1 * y0  # 2d cross product `cross2d` for arrays
    b = x2 * y3 - x3 * y2
    c = y1_y0 * x3_x2 - y3_y2 * x1_x0  # (y1-y0)*(x3-x2) - (y3-y2)*(x1-x0)
    #
    # -- see `p_int_p` in npg_bool_hlp for notes regarding intersections
    #    on both segments
    with np.errstate(divide='ignore', invalid='ignore'):
        n1 = (a * x3_x2 - b * x1_x0) / c
        n2 = (a * y3_y2 - b * y1_y0) / c
        # clean out bad values of `c`
        w = np.nonzero(abs(c) > 1e-12)[0]
        n1 = n1[w]
        n2 = n2[w]
        tmp = np.concatenate((n1[:, None], n2[:, None]), axis=1)
        tmp = np.concatenate((tmp[-1][None, :], tmp), axis=0)
        tmp = np.round(tmp, 6)  # try to address floating pnt issues
    #
    new_pnts = _del_seq_pnts_(tmp, True)
    #
    new_segs = np.concatenate((new_pnts[:-1], new_pnts[1:]), axis=1)
    #
    if as_segments:  # return both the polygon and the original segments
        #
        # o_segs = np.concatenate((poly[:-1], poly[1:]), axis=1)
        # plot_segments(np.concatenate((o_segs, _offset_orig_), axis=0))
        # plot_segments(np.concatenate((o_segs, new_segs), axis=0))
        #
        if _offset_ is None:
            _offset_ = _offset_orig_
        return new_segs, _offset_orig_, new_pnts
    #
    return new_pnts


# ---- ---------------------------
# ---- (2)  buffer, scale functions
#
def area_buffer(poly, factor=1, asGeo=False):
    """Scale a convex polygon geometry by its area.

    Warning
    -------
    This is really only applicable to convex shapes.

    Parameters
    ----------
    a : ndarray
        A polygon represented by an ndarray.
    factor : number
        Positive scaling as an integer or decimal number.

    Requires
    --------
    `is_Geo` from npGeo and `_area_centroid_` from npg_geom_hlp.

    Notes
    -----
    - Translate to the origin of the unique points in the polygon.
    - Determine the initial area.
    - Scale the coordinates.
    - Shift back to the original center.
    """
    def _area_scaler_(a, factor):
        """Do the work."""
        if factor <= 0.0:
            return None
        a = np.array(a)
        area_, cent = _area_centroid_(a)  # from npg_geom_hlp 2025-04-09
        shifted = a - cent
        alpha = np.sqrt(factor * area_ / area_)
        scaled = shifted * [alpha, alpha]
        return scaled + cent
    # --
    if npGeo.is_Geo(poly):
        if poly.K != 2:
            print("\n Polygon geo array required.")
            return None
        final = [_area_scaler_(a, factor) for a in poly.bits]
    else:
        final = _area_scaler_(poly, factor)
    if asGeo:
        a_stack, ift, extent = npGeo.array_IFT(final, shift_to_origin=False)
        return npGeo.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
    return final


def offset_buffer(poly, buff_dist=1, keep_holes=False, asGeo=False):
    """Buffer singlepart polygons with non-rounded ends.

    Parameters
    ----------
    poly : array_like
        The polygon feature to buffer in the form of an ndarray or Geo array.
    buff_dist : number
        The offset/buffer distance.  Positive for expansion, negative for
        contraction.

    Returns
    -------
    A buffer without rounded corners.  There are limits, so use a commercial
    package if you want perfection.

    Requires
    --------
    `_offset_np_` to create the offset geometry.

    Notes
    -----
    Use `plot_buffs` to view the results.

    If you want rounded corners, use something else.
    Singlepart shapes supported with or without holes.
    """

    def _Geo_buff_(poly, buff_dist, keep_holes):
        """Move the Geo array buffering separately."""
        arr = poly.bits
        cw = poly.CL
        final = []
        for i, a in enumerate(arr):
            if cw[i] == 0 and keep_holes:
                buff_dist = -buff_dist
                a = a[::-1]
                ext = [np.min(a, axis=0), np.max(a, axis=0)]
                b = _offset_np_(a, buff_dist,
                                offset_check=True, as_segments=False)
                in_check = _in_extent_(b, ext)
                if in_check:   # print(buff_dist, a, b, in_check)
                    final.append(b)
            elif cw[i] == 1:   # print(buff_dist, a, b)
                b = _offset_np_(a, buff_dist,
                                offset_check=True, as_segments=False)
                final.append(b)
        return final
    # --
    #
    # Buffer Geo arrays or ndarray
    if npGeo.is_Geo(poly):
        final = _Geo_buff_(poly, buff_dist, keep_holes)
    else:
        final = _offset_np_(poly, buff_dist)
    if asGeo:
        a_stack, ift, extent = npGeo.array_IFT(final, shift_to_origin=False)
        return npGeo.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
    return final


def node_buffer(nodes, radius=1.0, theta=1.0):
    """Return a circle to be placed at nodes around a polygon perimeter.

    Parameters
    ----------
    nodes : array_like
        The segment points forming a polygon or polyline feature
    radius : number
        circle radius
    theta : angle increment forsectorspacing
    xc, yc : number
        circle center planar coordinates

    Returns
    -------
    Clockwise oriented circles for each node point in the nodes list.

    angles = np.deg2rad(np.arange(-180.0, 180.0 + theta, step=theta))

    See Also
    --------
    `circle` and `circle_mini` in `npg.npg_create`.

    """
    angles = np.deg2rad(np.arange(180.0, -180.0 - theta, step=-theta))
    x_s = radius * np.cos(angles)  # X values
    y_s = radius * np.sin(angles)  # Y values
    pnts = np.array([x_s, y_s]).T  # assumes an origin of 0,0
    #
    # -- now produce the output points for all the nodes
    all_circs = [node + pnts for node in nodes]
    return all_circs


def rounded_buffer(poly, buff_dist=1.0, radius=1, step=5):
    """Perform polygon buffering.

    Parameters
    ----------
    poly : array_like
        This could be an individual array representing a polygon or a bit from
        a Geo array.
    buff_dist : float
        The distance to buffer.  It can be positive or negative.

    Requires
    --------
    `_x_ings_` to derive segment intersection points
    `_arc_` for rounded corners

    Notes
    -----
    To use a slice for testing rounded gone wrong, use
    ml is the maple leaf

    to_add = np.array([[11., 4.], [5., 4], [0., 8.]])
    sub = np.concatenate((ml[6:16], to_add), axis=0)
    poly = np.copy(sub)

    """
    def _a_(circ_pnts, p_st, cent, p_en, radius, step, outside=True):
        """Create arc from a mini circle.  A circle center of 0,0 is used."""
        #
        d0 = p_st - cent
        d1 = p_en - cent
        start = np.atan2(*d0[::-1])  # np.degrees(
        stop = np.atan2(*d1[::-1])   # np.degrees(
        start_deg = np.degrees(start)
        stop_deg = np.degrees(stop)
        # signs = [i >= 0 for i in [start_deg, stop_deg]]
        # inner_angle = np.degrees(_angle_between_(p_st, cent, p_en))
        # outer_angle = np.degrees(_angle_between_(p_en, cent, p_st))
        #
        if start_deg < stop_deg:
            if start_deg < 0 and stop_deg >= 0:
                w0 = np.nonzero(ang <= start_deg)[0]
                w1 = np.nonzero(ang >= stop_deg)[0]
                ids = np.concatenate((w0, w1))
            else:
                ids = np.logical_and(ang >= start_deg, ang <= stop_deg)
        elif start_deg > stop_deg:
            if start_deg >= 0:
                ids = np.logical_and(ang <= start_deg, ang >= stop_deg)
            else:  # both negative ?
                ids = np.logical_and(ang <= start_deg, ang >= stop_deg)
        #
        new_pnts = circ_pnts[ids] + cent
        return new_pnts
    #
    # -- rounded corners
    circ_pnts = None
    xc = 0.0
    yc = 0.0
    angles = np.deg2rad(np.arange(180.0, -180.0 - step, -step))
    ang = np.degrees(angles)
    x_s = buff_dist * np.cos(angles) + xc  # X and Y values
    y_s = buff_dist * np.sin(angles) + yc  # add the circle center later
    circ_pnts = np.array([x_s, y_s]).T
    # --
    # -- Derive the offset segments using `_offset_np_`, prior to intersection.
    #    segs0, segs1 => extended segments, original offset segments
    #
    # --
    r = _offset_np_(poly,
                    buff_dist=buff_dist,
                    offset_check=True,  # do the `_cross_chk` on the first run
                    as_segments=True)
    # --
    new_segs, _offset_, keep_ids, dump_segs, wn_vals, new_pnts = r
    #
    # o_segs = np.concatenate((poly[:-1], poly[1:]), axis=1)
    # plot_segments(np.concatenate((o_segs, new_segs), axis=0))
    #
    # (2) rounded corners are needed, so proceed -----------------------------
    #
    orig_ids = np.arange(0, poly.shape[0] - 1)
    angles_orig = _angles_3pnt_(poly, inside=False, in_deg=True)
    cents_orig = poly[orig_ids]
    #
    if orig_ids.shape[0] != keep_ids.shape[0]:
        cents_ = cents_orig[keep_ids]
        angles_ = _angles_3pnt_(poly[keep_ids], inside=False, in_deg=True)
        circ_ids = [keep_ids[i] for i in range(0, len(angles_))
                    if angles_[i] > 0]
    else:
        cents_ = np.copy(cents_orig)
        angles_ = np.copy(angles_orig)
    #
    # (3) All convex, shortcut -----------------------------------------------
    #
    is_convex = (angles_ > 180).all()  # all are convex this will return True
    if is_convex:
        segs = []
        for cnt, ang_ in enumerate(angles_):
            p_st = _offset_[cnt - 1][-2:]
            cent = cents_[cnt]
            p_en = _offset_[cnt][:2]
            arc = _a_(circ_pnts, p_st, cent, p_en,
                      radius=buff_dist, step=step, outside=True)
            segs.append(arc)
            segs.append(_offset_[cnt].reshape(2, 2))
        final = np.concatenate((segs), axis=0)
        return final  # plot_polygons([final, poly])
    #
    # (4) Not all convex so really working it --------------------------------
    #
    segs = []
    # max_id = angles_.shape[0] + 1  # limit slicing and force roll to 0
    for cnt, ang_ in enumerate(angles_):  # out):
        ang_ = angles_[cnt]  # keep in for testing
        #
        prev_ = _offset_[cnt - 1]
        p_st = prev_[-2:]
        cent = cents_[cnt]
        p_en = _offset_[cnt][:2]  # p_st, cent, p_en
        if ang_ <= 0:
            if angles_[cnt - 1] > 0:
                segs.append(prev_.reshape(2, 2))
            else:
                segs.append(np.array([p_st, p_en]))
        else:
            # arccc = np.array([p_st, p_en])  # attempt at just joining chord
            arc = _a_(circ_pnts, p_st, cent, p_en,
                      radius=buff_dist, step=step, outside=True)
            if angles_[cnt - 1] > 0:
                segs.append(prev_.reshape(2, 2))  # previous offset
            # segs.append(arccc)  # attempt at just joining chord
            segs.append(arc)
    final = np.concatenate((segs), axis=0)
    return final  # plot_polygons([final, poly])


"""
for n, i in enumerate(cs):
    c0 = cs[n-1]
    c1 = i
    r = 4
    v = circ_circ_intersection(c0, 4, c1, 4, return_arcs=False, step=1)
    v = v.ravel() if isinstance(v, np.ndarray) else []
    cen = np.array([c0, c1]).ravel()
    out.append([[nodes[n-1], nodes[n]], cen, v])

"""
# plot_polylines(segs
# works up to here by producing arcs
#
# frst = segs[0][0][None, :]
# segs.append(frst)


def plot_buffs(a, label_pnts=False, as_segments=False):
    """Plot the buffers.

    Parameters
    ----------
    a : array_like
        An ndarray or a list of arrays.
    label_pnts : boolean
        True to label the nodes or segments.
    as_segments : boolean
        True returns a segmented version of the contents of `a`.
    """
    if as_segments:
        if isinstance(a, (list, tuple)):
            s0 = [i for i in a if isinstance(i, np.ndarray)]
            s1 = [np.concatenate((i[:-1], i[1:]), axis=1)
                  for i in s0
                  if i.shape[1] == 2]
            s2 = np.concatenate(s1, axis=0)
            plot_segments(s2)
            return
        if isinstance(a, np.ndarray):
            if a.shape[1] == 2:
                orig_segs = np.concatenate((a[:-1], a[1:]), axis=1)
                plot_segments(orig_segs)
            else:
                plot_segments(a)
            return
    if len(a) == 1:
        if label_pnts:
            lbls = np.arange(0, a.shape[0])
        else:
            lbls = None
        plot_polygons(a, outline=True, vertices=True, labels=lbls)


# ---- keep for now
#


# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # optional controls here
