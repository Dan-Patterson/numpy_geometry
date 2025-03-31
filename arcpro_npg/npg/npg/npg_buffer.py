# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
----------------------------------
npg_geom_ops: Bufferin geometry focused methods
----------------------------------

**Buffering/offsetting methods that work with Geo arrays or np.ndarrays.**

----

Script :
    npg_buffer.py

Author :
    Dan_Patterson

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-03-10

Purpose
-------
Buffering/offsetting methods that work with Geo arrays or np.ndarrays.
In the case of the former, the methods may be being called from Geo methods
in such things as a list comprehension.

See Also
--------
`npg_geom_ops.py` contains the main documentation

"""

# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121

import sys
import numpy as np

from npg import npGeo  # noqa

from npg.npg_geom_hlp import _bit_area_, _in_extent_, _is_convex_  # noqa

from npg.npg_geom_ops import _is_pnt_on_line_  # noqa

from npg.npg_maths import (_angles_3pnt_, circ_circ_intersection,
                           line_circ_intersection)

from npg.npg_pip import np_wn, _side_

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
    'geo_buffer'
]

__helpers__ = [
    '_e_2d_',                          # (1) general helpers
    '_x_ings_',
    '_arc_',
    '_offset_simp',
    '_offset_np_'
]

__imports__ = [
    'npGeo',            # npGeo and sub modules
    'npg_geom_hlp',
    'npg_pip',
    'npg.npg_prn'
    'np_wn', '_side_',  # npg.npg_pip
    '_bit_area_',       # npg_geom_hlp
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


def _wn_(pnts, poly, return_winding=True, extras=False):
    """Return points in polygon using a winding number algorithm in numpy.

    See Also
    --------
    This is a direct copy of `npg_pip.np_wn` renamed and used for testing here.

    `_side_` from `npg_pip` is useful

    r, in_, inside, outside, equal_ = _side_(pnts, poly)

    Notes
    -----
    The following returns the indices of the points that are equal `w`.  This
    can be used to extract those points (or None)::

      w = np.nonzero((poly[:-1] == pnts[:, None]).all(-1).any(-1))[0]
      eq_ = pnts[w] if w.size != 0 else None

    Inclusion checks
    ----------------
    on the perimeter is deemed `out`
        chk1 (y_y0 > 0.0)  changed from >=
        chk2 np.less is ok
        chk3 leave
        pos  leave
        neg  chk3 <= 0  to keep all points inside poly on edge included
    """
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    x, y = pnts.T         # point coordinates
    y_y0 = y[:, None] - y0
    y_y1 = y[:, None] - y1
    x_x0 = x[:, None] - x0
    # -- diff = np.sign(np.einsum("ikj, kj -> ij", pnts[:, None], poly[:-1]))
    diff_ = ((x1 - x0) * y_y0 - (y1 - y0) * x_x0) + 0.0  # einsum originally
    chk1 = (y_y0 > 0.0)  # -- top and bottom point inclusion!   try `>`
    chk2 = (y_y1 < 0.0)  # was  chk2 = np.less(y[:, None], y1)  try `<`
    chk3 = np.sign(diff_).astype(np.int32)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)  # -- <= ??
    wn = pos - neg
    in_ = pnts[np.nonzero(wn)]
    if extras:
        eq_ids = np.isin(pnts, poly).all(-1).nonzero()[0]  # equal
        extra_info = ["equal pnt ids", eq_ids]
    if return_winding:
        if extras:
            return in_, wn, extra_info
        return in_, wn
    return in_


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


# def _a_(p_st, cent, p_en, radius=1., deg_step=5.0, outside=True):
#     """Create arc from a nimi circle.  A circle center of 0,0 is used."""
#     xc = 0.0
#     yc = 0.0
#     angles = np.deg2rad(np.arange(180.0, -180.0 - deg_step, step=-deg_step))
#     ang = np.degrees(angles)
#     x_s = radius * np.cos(angles) + xc  # X and Y values
#     y_s = radius * np.sin(angles) + yc  # add the circle center later
#     pnts = np.array([x_s, y_s]).T
#     #
#     d0 = p_st - cent
#     d1 = p_en - cent
#     start = np.atan2(*d0[::-1])  # np.degrees(
#     stop = np.atan2(*d1[::-1])   # np.degrees(
#     start_deg = np.degrees(start)
#     stop_deg = np.degrees(stop)
#     # signs = [i >= 0 for i in [start_deg, stop_deg]]
#     # inner_angle = np.degrees(_angle_between_(p_st, cent, p_en))
#     # outer_angle = np.degrees(_angle_between_(p_en, cent, p_st))
#     #
#     if start_deg < stop_deg:
#         if start_deg < 0 and stop_deg >= 0:
#             w0 = np.nonzero(ang <= start_deg)[0]
#             w1 = np.nonzero(ang >= stop_deg)[0]
#             ids = np.concatenate((w0, w1))
#         else:
#             ids = np.logical_and(ang >= start_deg, ang <= stop_deg)
#     elif start_deg > stop_deg:
#         if start_deg >= 0:
#             ids = np.logical_and(ang <= start_deg, ang >= stop_deg)
#     #
#     new_pnts = pnts[ids] + cent
#     return new_pnts


def _arc_(p_st, cent, p_en, radius, deg_step=5., outside=True):
    """Create an arc given start, end and center.  Radius is known.

    For testing ::

        p_st, cent, p_en = outers[X][1:4]   replace X with id
        radius, deg_step = 1.0, 22.5

    np.pi = 3.141592653589793
    np.pi/2. = 1.5707963267948966

    in 2d, cross(a, b) is = ax*by-ay*bx
    atan2(cross(a,b)), dot(a,b))
    #
    def angle3pt(a, b, c, ang_360=False):
        '''Counterclockwise angle in degrees by turning from a to c around b
            Returns a float between 0.0 and 360.0'''
        ang = np.degrees(
            np.atan2(c[1]-b[1], c[0]-b[0]) - np.atan2(a[1]-b[1], a[0]-b[0]))
        if ang_360:
            return ang + 360 if ang < 0 else ang
        return ang
    """
    def get_quad(c, s_e):
        """Return quadrants of arc segments."""
        x_, y_ = np.sign(np.diff([c, s_e], axis=0))[0]
        # right
        if x_ >= 0:
            if y_ >= 0:  # upper
                return 1
            return 4
        # left
        if y_ >= 0:  # upper
            return 2
        return 3
    #
    # np.mod((450.0 - angles), 360.)
    # trying with quadrants
    st_quad = get_quad(cent, p_st)
    en_quad = get_quad(cent, p_en)
    #
    # step_ = np.deg2rad(deg_step)
    d0 = p_st - cent
    d1 = p_en - cent
    start = np.atan2(*d0[::-1])  # np.degrees(
    stop = np.atan2(*d1[::-1])   # np.degrees(
    start_deg = np.degrees(start)
    stop_deg = np.degrees(stop)
    # inner_angle = np.degrees(_angle_between_(p_st, cent, p_en))
    # outer_angle = np.degrees(_angle_between_(p_en, cent, p_st))
    angles = None
    #
    # 1
    if st_quad == 1 and en_quad == 1:
        if stop_deg == 0.:
            f0 = np.arange(start_deg, stop_deg, -deg_step)
            angles = np.deg2rad(f0)
    elif st_quad == 1 and en_quad in [3, 4]:
        f0 = np.arange(start_deg, 0, -deg_step)
        f1 = np.arange(0, stop_deg, -deg_step)
        angles = np.deg2rad(np.concatenate((f0, f1)))
    # 2
    elif st_quad == 2 and en_quad in [1, 4]:  # check
        f0 = np.arange(start_deg, stop_deg, -deg_step)
        angles = np.deg2rad(f0)
    # 3
    elif st_quad == 3 and en_quad in [1, 2]:
        f0 = np.arange(-180., start_deg, deg_step)[::-1]
        f1 = np.arange(180., stop_deg, -deg_step)
        angles = np.deg2rad(np.concatenate((f0, f1)))
    elif st_quad == 3 and en_quad == 3:
        f0 = np.arange(start_deg, stop_deg, -deg_step)
        angles = np.deg2rad(f0)
    # 4
    elif st_quad == 4 and en_quad in [1, 2]:  # and outer_angle > 180
        if stop_deg == 0.:
            f0 = np.arange(start_deg, stop_deg, deg_step)
            angles = np.deg2rad(f0)
        else:
            f0 = np.arange(-180., start_deg, deg_step)
            f1 = np.arange(0, stop_deg, deg_step)  # could be empty
            angles = np.deg2rad(np.concatenate((f0, f1)))
    #
    if angles is None:
        print("cent {}  :  st, en quad {}, {}".format(cent, st_quad, en_quad))
        return []
    else:
        x_s = radius * np.cos(angles)         # X values
        y_s = radius * np.sin(angles)         # Y values
        pnts = cent + np.array([x_s, y_s]).T
    # -- with 360 degree check
    # # =================================================
    # if stop == np.pi:
    #     stop = -stop
    # # check for +/- pi
    # if start < 0 and stop > 0:
    #     f0 = np.arange(0, start, -step_)
    #     f1 = np.arange(0, stop, step_)
    #     angles = np.concatenate(([start], f0[::-1], f1[1:], [stop]))
    # elif start > 0 and stop < 0:
    #     f0 = np.arange(0, start, step_)
    #     f1 = np.arange(0, stop, -step_)
    #     angles = np.concatenate(([stop], f1[::-1], f0[1:], [start]))
    # elif start > stop:  # start or stop
    #     angles = np.arange(start, stop, -step_)
    # elif start < 0 and stop == 0:
    #     angles = np.arange(start, stop, step_)
    # else:
    #     angles = np.arange(start, stop, step_)  # default, print the case
    #     print("start {} stop {}".format(start, stop))
    # # cent_ = np.array([[xc, yc]])
    # x_s = radius * np.cos(angles)         # X values
    # y_s = radius * np.sin(angles)         # Y values
    # if outside:
    #     pnts = cent + np.array([x_s, y_s]).T
    # else:
    #     pnts = cent + np.array([x_s, y_s]).T  # -- could be + or -
    return pnts


def _offset_simp(poly, buff_dist=1.0, offset_check=True):
    """Return offset line buffering.

    Parameters
    ----------
    bit : array_like
        Normally a `bit` is a polygon part from a Geo array.  This can also be
        an Nx2 ndarray.
    buff_dist : number
        The offset(buffer) distance, positive or negative.
    offset_check : boolean
        True, performs a check to determine whether an offset segment places
        both their points inside the original polygon.  If so, corrections are
        made prior to determining intersections to form the final geometry.
    """
    # -- calculate the offsets
    ft_ = []
    segs = []
    poly = np.array(poly)
    for i in range(poly.shape[0] - 1):
        x1, y1, x2, y2 = poly[i: i + 2].ravel()  #
        hypot_ = np.hypot(x2 - x1, y2 - y1)
        if hypot_ != 0.0:
            r = buff_dist / hypot_
            vx, vy = (x2 - x1) * r, (y2 - y1) * r
            pnt0 = (x1 - vy, y1 + vx)
            pnt1 = (x2 - vy, y2 + vx)
            ft_.append([pnt0, pnt1])
    f_t = np.array(ft_)
    #
    # -- optional add section
    # orig_ = np.reshape(f_t, shape=(-1, 4))
    # to_add = np.concatenate((orig_[-1][-2:], orig_[0][:2]))[None, :]
    # orig_offset = np.concatenate((orig_, to_add), axis=0)
    #
    # -- do the check here
    if _is_convex_(poly):
        offset_check = False
        print("\nshape is convex, `offset_check` probably not needed.\n")
    #
    # -- assemble segment pairs to derive intersection points using `x_ings_`.
    #    This is done with or without offset checking. `np_wn`, is used for the
    #    segment-in-polygon determination.
    if offset_check:
        keep = [i for i in f_t if np_wn(i, poly).size <= 2]
        z = list(zip(keep[:-1], keep[1:]))
        z.append([z[-1][-1], z[0][0]])
        z = np.array(z)
    else:
        z = list(zip(f_t[:-1], f_t[1:]))
        z.append([z[-1][-1], z[0][0]])
        z = np.array(z)
    #
    # -- Get the offset segment intersection points and return the result.
    for pair in z:
        x_tion = _x_ings_(pair)
        segs.append(x_tion)  # np.array([i[0], middle]))
    frst = np.atleast_2d(segs[-1])
    final = np.concatenate((frst, np.array(segs)), axis=0)
    return final


def _offset_np_(poly,
                buff_dist=1.0,
                offset_check=True,
                as_segments=False):  #
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
    dxdy = poly[1:] - poly[:-1]
    r = buff_dist / np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))
    rr = np.concatenate((r[:, None], -r[:, None]), axis=1)
    dx_dy = dxdy * rr
    dy_dx = dx_dy[:, [1, 0]]  # -- swap order yielding dy, dx
    pnt0 = poly[:-1] + dy_dx     # new start and end points for offset segments
    pnt1 = poly[1:] + dy_dx      #
    _offset_ = np.concatenate((pnt0, pnt1), axis=1)
    # -- this is where you have to put the point in polygons check, prior
    #    to intersections... it is slow, since it is the pair not singletons
    #
    if offset_check:  # 51.9 μs  much faster than original
        f_t = np.reshape(_offset_, shape=(-1, 2))
        in_, wn_vals = np_wn(f_t, poly, return_winding=True)  # segment in chk
        #
        # -- the ids in `keep_ids` should be incremented by 1
        keep_ids = np.nonzero((wn_vals.reshape(-1, 2) == 0).any(-1))[0]
        #
        keep = np.array(_offset_[keep_ids])
        z = np.concatenate((keep[:-1], keep[1:]), axis=1)
        to_add = np.array([z[-1][-4:], z[0][:4]]).reshape(-1, 8)  # cleanup
        args = np.concatenate((z, to_add), axis=0)
    else:
        tmp = np.concatenate((_offset_[:-1], _offset_[1:]), axis=1)
        to_add = np.concatenate(
            (_offset_[-1], _offset_[0]))[None, :]  # atleast_2d
        args = np.concatenate((tmp, to_add), axis=0)  # 8xN array of x,y values
    #
    # -- get the crossings for the array `args`
    #    see `_x_ings_` for simple two line crossings
    x0, y0, x1, y1, x2, y2, x3, y3 = args.T
    dx_10, dy_10 = x1 - x0, y1 - y0
    dx_32, dy_32 = x3 - x2, y3 - y2
    #
    a = x0 * y1 - x1 * y0  # 2d cross product `cross2d` for arrays
    b = x2 * y3 - x3 * y2
    c = dy_10 * dx_32 - dy_32 * dx_10  # (y1-y0)*(x3-x2) - (y3-y2)*(x1-x0)
    #
    # -- see `p_int_p` in npg_bool_hlp for notes regarding intersections
    #    on both segments
    with np.errstate(divide='ignore', invalid='ignore'):
        n1 = (a * dx_32 - b * dx_10) / c
        n2 = (a * dy_32 - b * dy_10) / c
        # clean out bad values of `c`
        w = np.nonzero(abs(c) > 1e-12)[0]
        n1 = n1[w]
        n2 = n2[w]
        new_pnts = np.concatenate((n1[:, None], n2[:, None]), axis=1)
        new_pnts = np.concatenate((new_pnts[-1][None, :], new_pnts), axis=0)
    #
    # -- concatenate the intersection points with the segments
    #
    # -- for plotting
    #
    #  o_segs = np.concatenate((poly[:-1], poly[1:]), axis=1)
    #  plot_segments([o_segs, new_segs])  # original and offset connected
    #  plot_segments([o_segs, _offset_])  # offset simple, not extended
    #
    if as_segments:  # return both the polygon and the original segments
        _extended_ = np.concatenate((new_pnts[:-1], new_pnts[1:]), axis=1)
        #
        # o_segs = np.concatenate((poly[:-1], poly[1:]), axis=1)
        # plot_segments(np.concatenate((o_segs, _extended_), axis=0))
        #
        return _extended_, _offset_, keep_ids, wn_vals, new_pnts
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
    `is_Geo` from npGeo and `_bit_area_` from npg_geom_hlp.

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
        cent = np.mean(np.unique(a, axis=0), axis=0)
        shifted = a - cent
        area_ = _bit_area_(shifted)
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
    poly : ndarray
        The poly feature to buffer in the form of an ndarray.
    buff_dist : number
        The offset/buffer distance.  Positive for expansion, negative for
        contraction.

    Returns
    -------
    A buffer without rounded corners.

    Requires
    --------
    `_array_buff_` to create the offset geometry.

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
                # b = _offset_np_(a, buff_dist)  # 2025-03-10
                b = _offset_simp(a, buff_dist)  # 2025-03-13
                in_check = _in_extent_(b, ext)
                if in_check:   # print(buff_dist, a, b, in_check)
                    final.append(b)
            elif cw[i] == 1:   # print(buff_dist, a, b)
                # b = _offset_np_(a, buff_dist)
                b = _offset_simp(a, buff_dist)  # 2025-03-13
                final.append(b)
        return final
    # --
    #
    # Buffer Geo arrays or ndarray
    if npGeo.is_Geo(poly):
        final = _Geo_buff_(poly, buff_dist, keep_holes)
    else:
        # final = _offset_np_(poly, buff_dist)
        final = _offset_simp(poly, buff_dist)  # 2025-03-13
    if asGeo:
        a_stack, ift, extent = npGeo.array_IFT(final, shift_to_origin=False)
        return npGeo.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
    return final  # fr_to, z, final


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


def geo_buffer(poly, buff_dist=1.0, round_corners=True, radius=1, step=5):
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

    # --
    # -- Derive the offset segments using `_offset_np_`, prior to intersection.
    #    segs0, segs1 => extended segments, original offset segments
    #
    #  o_segs = np.concatenate((poly[:-1], poly[1:]), axis=1)
    #  plot_segments([o_segs, _extended_])  # original and offset connected
    #  plot_segments([o_segs, _offset_])    # original, not extended
    #
    circ_pnts = None
    if round_corners:
        xc = 0.0
        yc = 0.0
        angles = np.deg2rad(np.arange(180.0, -180.0 - step, -step))
        ang = np.degrees(angles)
        x_s = buff_dist * np.cos(angles) + xc  # X and Y values
        y_s = buff_dist * np.sin(angles) + yc  # add the circle center later
        circ_pnts = np.array([x_s, y_s]).T
    #
    r = _offset_np_(poly,
                    buff_dist=buff_dist,
                    offset_check=True,  # use its boolean
                    as_segments=True)
    # -- new_segs, off_01, keep_ids, wn_vals, new_pnts
    _extended_, _offset_, keep_ids, wn_vals, new_pnts = r
    new_poly = np.concatenate(
        (_extended_[:, :2], _extended_[0, :2][None, :]), axis=0)
    #
    if not round_corners:
        # -- test plotting to this stage  plot_polygons([poly, new_poly])
        return new_poly
    # ------------------------------------------------------------------------
    # -- rounded corners are needed, so proceed
    #   `_offset_` has the first and last segment duplicated so you can wrap
    #     around at the end
    #
    # -- compare original and new angles, if they are the same, then it is
    #    easier
    angles_orig = _angles_3pnt_(poly, inside=False, in_deg=True)
    angles_out = _angles_3pnt_(new_poly, inside=False, in_deg=True)
    a_diff = (np.abs((angles_orig[:, None] - angles_out)) <= 1.e-04)  # tol
    orig_ids, out_ids = np.nonzero(a_diff)
    #
    chk = False
    if angles_orig.shape[0] == angles_out.shape[0]:  # shapes are the same
        chk = np.allclose(angles_orig, angles_out)
    if chk:
        k_cents = poly[orig_ids]
    else:
        k_cents = poly[out_ids]
    #
    # -- offset polygon from new_segs
    #
    segs = []
    max_id = angles_out.shape[0] - 1  # limit slicing and force roll to 0
    for cnt, ang_ in enumerate(angles_out):  # out):
        ang_ = angles_out[cnt]  # keep in for testing
        #
        if cnt + 1 > max_id:
            next_curve = np.abs(angles_out[0]) >= 180
            id_nxt = 0
        else:
            next_curve = np.abs(angles_out[cnt + 1]) >= 180
            id_nxt = cnt + 1
        #
        o_ = _offset_[cnt]            # offset start segment
        cen = k_cents[cnt]  # cen == cents_old[cnt]  # as a check
        o_nxt = _offset_[id_nxt]     # offset end
        n_ = _extended_[cnt]         # seg from intersections
        #   o_, cen, o_nxt, n_
        #
        if ang_ >= 180:
            # head = np.array([n_[:2], o_[-2:]], ndmin=True)  # note below
            # segs.append(head)  # add the seg before arc
            #
            p_st, cent, p_en = o_[-2:], cen, o_nxt[:2]
            arc = _a_(circ_pnts, p_st, cent, p_en,
                      radius=buff_dist, step=step, outside=True)
            # -- check line intersections
            #
            mn = min([p_st[0], p_en[0]])
            mx = max([p_st[0], p_en[0]])
            N = cnt + 1  # skipping current pairs
            w0 = np.logical_and(_offset_[N:, 0] > mn, _offset_[N:, 0] < mx)
            w1 = np.logical_and(_offset_[N:, 2] > mn, _offset_[N:, 2] < mx)
            rows = _offset_[N:]
            chk = rows[w0 | w1]
            if len(chk) > 0:
                kp = []
                for i in chk:
                    v = line_circ_intersection(cen, i[:2], i[-2:], radius=1)
                    if v:
                        kp.append(v)
                if len(kp) > 0:
                    lin = kp[0]
                    if lin[0][1] <= cent[1]:  # line y <= cent y, line ascends
                        new_x = lin[2][0]
                        arc_0 = arc[np.logical_and(
                            arc[:, 0] >= arc[0][0], arc[:, 0] <= new_x)
                            ]
                        if len(arc_0) == 0:  # swap the check if above is empty
                            arc_0 = arc[np.logical_and(
                                arc[:, 0] <= arc[0][0], arc[:, 0] >= new_x)
                                ]
                    new_end = lin[2]  # use this as out last end point
                    _extended_[cnt + 1][:2] = new_end  # replace for the next loop
                    _offset_[cnt + 1][:2] = new_end
                    segs.append(arc_0)  # append sub arc
                else:
                    segs.append(arc)
            else:
                segs.append(arc)  # full arc can be used
        else:
            strt = o_[:2] if next_curve else n_[:2]  # n_[:2]
            head = np.array([strt,  n_[-2:]], ndmin=True)  # note above
            segs.append(head)  #
    #
    # frst = segs[0][0][None, :]
    # segs.append(frst)
    final = np.concatenate((segs), axis=0)
    return final  # plot_polygons([final, poly])


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
