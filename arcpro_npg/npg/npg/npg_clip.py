# -*- coding: utf-8 -*-
# noqa: D205, D400, F403, F401
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
    2022-03-04

Example
-------
Put example here.
"""
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E1101,E1121
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0621
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915


import sys
import math
# from textwrap import dedent
import numpy as np

# -- optional numpy imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

import npg
from npg import npg_plots
from npg.npg_plots import plot_polygons
# from npg.npg_utils import time_deco
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)


script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['clip', 'common_extent', 'roll_arrays', 'split_seq']
__helpers__ = ['_concat_', '_is_pnt_on_line_']
__all__ = __helpers__ + __all__


# ---- (1) general helpers
#
def _is_pnt_on_line_(start, end, xy, tolerance=1.0e-12):
    """Perform a distance check of whether a point is on a line.

    eps = 2**-52 = 2.220446049250313e-16
    np.finfo(float).eps = 2.220446049250313e-16
    np.finfo(float)
    finfo(resolution=1e-15, min=-1.7976931348623157e+308,
          max=1.7976931348623157e+308, dtype=float64)
    """
    #
    def dist(a, b):
        """Add math.sqrt() for actual distance."""
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    #
    line_leng = dist(start, end)
    if tolerance == 0.0:
        return dist(start, xy) + dist(end, xy) == line_leng
    else:
        d = (dist(start, xy) + dist(end, xy)) - line_leng
        return -tolerance <= d <= tolerance


def _roll_(ar, num=1):
    """Roll coordinates by `num` rows so that row `num` is in position 0."""
    return np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0)


def common_extent(a, b):
    """Return the extent overlap for two polygons as L, B, R, T or None."""
    ext0 = np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))
    ext1 = np.concatenate((np.min(b, axis=0), np.max(b, axis=0)))
    es = np.concatenate((ext0[None, :], ext1[None, :]), axis=0)
    maxs = np.max(es, axis=0)
    mins = np.min(es, axis=0)
    L, B = maxs[:2]
    R, T = mins[2:]
    if (L <= R) and (B <= T):
        return (True, np.array([L, B, R, T]))  # (x1, y1, x2, y2)
    return (False, None)


def roll_arrays(arrs):
    """Roll point coordinates to a new starting position.

    Notes
    -----
    Rolls the coordinates of the Geo array attempting to put the start/end
    points as close to the lower-left of the ring extent as possible.
    """
    # --
    def _LL_(arr):
        """Return the closest point to the lower left of the polygon."""
        LL = np.min(arr, axis=0)
        idx = (np.abs(arr - LL)).argmin(axis=0)
        return idx[0]
    # --
    if not isinstance(arrs, (list, tuple)):
        print("List/tuple of arrays required.")
        return
    out = []
    for ar in arrs:
        num = _LL_(ar)
        out.append(np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0))
    return out


def uniq_1d(arr):
    """Return mini 1D unique for sorted values."""
    arr.sort()
    mask = np.empty(arr.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = arr[1:] != arr[:-1]
    return arr[mask]


def uniq_2d(arr, return_sorted=False):
    """Return mini `unique` for 2D coordinates.  Derived from np.unique.

    Notes
    -----
    For returning in the original order this is equivalent to::

        u, idx = np.unique(x_pnts, return_index=True, axis=0)
        x_pnts[np.sort(idx)]
    """
    def _reshape_uniq_(uniq, dt, shp):
        n = len(uniq)
        uniq = uniq.view(dt)
        uniq = uniq.reshape(n, *shp[1:])
        uniq = np.moveaxis(uniq, 0, 0)
        return uniq

    arr = np.asarray(arr)
    shp = arr.shape
    dt = arr.dtype
    st_arr = arr.view(dt.descr * shp[1])
    ar = st_arr.flatten()
    if return_sorted:
        perm = ar.argsort(kind='mergesort')
        aux = ar[perm]
    else:
        # ar.sort(kind='mergesort')  # removed
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    ret = aux[mask]
    uniq = _reshape_uniq_(ret, dt, shp)
    if return_sorted:  # return_index in unique
        return uniq, perm[mask]
    return uniq


def to_struct(whr, x_pnts, as_view=True):
    """Return structured array for intersections.

    Use z.reshape(z.shape[0], 1) to view in column format.
    """
    dt = [("p_seg", "i8"), ("c_seg", "i8"), ("x", "f8"), ("y", "f8")]
    z = np.empty((whr.shape[0],), dtype=dt)
    z["p_seg"] = whr[:, 0]
    z["c_seg"] = whr[:, 1]
    z["x"] = x_pnts[:, 0]
    z["y"] = x_pnts[:, 1]
    if as_view:
        return z.reshape(z.shape[0], 1)
    return z


def reverse_whr(whr, x_pnts, return_pnts=True):
    """Return the reverse of the intersections of the `whr` array.

    Parameters
    ----------
    whr : array-like
        The intersection ids of the clipper/poly.
    x_pnts : array
        The intersection points of the two polygons in the `whr` order.

    Returns
    -------
    Returns the new `whr` and `crossings` arrays for both incarnations of the
    polygon intersections.  Optionally, the intersection points are returned.

    """
    def _crossings_(w):
        """Return the crossing array."""
        s, r = divmod(w.shape[0], 2)  # check for even pairing
        if r == 0:
            x = (w.reshape(-1, 4)).copy()
            x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
        else:
            x = (w[:s * 2].reshape(-1, 4)).copy()
            x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
            lstx, lsty = w[s * 2:][0]
            x = np.concatenate((x, np.array([[lstx, -1, lsty, -1]])), axis=0)
        crossings = x.copy()
        return crossings

    # --
    cross1 = _crossings_(whr)
    idx = np.lexsort((whr[:, 0], whr[:, 1]))
    whr2 = whr[idx].copy()
    whr2.T[[0, 1]] = whr2.T[[1, 0]]
    cross2 = _crossings_(whr2)
    if return_pnts:
        x_pnts2 = x_pnts[idx]
        return whr, cross1, x_pnts, whr2, cross2, x_pnts2
    return whr, cross1, whr2, cross2


# ---- (2) winding number, intersections
#
def _wn_clip_(pnts, poly, all_info=True):
    """Return points in a polygon or on its perimeter, using `winding number`.

    Parameters
    ----------
    pnts, poly : array_like
        Geometries represent the points and polygons.  `pnts` is assumed to be
        another polygon since clipping is being performed.
    all_info : boolean
        True, returns points in polygons, the in and out id values, the
        crossing type and winding number.  False, simply returns the winding
        number, with 0 being outside points and -1 being inside points for a
        clockwise-oriented polygon.

    Notes
    -----
    Negative and positive zero np.NZERO, np.PZERO == 0.0.

    Other
    -----
    z = np.asarray(np.nonzero(npg.eucl_dist(a, b) == 0.)).T
    a[z[:, 0]] and b[z[:, 1]] return the points from both arrays that have a
    distance of 0.0 and they intersect
    """
    def _w_(a, b, all_info):
        """Return winding number and other values."""
        x0, y0 = a[:-1].T   # point `from` coordinates
        # x1, y1 = a[1:].T  # point `to` coordinates
        x1_x0, y1_y0 = (a[1:] - a[:-1]).T
        #
        x2, y2 = b[:-1].T  # polygon `from` coordinates
        x3, y3 = b[1:].T   # polygon `to` coordinates
        x3_x2, y3_y2 = (b[1:] - b[:-1]).T
        # reshape poly deltas
        x3_x2 = x3_x2[:, None]
        y3_y2 = y3_y2[:, None]
        # deltas between pnts/poly x and y
        x0_x2 = x0 - x2[:, None]
        y0_y2 = y0 - y2[:, None]
        #
        a_0 = y0_y2 * x3_x2
        a_1 = y3_y2 * x0_x2
        b_0 = y0_y2 * x1_x0
        b_1 = y1_y0 * x0_x2
        a_num = a_0 - a_1
        b_num = b_0 - b_1
        #
        # pnts in poly
        chk1 = (y0_y2 >= 0.0)  # y above poly's first y value, per segment
        chk2 = np.less(y0, y3[:, None])  # y above the poly's second point
        chk3 = np.sign(a_num).astype(np.int32)
        pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=0, dtype=np.int32)
        neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=0, dtype=np.int32)
        wn_vals = pos - neg
        wn_ = np.concatenate((wn_vals, np.array([wn_vals[0]])))
        #
        if all_info:
            denom = (x1_x0 * y3_y2) - (y1_y0 * x3_x2)
            return wn_, denom, x0, y0, x1_x0, y1_y0, a_num, b_num
        return wn_

    def _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0):
        """Return the intersection."""
        with np.errstate(all="ignore"):  # ignore all errors
            u_a = a_num / denom
            u_b = b_num / denom
            z0 = np.logical_and(u_a >= 0., u_a <= 1.)  # np.isfinite(u_a)`
            z1 = np.logical_and(u_b >= 0., u_b <= 1.)  # np.isfinite(u_b)
            both = (z0 & z1)
            xs = (u_a * x1_x0 + x0)[both]
            ys = (u_a * y1_y0 + y0)[both]
        x_pnts = []
        if xs.size > 0:
            x_pnts = np.concatenate((xs[:, None], ys[:, None]), axis=1)
        whr = np.array(np.nonzero(both)).T
        return whr, x_pnts
    # --
    # Use `_w_` and `_xsect_` to determine pnts in poly
    wn_, denom, x0, y0, x1_x0, y1_y0, a_num, b_num = _w_(pnts, poly, True)
    whr, x_pnts = _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0)
    p_in_c = np.nonzero(wn_)[0]
    # p_out_c = np.nonzero(wn_ + 1)[0]
    x_type = np.concatenate((wn_[:-1, None], wn_[1:, None]), axis=1)
    #
    wn2_ = _w_(poly, pnts, False)  # get poly points in other geometry
    c_in_p = np.nonzero(wn2_)[0]
    # c_out_p = np.nonzero(wn2_ + 1)[0]
    vals = [x_pnts, p_in_c, c_in_p, x_type, whr]
    # if ..outs needed [x_pnts, p_in_c, c_in_p, c_out_p, ...wn_, whr]
    if all_info:
        return vals
    return whr  # wn_


# ---- (3) polygon clipping
# -- helpers
def _concat_(x):
    """Concatenate array pieces."""
    z1 = [np.atleast_2d(i) for i in x if len(i) > 0]
    return np.concatenate(z1, axis=0)


def _dist_(a, b):
    """Return squared distance.  Add math.sqrt() for actual distance."""
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


def _a_eq_b_(a, b, atol=1.0e-8, rtol=1.0e-5, return_pnts=False):
    """
    Return the points in `b` that are equal to the points in `a`.

    Parameters
    ----------
    a, b : ndarrays
        The arrays must have ndim=2, although their shapes need not be equal.

    See Also
    --------
    See `npg_helpers.a_eq_b` for details.

    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if b.size > 2:
        b = b[:, None]
    w = np.less_equal(np.abs(a - b), atol + rtol * abs(b)).all(-1)
    if w.ndim > 1:
        w = w.any(-1).squeeze()
    if return_pnts:
        return b[w].squeeze()
    return w.squeeze()  # w.any(0).any()


def _eq_(p_0, c_0):
    """Return whether the poly and clp points are equal."""
    v0, v1 = p_0[c_0] - c_0[p_0]
    return math.isclose(v0, v1)


def _before_chk_(poly, seen_, id_, a_in_b):
    """Return the points which are before the point in question.

    Parameters
    ----------
    poly : array_like
        The polygon to check.
    seen_ : list
        The id values already to seen.
    id_ : integer
        The id value to check
    a_in_b : list
    """
    lst = seen_ + [id_]
    strt = lst[0] if len(lst) > 1 else 0
    end_ = lst[-1] + 1
    bf = [x for x in range(strt, end_) if x not in lst]
    if len(bf) > 0:
        bf = sorted(list(set(a_in_b).intersection(bf)))
        pnts = [poly[i] for i in bf]
        if pnts:
            return bf, pnts
    return bf, None


def _btw_chk_(a, b, in_, seen):  # include_ends=True):
    """Return points which are between/within polygon ids `a` and `b`.

    Notes
    -----
    If `a` > `b`, then any found points will be returned in reverse order.

    Example
    -------
    Order of entities seen is important so they are returned in that order.

    >>> _btw_chk_(a=10, b=0, in_=[3, 4, 5], seen=[3])
    ... [5, 4]
    >>> _btw_chk_(a=0, b=10, in_=[3, 4, 5], seen=[3])
    ... [4, 5]
    """
    if a > b:
        a, b = b, a
    btw = set(range(a, b + 1))
    if len(btw) > 0:
        s = set(in_).difference(set(seen))
        ids = btw.intersection(s)  # or [i for i in btw if i in s]
        if ids:
            return sorted(list(ids))
    return []


def _to_add_(XCsub, sub_, ply_seen, clp_seen, tot_):
    """Simplify updating ply_seen, clp_seen and tot_ arrays."""
    c_0, c_1, p_0, p_1 = XCsub[:4]
    clp_seen = list(set(clp_seen).union([c_0, c_1]))
    ply_seen = list(set(ply_seen).union([p_0, p_1]))
    tot_.append([XCsub.ravel(), sub_])
    return ply_seen, clp_seen, tot_


def nodes(p_in_c, c_in_p, clp, poly, x_pnts):
    """Return node intersection data. `_cpx_ ,clipper polygon intersection`.

    Parameters
    ----------
    p_in_c, c_in_p : lists
        Id values of points in poly and clipper respectively.
    clp, poly : array_like
        The geometry of the clipper and the polygon being clipped.
    x_pnts : array_like
        The intersection points of the geometry edges.

    Returns
    -------
    - p_in_c : polygon points in clipper and reverse
    - c_in_p : clipper points in polygon and those that are equal
    - c_eq_p, p_eq_c : clipper/polygon equality
    - c_eq_x, p_eq_x : intersection points equality checks for both geometries
    - cp_eq, cx_eq, px_eq : clipper, polygon and intersection equivalents

    Notes
    -----
    Forming a dictionary for cp_eq, cs_eq, px_eq::

        kys = uniq_1d(px_eq[:, 0]).tolist()  # [ 0,  2,  3,  4, 12]
        dc = {}  # -- dictionary
        dc[0] = px_eq[px_eq[:, 0] == 0][:,1].tolist()
        for k in kys:
            dc[k] = px_eq[px_eq[:, 0] == k][:,1].tolist()
        dc
        {0: [0, 3, 13, 14],
         2: [7, 8],
         3: [9, 10, 11],
         4: [1, 2, 5, 6],
         12: [0, 3, 13, 14]}

    Or, you can split the array::

        whr = np.nonzero(np.diff(cx_eq[:, 0]))[0] + 1
        np.array_split(cx_eq, whr)
        [array([[ 0,  0],
                [ 0,  3],
                [ 0, 13],
                [ 0, 14]], dtype=int64),
         array([[1, 1],
                [1, 2],
                [1, 5],
                [1, 6]], dtype=int64),
         array([[2, 4]], dtype=int64),
         array([[3, 7],
                [3, 8]], dtype=int64),
         array([[ 4,  9],
                [ 4, 10],
                [ 4, 11]], dtype=int64),
         array([[ 6,  0],
                [ 6,  3],
                [ 6, 13],
                [ 6, 14]], dtype=int64)]

    # -- Point equality check. -- c_eq_p, c_eq_x, p_eq_c, p_eq_x
    # poly[p_eq_c], poly[p_eq_x] and clp[c_eq_p], clp[c_eq_x]
    """
    # p_out_c = []  # not used anymore
    c_eq_p, p_eq_c = np.nonzero((poly == clp[:, None]).all(-1))
    if c_eq_p.size > 0:
        c_eq_p = uniq_1d(c_eq_p)
        p_eq_c = uniq_1d(p_eq_c)
    p_eq_x, x_eq_p = np.nonzero((x_pnts == poly[:, None]).all(-1))
    if p_eq_x.size > 0:
        p_eq_x = uniq_1d(p_eq_x)
    c_eq_x, x_eq_c = np.nonzero((x_pnts == clp[:, None]).all(-1))
    if c_eq_x.size > 0:
        c_eq_x = uniq_1d(c_eq_x)
    if p_eq_c.size > 0 or p_eq_x.size > 0:  # p_in_c + (c_eq_p, p_eq_x)
        p_in_c = np.unique(np.concatenate((p_in_c, p_eq_c, p_eq_x)))
    # -- (2) clp, poly point equal
    if c_eq_p.size > 0 or c_eq_x.size > 0:  # c_in_p + (p_eq_c, c_eq_x)
        c_in_p = uniq_1d(np.concatenate((c_in_p, c_eq_p, c_eq_x)))
    args = (p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x)
    # cp_eq, cx_eq, px_eq)  simplified output
    out = [i.tolist() if isinstance(i, np.ndarray) else i for i in args]
    return out


def _x_mkr_(to_chk, x_pnts, p_in_c, c_in_p):  # poly, p_eq_x, c_eq_x):
    """Return intersections/crossings and checks, `xCheck`, given inputs.

    to_chk : whr values, whr or whr_on
    x_pnts : x_pnts or x_pnts_on
    poly   : input polygon
    p_in_c, c_in_p : polygon points in clipper and clipper points in polygon
    """
    s, r = divmod(to_chk.shape[0], 2)  # check for even pairing
    if r == 0:
        x = (to_chk.reshape(-1, 4)).copy()
        x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
    else:
        x = (to_chk[:s * 2].reshape(-1, 4)).copy()
        x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
        lastx, lasty = to_chk[s * 2:][0]
        x = np.concatenate((x, np.array([[lastx, -1, lasty, -1]])), axis=0)
    crossings = x.copy()
    z0 = np.isin(crossings[:, :2], c_in_p)
    z1 = np.isin(crossings[:, 2:4], p_in_c)
    #
    in_chks = np.concatenate((z0, z1), axis=1)
    xCheck = np.concatenate((crossings, in_chks.astype(int)), axis=1)
    # -- intersection points
    x0x1 = [x_pnts[i: i + 2] for i in range(0, len(to_chk), 2)]
    if x0x1[-1].shape == (1, 2):
        pad = np.concatenate((x0x1[-1], np.array([[np.nan, np.nan]])), axis=0)
        x0x1[-1] = pad
    return xCheck, x0x1


def _f0f1_(_f0, _f1, _last, _st_en, _in, _eq, _seen):
    """Pick `from-to` points for various functions."""
    if _f0 == 0:
        _f0, _f1 = _last, _st_en
    elif _f0 != 0 and _f0 < _f1:
        _f0, _f1 = 0, _f0
    tmp = _btw_chk_(_f0, _f1, _in, _seen)
    return [i for i in tmp if i not in _eq]


def split_seq(seq, last, prn=False):  # sec_last
    """Return a sequence of point ids split at its numeric gaps.

    The defined gap is 1, since we are working with sequences of points.

    Parameters
    ----------
    seq : array_like
        A sorted ndarray is expected, but will be converted if needed.  Sorting
        is your task.
    sec_last, last : integers
        Indices of the second last and last points in the sequence.

    Returns
    -------
    A list of sub-arrays or the original sequence if there are no missing
    values in the sequence.
    """
    if len(seq) == 0:
        return seq
    if isinstance(seq, (list, tuple)):
        seq = np.asarray(seq)
    if seq.ndim > 1 or seq.shape[0] <= 1:
        if prn:
            print("\n A 1D ndarray required.")
        return [seq.tolist()]
    if seq[0] == 0 and last == seq[-1]:
        tmp = seq[1:]
        whr = np.nonzero(np.abs(tmp[1:] - tmp[:-1]) != 1)[0]
        if whr.size > 0:
            z = [s.tolist() for s in np.array_split(tmp, whr + 1)]
            lst = z.pop()
            z.insert(0, lst)
            return z
    else:
        whr = np.nonzero(np.abs(seq[1:] - seq[:-1]) != 1)[0]
        if whr.size > 0:
            return [s.tolist() for s in np.array_split(seq, whr + 1)]
    return [seq.tolist()]


def p_type(ply, eqX, eqOther, inOther):
    """Return point class for polygon.

    Parameters
    ----------
    ply : ndarray
        Polygon points to be classed.
    eqX, eqOther, inOther : ndarrays or lists of values
      - poly/clipper equal to an intersection point
      - one equals the other point
      - one is in the other
        column names (0, 1, 2 positionally)

    Requires
    --------
    `_wn_clip_`, `nodes` are used to determine whether each point meets the
    conditions outlined above.

    Notes
    -----
    Conversion values are based on binary conversion as shown in the
    `keys` line, then reclassed using a dictionary conversion.

    - keys = eqX * 100 + eqOther * 10 + inOther
    - 0 (0 0 0) is outside
    - 1 (0 0 1) is inside with no intersection
    -   position before a 1 is an intersection point not at an endpoint
    - 5 (1 0 1) endpoint intersects a segment
    - 7  (111) clp, poly and intersection meet at a point.

    Example
    -------
    >>> f0 = p_type(poly, p_eq_x, p_eq_c, p_in_c)
    >>> f1 = p_type(clp, c_eq_x, c_eq_p, c_in_p)
    """
    k = [0, 1, 10, 11, 100, 101, 110, 111]
    v = [0, 1, 2, 3, 4, 5, 6, 7]
    d = dict(zip(k, v))
    N = ply.shape[0]
    z = np.zeros((N, 5), 'int')
    z[:, 0] = np.arange(N)
    z[:, 1][eqX] = 1
    z[:, 2][eqOther] = 1
    z[:, 3][inOther] = 1
    keys = z[:, 1] * 100 + z[:, 2] * 10 + z[:, 3]
    vals = [d[i] for i in keys.tolist()]
    z[:, 4] = vals
    return z


# -- main
def clip(poly, clp):
    """Return the result of a polygon clip.

    Parameters
    ----------
    poly, clp : ndarrays representing polygons
        Rings are oriented clockwise.  Holes are ignored.

    Requires
    --------
    -  `_wn_clip_`, `nodes`, `_x_mkr_`
    - `_concat_`, `_dist_`, `_a_eq_b_`, `_before_chk_`, `_btw_chk_`, `_to_add_`

    - Fixes **** p_out_c, c_out_p not used so commented out.

    # -- `run a bail_chk_` ... doesn't work b0,E because all points are in
    # if nP == len(p_in_c):  # but all of b0 is retained instead of the E shape
    #     print("...poly completely within clp...")
    #     return poly, None, None
    # elif nC == len(c_in_p):
    #     print("... clp completely within poly...")
    #     return clp, None, None
    #
    Notes
    -----
    Sample data::

        t0 = np.array([[0., 0], [0., 2], [8., 2], [8., 0.], [0., 0]])
        t01 = np.array([[1., 0], [1., 2], [3., 2], [8., 2], [8., 0.], [0., 0]])
        t1 = np.array([[0., 0], [0., 2], [1., 2.], [8., 2], [8., 0.], [0., 0]])
        t2 = np.array([[0., 0], [0., 2], [2., 2.], [8., 2], [8., 0.], [0., 0]])
        t3 = np.array([[0., 0], [0., 2], [4., 2.], [8., 2], [8., 0.], [0., 0]])
        t4 = np.array([[0., 0], [0., 2], [4., 2.], [6., 2], [8., 2],
                       [8., 0.], [0., 0]])
        s0 = np.array([[1., 1.], [2., 2.], [5., 2.], [6., 1.]])
        s00 = np.array([[1., 1.], [2., 2.], [5., 2.], [6., 1.], [1., 1.]])
        s01 = np.array([[1., 1.], [3., 3.], [4., 3.], [6., 1.], [1., 1.]])
        s02 = np.array([[1., 1.], [3., 3.], [3.5, 1.5], [4., 3.], [6., 1.],
                        [1., 1.]])
        s03 = np.array([[1., 1.], [2., 2], [3., 3.], [3.5, 1.5], [4., 3.],
                        [6., 1.], [1., 1.]])  # dupl x on line
    """
    # --
    if hasattr(poly, "IFT"):
        poly = poly.XY
    if hasattr(clp, "IFT"):
        clp = clp.XY
    #
    # -- winding number to get points inside, on, outside each other
    poly, clp = roll_arrays([poly, clp])  # roll the arrays to orient to LL
    vals = _wn_clip_(poly, clp, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals  # wn_
    #
    if len(x_pnts) == 0:
        print("No intersection between `poly` and `clp`.")
    #     return clp, None, None
    #
    # -- derive the unique intersection points and get their first found index.
    uni, idx, cnts = np.unique(x_pnts, return_index=True,
                               return_counts=True, axis=0)
    idx_srt = np.sort(idx)       # get the original order
    x_pnts_on = x_pnts[idx_srt]
    whr_on = whr[idx_srt]        # whr crossings in that order
    # -- run `nodes` to get the nodes, and their info
    args = nodes(pInc, cInp, clp, poly, x_pnts)
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args  # [:6]
    # cp_eq, cx_eq, px_eq = args[-3:]
    #
    nC = clp.shape[0]
    nP = poly.shape[0]
    # -- run `_x_mkr_` for both types of analysis until I choose.
    xChk, x0x1 = _x_mkr_(whr_on, x_pnts_on, p_in_c, c_in_p)  # poly
    # difference in index pairs for clp and poly
    dC_dP = np.abs(np.vstack((xChk[:, 1] - xChk[:, 0],
                              xChk[:, 3] - xChk[:, 2]))).T
    #
    c_prev, p_prev = [-1, -1]
    p_seen, c_seen, tot_ = [[], [], []]  # create empty lists
    c_last, c_st_en = nC - 2, nC - 1     # -- clp  : last, dupl. start-end pnt
    p_last, p_st_en = nP - 2, nP - 1     # -- poly : last, dupl. start-end pnt
    #
    p_seq = split_seq(p_in_c, p_st_en)  # p_last
    c_seq = split_seq(c_in_p, c_st_en)  # c_last
    Np_seq = len(p_seq)
    Nc_seq = len(c_seq)
    for cnt, xC in enumerate(xChk):  # xChk is the crossings list
        c0_fr, c1_fr, p0_fr, p1_fr = xC[:4]
        c0_to, c1_to, p0_to, p1_to = xC[:4] + 1
        c0_in, c1_in, p0_in, p1_in = np.asarray(xC[-4:], bool)
        in_case = "{} {} {} {}".format(*np.asarray(xC[-4:]))
        x0, x1 = x0x1[cnt]  # -- intersection points for the segment pair
        diff_C, diff_P = dC_dP[cnt]  # difference in indices
        bf_c, btw_c = [], []         # Initial values for `before`, `between`
        bf_p, btw_p = [], []         # for both poly and clp
        c0c1_ = in_case[:3]
        p0p1_ = in_case[4:]
        sub_ = []
        #
        seq_p = [] if cnt + 1 > Np_seq else p_seq[cnt]
        seq_c = [] if cnt + 1 > Nc_seq else c_seq[cnt]
        # --
        #
        # Last segment with only one intersection point.
        if c1_fr == -1:  # one of x0,x1 will be (nan,nan)
            p_max = max(p_seen)
            c_max = max(c_seen)
            btw_p = _btw_chk_(p_max, p0_fr, p_in_c, p_seen)
            btw_c = _btw_chk_(c_max, c0_fr, c_in_p, c_seen)
            if btw_p:
                tmp = [i for i in btw_p if i in p_in_c and i not in p_eq_x]
                if tmp:
                    p_seen.extend(tmp)
                    sub_.extend(poly[tmp])
            if btw_c:
                tmp = [i for i in btw_c if i in c_in_p and i not in c_eq_x]
                if tmp:
                    c_seen.extend(tmp)
                    sub_.extend(clp[tmp])
            zz = x1 if np.isnan(x0).all() else x0  # get the correct x0,x1
            sub_.append(zz)  # was x0, but sometimes x1 is the one! check pairs
            _out_ = _to_add_(xC, sub_, p_seen, c_seen, tot_)
            p_seen, c_seen, tot_ = _out_
            #
            # print(f"{cnt} {x0}, {x1}  {xC}")
            # --
            break
        #
        # -- pre section
        # -- Either `bf_c` or `bf_p` will have values, not both
        if cnt == 0:
            # -- before_c `bf_c`
            if c0_in or diff_C == 0:
                bf_c = _f0f1_(c0_fr, c1_fr, c_last, c_st_en,
                              c_in_p, c_eq_x, c_seen)
            if bf_c:
                c_seen.extend(bf_c)
                if c_st_en in bf_c or c_last in bf_c:
                    c_seen.extend([0])
                sub_.extend(clp[bf_c])
            # -- before_p `bf_p`
            if p0_in or diff_P == 0:
                if p0_fr == 0 and p_st_en in seq_p:
                    bf_p = seq_p
                else:
                    bf_p = _f0f1_(p0_fr, p1_fr, p_last, p_st_en,
                                  p_in_c, p_eq_x, p_seen)
            if bf_p:
                p_seen.extend(bf_p)
                if p_st_en in bf_p or p_last in bf_p:
                    p_seen.extend([0])
                sub_.extend(poly[bf_p])
        # --
        # -- subsequent crossings
        elif cnt > 0:
            c_prev, p_prev = xChk[cnt - 1][[1, 3]]  # slice previous values
            if c0_fr - c_prev > 1 and c_prev not in c_eq_x:
                if (c_prev in c_in_p) and (c_prev not in c_seen):
                    sub_.extend([clp[c_prev]])
                    c_seen.extend([c_prev])
                bf_c = _btw_chk_(c0_fr, c_prev, c_in_p, c_seen)
                # -- include other clp pnts inside
                if bf_c:
                    sub_.extend(clp[bf_c])
                    c_seen.extend(bf_c)
            # -- repeat for poly differences
            if p0_fr - p_prev > 1 and p_prev not in p_eq_c:
                if (p_prev in p_in_c) and (p_prev not in p_seen):
                    sub_.extend([poly[p_prev]])
                    p_seen.extend([p_prev])
                bf_p = _btw_chk_(p0_fr, p_prev, p_in_c, p_seen)
                # -- include any other poly pnts inside
                if bf_p:
                    bf_p = [i for i in bf_p if i not in p_eq_x]
                    sub_.extend(poly[bf_p])
                    p_seen.extend(bf_p)
            # --
        # -- end `before`
        # -- begin `between` section
        if diff_C != 0:
            btw_c = _btw_chk_(c0_fr, c1_fr, c_in_p, c_seen)
            if btw_c and c1_fr - c0_fr == 1:  # differ by 1 and both inside
                c_seen.extend(btw_c)
            elif btw_c and c1_fr - c0_fr == 2:  # unique case!
                btw_c = [i for i in btw_c if i in c_in_p]  # and i in c_eq_x]
                # c == x
            elif btw_c:
                btw_c = [i for i in btw_c if i in c_in_p and i not in c_eq_x]
                c_seen.extend(btw_c)
        if diff_P != 0:
            btw_p = _btw_chk_(p0_fr, p1_fr, p_in_c, p_seen)
            if len(btw_p) > 1:  # -- skip sequences with gaps eg b4,c1
                gap = btw_p[0] + len(btw_p) - 1 != btw_p[-1]
                if gap:
                    btw_p = []
            else:
                p_seen.extend(btw_p)  # B0,a has 1 value
        # print(f"{cnt} {x0}, {x1}  {xC}")
        # print(f"  bfc:bfp {bf_c}  {bf_p}\n  btc:btp {btw_c}  {btw_p}")
        # print(f"{cnt} {x0}, {x1}  {c0c1_}{p0p1_}")
        # -------- '0 0' prep --------------------------------------------
        if c0c1_ == '0 0':
            # -- Both pnts out but cross poly and poly pnts may be inside.
            #  `btw_p` may be empty or return a list of indices.
            if diff_C == 0 and btw_p:  # b4, c1 issue
                sub_.extend(poly[btw_p])  # btw_p added to p_seen before
                btw_p = None  # -- empty btw_p
            #
            # -----------------------------------
            # -- poly checks
            # --
            if p0p1_ == '0 0':                           # 0,  0,  0,  0
                if p0_fr == 0:  # check when p0==0 and change its value
                    if p1_to == p_st_en:  # see if you are back at the start
                        sub_.extend([x1, x0])
                    else:
                        sub_.extend([x0, x1])
                elif p0_fr <= p1_fr:
                    sub_.extend([x0, x1])
                elif p0_fr > p1_fr:
                    sub_.extend([x1, x0])
                if p1_to in p_in_c and cnt != 0:
                    sub_.extend([poly[p1_to]])
                if c1_to in c_in_p:
                    sub_.extend([clp[c1_to]])  # c1,b3 add last clp
            # --
            elif p0p1_ == '0 1':                           # 0,  0,  0,  1
                if p1_fr in p_eq_x:
                    sub_.extend([x0, x1])
                    p_seen.append(p1_fr)
                elif p0_fr == 0 or p0_fr < p1_fr:
                    sub_.append(x1)
                    if btw_p:  # -- set to [] in c0c1_ section, btw_p
                        sub_.extend(poly[btw_p])  # checks may be redundant
                    sub_.append(x0)
                else:
                    sub_.append(x1)
                    if btw_p:
                        sub_.extend(poly[btw_p])
                    sub_.append(x0)
                if p0_fr not in p_in_c and p0_to not in p_seen:  # p0_fr not in
                    sub_.extend([poly[p0_to]])               # check endpoint.
                    p_seen.extend([p0_to])
            # --
            elif p0p1_ == '1 0':                           # 0,  0,  1,  0
                if p0_fr < p1_fr:
                    sub_.extend([x0, x1])
                else:
                    sub_.extend([x1, x0])
                if p0_to in p_in_c and p0_to not in p_seen:  # b4,c1  s03,t0
                    sub_.extend([poly[p0_to]])
                    p_seen.extend([p0_to])
                if p1_to not in p_eq_x and p1_to in p_in_c:  # p1 end check
                    sub_.extend([poly[p1_to]])
                    p_seen.extend([p1_to])
            # --
            elif p0p1_ == '1 1':                           # 0,  0,  1,  1
                # x0,x1 on the clipping line, may be added earlier in c0c1
                # alternate:  all([i in p_eq_x for i in btw_p])
                if btw_p is not None:
                    sub_.extend([x0, x1])  # or btw_p, s00,t0
        #
        # -------- '0 1' prep --------------------------------------------
        elif c0c1_ == '0 1':  # example b0,c1 b1,c0
            # -- prep     bf_p already added if present
            if c0_fr <= c1_fr:  # order additions, add first intersection
                sub_.append(x0)
                if btw_c:
                    sub_.extend(clp[btw_c])
            else:
                if btw_c:
                    sub_.extend(clp[btw_c])
                sub_.append(x0)
            #
            # -----------------------------------
            # -- poly checks
            # --
            if p0p1_ == '0 0':                             # 0,  1,  0,  0
                sub_.append(x1)
                # if p1_to in p_in_c:             # didn't work for poly, clp
                #     sub_.extend([poly[p1_to]])  # from featureclass
                #     p_seen.extend([p1_to])      # it extrapolated
            # --
            elif p0p1_ == '0 1':  # p1_seen:  # not done
                sub_.extend([poly[p1_fr]])                 # 0,  1,  0,  1
            # --
            elif p0p1_ == '1 0':   # example b0,c1         # 0,  1,  1,  0
                # -- add pnts in the correct order
                if p0_fr < p1_fr:  # b4,c1  p1_to is > p0_fr & p1_fr
                    if p0_to not in btw_p and p1_to not in btw_p:  # edgy1,ecl
                        if len(btw_p) > 0:
                            sub_.extend(poly[btw_p])
                    sub_.append(x1)
                else:
                    sub_.append(x1)
                    if btw_p and p1_to not in btw_p:
                        sub_.extend(poly[btw_p])
                if p1_to in p_in_c and p1_to not in p_seen + p_eq_x:  # c2,K
                    if len(seq_p) == 0:
                        sub_.append(poly[p1_to])
                    elif len(seq_p) == 1:
                        sub_.extend([poly[p1_to]])
                    elif len(seq_p) > 1:  # b4, c1
                        sub_.extend(poly[seq_p])
                        # to_add = [i for i in seq_p if i not in p_seen]
                        # if to_add:
                        #     sub_.extend(poly[to_add])
                    p_seen.extend([p1_to])
            # --
            elif p0p1_ == '1 1':  # 2 clp on same poly     # 0,  1,  1,  1
                if c1_fr in c_in_p and c1_fr not in c_seen:
                    sub_.extend([clp[c1_fr]])
                sub_.append(x1)
                tmp = p1_fr  # -- needed for cases where p0, p1 are equal
                if diff_P == 0 and p0_to in c_in_p:  # diff_P == 0
                    tmp = p0_to
                    sub_.extend([poly[tmp]])
                    p_seen.extend([tmp])
        #
        # -------- '1 0' prep --------------------------------------------
        elif c0c1_ == '1 0':  # c0, b0  header is the reverse of '0 1'
            #
            if c0_fr in c_seen and cnt != 0:  # check for last clipping line
                if c0_fr in c_eq_x and c1_to == c_st_en:
                    break
                    # last clipping line,
            if c0_fr in btw_c:  # added for c1, b5
                sub_.extend(clp[btw_c])
                sub_.append(x0)
            elif c0_fr <= c1_fr:
                sub_.append(x0)
                if btw_c:
                    sub_.extend(clp[btw_c])
            elif c0_fr > c1_fr:
                if btw_c:
                    sub_.extend(clp[btw_c])
                sub_.append(x1)
            #
            # -----------------------------------
            # -- poly checks
            # --
            if p0p1_ == '0 0':                             # 1,  0,  0,  0
                if diff_P == 0:  # p0_fr to p0_to (p0_to=p1_fr), eg, 2 2
                    sub_.append(x1)  # c1, b3
                elif not sub_:
                    sub_.append(x0)
                elif btw_c:  # btw_c could be 1 or more points
                    if c0_fr in btw_c and len(btw_c) == 1:
                        tmp_ = [clp[c0_fr]]
                    else:
                        tmp_ = clp[btw_c]
                    sub_.extend(tmp_)
                    sub_.append(x0)
                if p1_to in p_in_c and p1_to in p_eq_x:
                    p_seen.extend([p1_to])
                # sub_.append(x1)  # ** changed to try and fix edgy1/eclip
            # --
            elif p0p1_ == '0 1':                           # 1,  0,  0,  1
                # -- add pnts in the correct order
                if p0_fr <= p1_fr:
                    if btw_p:  # btw_p could be 1 or more points
                        sub_.extend(poly[btw_p])
                        p_seen.extend(btw_p)
                    sub_.append(x1)
                else:
                    sub_.append(x1)
                    if btw_p:
                        sub_.extend(poly[btw_p])
            # --
            elif p0p1_ == '1 0':                           # 1,  0,  1,  0
                sub_.append(x0)
                if diff_P == 1:
                    sub_.extend([poly[p1_fr]])
                if btw_p:
                    sub_.extend(poly[btw_p])
                if c1_fr != -1 and p0_to == p_st_en:
                    sub_.append(x1)
            # --
            elif p0p1_ == '1 1':  # most common
                if diff_P == 0:                            # 1,  0,  1,  1
                    if p0_fr in p_seen:
                        sub_.append(x1)
                    else:
                        sub_.extend([x0, x1])
                elif diff_P == 1 and p0_fr not in p_seen:
                    if p0_fr not in p_eq_c:  # or p0_fr not in btw_p:
                        sub_.extend([poly[p0_fr], x1])  # ?? not sure for K,b0
                    else:
                        sub_.extend([poly[p1_fr], x1])
                elif diff_P >= 1:  # b0,E
                    if len(btw_p) > 1 and p0_fr in btw_p:
                        btw_p = btw_p[1:]
                    sub_.extend(poly[btw_p])  # this is probably x1
                # -- add second crossing
                # sub_.append(x1)  # -- add here ??
                # -- close
                if p1_to in p_in_c and p1_to not in p_eq_x:
                    sub_.extend([poly[p1_to]])
                p_seen.extend([p1_to])
                if (c1_to in c_in_p) and (c1_to not in c_eq_x):
                    sub_.extend([clp[c1_to]])
                c_seen.extend([c1_to])
                # finally
                sub_.append(x1)  # -- or add here in case the above are before
                #
                # if (clp[[c0_fr, c0_to]] == x0x1[cnt]).all():  # b0, E
                #     c_seen.extend([c_0, c0_to])
                #     sub_.extend([x0, x1])
        #
        # ---------- '1 1' prep ------------------------------------------
        elif c0c1_ == '1 1':                               # 1,  1,  ?,  ?
            # -- begin checks  * this is a mess!!!
            #
            if diff_C == 0 or not btw_c:
                sub_.append(x0)
            if c0_to in c_in_p and c0_to not in c_seen and c0_to not in c_eq_x:
                sub_.extend([clp[c0_to]])
                c_seen.extend([c0_fr, c0_to])
            elif diff_C >= 1 and len(btw_c) >= 1:  # >= 1 and btw_c has some
                if btw_c == [c0_fr, c1_fr]:  # both inside
                    btw_c = [i for i in btw_c if i not in c_seen]
                    if btw_c:
                        sub_.extend(clp[btw_c])
                        c_seen.extend(btw_c)
                elif btw_c:  # or diff_C > 1:  # either or don't add any
                    sub_.append(x0)
                    to_add = [i for i in btw_c if i not in c_eq_p]
                    if to_add:
                        sub_.extend(clp[to_add])  # -- had issues before
                        c_seen.extend(to_add)
            # else:
            #    not_used = x0
            #
            # -----------------------------------
            # -- poly checks
            # --
            if p0p1_ == '0 0':                             # 1,  1,  0,  0
                # -- a,C leaves 3 points dangling and were included in c0c1_
                if p0_fr not in p_in_c or p0_fr not in p_eq_x:
                    sub_.extend([x1])
                elif p0_fr not in p_in_c and p0_fr == p_last:
                    # print("unsupported segment, reset sub_ for 1 1 0 0")
                    sub_ = []
                    break
            # --
            elif p0p1_ == '1 0':                           # 1,  1,  1,  0
                # add points between p0_fr and p1_fr
                if btw_p or diff_P > 1:
                    to_add = [i for i in btw_p if i in p_in_c and
                              i not in p_eq_x]
                    if to_add:
                        sub_.extend(poly[to_add])  # -- had issues before
                        p_seen.extend(to_add)
                    btw_p = []  # empty btw_p
                # add the second intersection
                sub_.append(x1)
                # -- now check the end point/next start point for inclusion
                #    but btw_p must be empty
                if p1_to not in p_eq_x and p1_to in p_in_c:
                    sub_.extend([poly[p1_to]])
                p_seen.extend([p1_to])  # **** added for edgy1 3rd clip
            # --
            elif p0p1_ == '0 1':                           # 1,  1,  0,  1
                if p0_fr <= p1_fr:
                    if btw_p:  # btw_p could be 1 or more points
                        sub_.extend(poly[btw_p])
                        p_seen.extend(btw_p)
                    sub_.append(x1)
                else:
                    sub_.append(x1)
                    if btw_p:
                        sub_.extend(poly[btw_p])
                if p1_to in p_in_c:
                    sub_.extend([poly[p1_to]])
                p_seen.extend([p1_to])
            # --
            elif p0p1_ == '1 1':                           # 1,  1,  1,  1
                if btw_p and p0_fr != 0:  # E,b0 to prevent 0,3 adding 2,3
                    sub_.extend(poly[btw_p])
                p_seen.extend(btw_p)
                sub_.append(x1)
        # ----
        # -- Assemble for the output
        p_seen.extend([p0_fr, p1_fr])
        c_seen.extend([c0_fr, c1_fr])
        _out_ = _to_add_(xC, sub_, p_seen, c_seen, tot_)
        p_seen, c_seen, tot_ = _out_
    # --
    new_ = []
    for i in tot_:
        new_.extend([j for j in i[1] if len(j) > 0])
    new_ = np.asarray(new_)
    # dump first point for trailers like a,C
    if (new_[0] == x0).all():
        new_ = new_[1:]
    #
    if not (new_[0] == new_[-1]).all(-1):
        new_ = np.concatenate((new_, new_[0, None]), axis=0)
    return new_, tot_, xChk
    # new_, tot_, xChk = clip(


# ---- (4) multi shape version  ** not final
#
def poly_clip(clippers, polys):
    """Return the inside or outside clipped geometry.

    Parameters
    ----------
    clippers, polys : list or array_like
        The geometries representing the clipping features and the polygon
        geometry.  A list or array with ndim > 2 is expected for both

    Requires
    --------
    `common_extent`, `clip`

    Returns
    -------
    The clipped polygons and the shared_extent between the clipping and poly
    features.

    Example
    -------
    >>> poly_clip([c1], [b0, b1, b2, b3, b4, b5])  # lists provided
    >>> shared_extent
    ... [[0, 0, None],
    ...  snip ...
    ... [0, 2, array([  5.00,  10.50,   6.50,  12.00])],
    ... snip ...
    ...  [0, 5, array([  1.00,  11.00,   4.00,  13.00])]]
    >>> to_keep  # 2 & 5
    ... [array([[  5.00,  12.00],
    ...         [  5.83,  12.00],
    ...         [  5.50,  10.50],
    ...         [  5.00,  10.64],
    ...         [  5.00,  12.00]]),
    ... snip
    ...  array([[  1.00,  11.50],
    ...         [  2.50,  13.00],
    ...         [  4.00,  12.50],
    ...         [  3.12,  11.18],
    ...         [  2.00,  11.50],
    ...         [  1.57,  11.21],
    ...         [  1.00,  11.50]])]
    """
    shared_extent = []
    to_keep = []
    if hasattr(clippers, "IFT"):
        clippers = clippers.outer_rings(False)
    elif not isinstance(clippers, (list, tuple)):
        clippers = [clippers]
    if hasattr(polys, "IFT"):
        polys = polys.outer_rings(False)
    elif not isinstance(polys, (list, tuple)):
        polys = [polys]
    for i, clp in enumerate(clippers):
        for j, poly in enumerate(polys):
            shared, extent = common_extent(poly, clp)
            shared_extent.append([i, j, extent])
            if shared:
                in_ = clip(poly, clp)
                if in_ is not None:
                    if len(in_) > 3:
                        to_keep.append(in_)
    return to_keep, shared_extent


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))

    z0 = r"C:\Arc_projects\Test_29\Test_29.gdb\clp"
    z1 = r"C:\Arc_projects\Test_29\Test_29.gdb\poly"
