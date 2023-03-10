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
# import math
# from textwrap import dedent
from functools import reduce
import numpy as np

# -- optional numpy imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

import npg  # noqa
from npg.npGeo import roll_arrays
from npg.npg_geom import common_extent
from npg.npg_helpers import a_eq_b, del_seq_dups  # uniq_1d
from npg import npg_plots  # noqa
from npg.npg_plots import plot_polygons  # noqa
# from npg.npg_utils import time_deco

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=300,
    formatter=ft
)


script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['clip', 'roll_arrays', 'split_seq']
__helpers__ = ['_concat_']


# ---- (1) general helpers

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
    `The denominator of this expression is the distance between P1 and P2.
    The numerator is twice the area of the triangle with its vertices at the
    three points, (x0, y0), p1 and p2.` Wikipedia
    With p1, p2 defining a line and x0,y0 a point.

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
        a_num = (a_0 - a_1) + 0.0  # signed distance diff_ in npg.pip.wn_np
        b_num = (b_0 - b_1) + 0.0
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
            denom = (x1_x0 * y3_y2) - (y1_y0 * x3_x2)  # denom of determinant
            return wn_, denom, x0, y0, x1_x0, y1_y0, a_num, b_num
        return wn_

    def _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0):
        """Return the intersection."""
        with np.errstate(all="ignore"):  # ignore all errors
            u_a = (a_num / denom) + 0.0
            u_b = (b_num / denom) + 0.0
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
    # --
    # Use `_w_` and `_xsect_` to determine poly pnts in pnts (as polygon)
    wn2_ = _w_(poly, pnts, False)
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


def _before_chk_(poly_, seen_, id_, a_in_b):
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
        pnts = [poly_[i] for i in bf]
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


def _f0f1_(_f0, _f1, _last, _st_en, _in, _eq, _seen):
    """Pick `from-to` points for various functions for the first clip.

    Notes
    -----
    The following are examples using clip information::

        bf_c = _f0f1_(c0_fr, c1_fr, c_last, c_st_en, c_in_p, c_eq_x, c_seen)
        bf_p = _f0f1_(p0_fr, p1_fr, p_last, p_st_en, p_in_c, p_eq_x, p_seen)
    """
    if _f0 == 0:
        _f0, _f1 = _last, _st_en
    elif _f0 != 0 and _f0 < _f1:
        _f0, _f1 = 0, _f0
    tmp = _btw_chk_(_f0, _f1, _in, _seen)
    return [i for i in tmp if i not in _eq]


def _to_add_(XCsub, sub_, ply_seen, clp_seen, tot_):
    """Simplify updating ply_seen, clp_seen and tot_ arrays."""
    c_0, c_1, p_0, p_1 = XCsub[:4]
    clp_seen = list(set(clp_seen).union([c_0, c_1]))
    ply_seen = list(set(ply_seen).union([p_0, p_1]))
    tot_.append([XCsub.ravel(), sub_])
    return ply_seen, clp_seen, tot_


def node_type(p_in_c, c_in_p, poly, clp, x_pnts):
    """Return node intersection data. `_cpx_ ,clipper polygon intersection`.

    Parameters
    ----------
    p_in_c, c_in_p : lists
        Id values of points in poly and clipper respectively.
    clp, poly : array_like
        The geometry of the clipper and the polygon being clipped.
    x_pnts : array_like
        The intersection points of the geometry edges.

    Requires
    --------
    `from functools import reduce` as import

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
    # -- checks
    # poly/clp, poly/x_pnts, clp/x_pnts equalities
    c_eq_p, p_eq_c = np.nonzero((poly == clp[:, None]).all(-1))
    p_eq_x, _ = np.nonzero((x_pnts == poly[:, None]).all(-1))
    c_eq_x, _ = np.nonzero((x_pnts == clp[:, None]).all(-1))
    # -- check  equality
    c_eq_p = sorted(list(set(c_eq_p))) if len(c_eq_p) > 0 else []
    p_eq_c = sorted(list(set(p_eq_c))) if len(p_eq_c) > 0 else []
    p_eq_x = sorted(list(set(p_eq_x))) if len(p_eq_x) > 0 else []
    c_eq_x = sorted(list(set(c_eq_x))) if len(c_eq_x) > 0 else []
    #
    # -- build the output
    p_in_c = list(set(p_in_c))
    c_in_p = list(set(c_in_p))
    if p_eq_c or p_eq_x:  # -- non-empty lists check
        #  p_in_c = reduce(np.union1d, [p_in_c, p_eq_c, p_eq_x])  # slow equiv.
        p_in_c = sorted(list(set(p_in_c + p_eq_c + p_eq_x)))
    if c_eq_p or c_eq_x:  # c_in_p + (p_eq_c, c_eq_x)
        c_in_p = sorted(list(set(c_in_p + c_eq_p + c_eq_x)))
    return p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x


def _x_mkr_(whr_on, x_pnts_on, p_in_c, c_in_p):  # poly, p_eq_x, c_eq_x):
    """Return intersections/crossings and checks, `xChk`, given inputs.

    whr_on : array where intersections occur
    x_pnts_on : array of xy intersection values
    p_in_c, c_in_p : polygon points in clipper and clipper points in polygon.
        `in` means that the points are inside, or on the boundary.
    """
    s, r = divmod(whr_on.shape[0], 2)  # check for even pairing
    if r == 0:
        x = (whr_on.reshape(-1, 4)).copy()
        x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
    else:
        x = (whr_on[:s * 2].reshape(-1, 4)).copy()
        x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
        lastx, lasty = whr_on[s * 2:][0]
        x = np.concatenate((x, np.array([[lastx, -1, lasty, -1]])), axis=0)
    crossings = x.copy()
    z0 = np.isin(crossings[:, :2], c_in_p)
    z1 = np.isin(crossings[:, 2:4], p_in_c)
    #
    in_chks = np.concatenate((z0, z1), axis=1)
    xCheck = np.concatenate((crossings, in_chks.astype(int)), axis=1)
    # -- intersection points
    x0x1 = [x_pnts_on[i: i + 2] for i in range(0, len(whr_on), 2)]
    if x0x1[-1].shape == (1, 2):
        pad = np.concatenate((x0x1[-1], np.array([[np.nan, np.nan]])), axis=0)
        x0x1[-1] = pad
    return xCheck, x0x1


def _split_whr_(arr, aslist=False):  # sec_last
    """Return sub-arrays where `arr` is split at sequential differences.

    Parameters
    ----------
    arr : array_like
        A sorted ndarray is expected, but will be converted if needed.  Sorting
        is your task.

    Returns
    -------
    A list of sub-arrays or the original sequence if there are no missing
    values in the sequence.
    """
    sub = np.copy(arr)
    if arr.ndim > 1:
        sub = arr[:, 0]
    idx_ = np.nonzero(sub[:-1] != sub[1:])[0] + 1
    vals = np.array_split(arr, idx_)
    if aslist:
        return [s.tolist() for s in vals], idx_
    return vals, idx_


def split_seq(seq, p_last):  # sec_last
    """Return a sequence of point ids split at its numeric gaps.

    The defined gap is 1, since we are working with sequences of points.

    Parameters
    ----------
    seq : array_like
        A sorted ndarray is expected, but it will sorted anyway.
    p_last : integer
        Index of the second last point if the points are from a polygon since
        the first and last points are identical.

    Returns
    -------
    A list of sub-arrays or the original sequence if there are no missing
    values in the sequence.
    """
    N = 0
    if len(seq) == 0:
        return N, []
    if isinstance(seq, (list, tuple)):
        seq = np.asarray(seq)
    #
    # -- sort the sequence
    seq.sort()
    if seq.ndim > 1 or seq.shape[0] <= 1:
        return seq.size, seq.tolist()
    if seq[0] == 0 and p_last == seq[-1]:  # piece first to the end if present
        seq = seq[:-1]
        tmp = np.concatenate((seq[1:], [p_last + 1]))
        whr = np.nonzero(np.abs(tmp[1:] - tmp[:-1]) != 1)[0]
        if whr.size > 0:  # check for other splits
            z = [s.tolist() for s in np.array_split(tmp, whr + 1)]
            lst = z.pop()
            z.insert(0, lst)
            N = len(z)
            return N, z
        return N, tmp.tolist()  # move first to end
    whr = np.nonzero(np.abs(seq[1:] - seq[:-1]) != 1)[0]
    if whr.size > 0:
        N = whr.size + 1
        return N, [s.tolist() for s in np.array_split(seq, whr + 1)]
    return N, seq.tolist()


def _sort_on_line_(ln_pnts, cross_pnts):
    """Order intersection points on a straight line, from the start.

    Parameters
    ----------
    ln_pnts, cross_pnts : array-like
        Two point line and its intersection points.  These are Nx2 arrays with
        at least two intersection points.

    Notes
    -----
    If the points are on a vertical line, then sort on the y-values.  The order
    of the line points is not altered, but the intersecton points are arranged
    in sequential order on the line even if x or y is ascending and/or
    descending.

    """
    p = np.concatenate((ln_pnts, cross_pnts), axis=0)
    dxdy = np.abs(p[0] - p[1:])
    order = np.argsort(dxdy[:, 0])
    if dxdy.sum(axis=0)[0] == 0:  # -- vertical line check
        order = np.argsort(dxdy[:, 1])  # sort ascending on y-values
    tmp = p[1:][order]
    p[1:] = tmp
    return p  # uniq_2d(tmp), order


def _order_pnts_(whr_on, x_pnts_on, poly, clp):
    """Return intersection points ordered on the intersection lines.

    Parameters
    ----------
    whr_on : array
        The array showing the segments that intersection points occur on.
    x_pnts_on : array
        The intersection points formed from the segment intersections.
    poly, clp : arrays
        The polygon being clipped and the clipping polygon

    Example
    -------
    polygons p02 and c02 are used as an example::

    whr_on             x_pnts_on
    array([[0, 1],     array([[  1.75,   1.75],  clp 0 crosses polys 1 and 2
           [0, 2],            [  2.00,   2.00],
           [1, 1],            [  3.50,   1.50],  clp 1 crosses polys 1 and 2
           [1, 2],            [  3.33,   2.00],
           [2, 2],            [  3.67,   2.00],  clp 2 crosses poly 2
           [3, 1],            [  5.83,   1.17],  clp 3 crosses polys 1 and 2
           [3, 2]],           [  5.00,   2.00]])

    correct order
    array([[  1.75,   1.75],
           [  2.00,   2.00],
           [  3.33,   2.00],
           [  3.50,   1.50],
           [  3.67,   2.00],
           [  5.00,   2.00],
           [  5.83,   1.17]])
    """
    ft_c = np.concatenate((clp[:-1], clp[1:]), axis=1)    # clp from-to pnts
    vals, split_idx_ = _split_whr_(whr_on)                # split whr_on
    crossings = np.array_split(x_pnts_on, split_idx_)  # split the crossings
    # -- sort by distance
    out_ = []
    for i, cr in enumerate(crossings):  # -- if more that 1 crossing point
        if len(cr) > 1:
            z0 = ft_c[i, :2]  # first point
            sub = [_dist_(z0, c) for c in cr]
            new_idx = np.argsort(sub)
            new_pnts = cr[new_idx]
            out_.extend(new_pnts)
        else:
            out_.extend(cr)
    return np.asarray(out_)


# -- main
def clip(poly, clp):
    """Return the result of a polygon clip.

    Parameters
    ----------
    poly, clp : ndarrays representing polygons
        Rings are oriented clockwise.  Holes are ignored.

    Requires
    --------
    -  `_wn_clip_`, `node_type`, `_x_mkr_`
    - `_concat_`, `_dist_`, `a_eq_b`, `_before_chk_`, `_btw_chk_`, `_to_add_`

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
    Notes here.

    """
    # --
    if hasattr(poly, "IFT"):
        poly = poly.XY
    if hasattr(clp, "IFT"):
        clp = clp.XY
    #
    # -- array basic information
    poly, clp = roll_arrays([poly, clp])  # roll the arrays to orient to LL
    #
    # -- quick bail 1
    # bail = a_eq_b(poly, clp).all()
    # if bail:
    #    print("\nInput polygons are equal.\n")
    #    return  # -- uncomment when done
    nC = clp.shape[0]
    nP = poly.shape[0]
    c_last, c_st_en = nC - 2, nC - 1     # -- clp  : last, dupl. start-end pnt
    p_last, p_st_en = nP - 2, nP - 1     # -- poly : last, dupl. start-end pnt
    #
    # -- use `winding number` to get points inside, on, outside each other
    vals = _wn_clip_(poly, clp, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals  # wn_
    #
    # -- quick bail 2
    # if len(x_pnts) == 0:
    #     print("No intersection between `poly` and `clp`.")
    #     return  # -- uncomment when done
    #
    # -- derive the unique intersection points and get their first found index.
    uni, idx, cnts = np.unique(
        x_pnts, return_index=True, return_counts=True, axis=0)
    if (cnts > 1).any():  # print("\nDuplicate intersection points.\n")
        idx_srt = np.sort(idx)       # get the original order
        x_pnts_on = x_pnts[idx_srt]
        whr_on = whr[idx_srt]        # whr crossings in that order
    else:
        x_pnts_on = x_pnts
        whr_on = whr
    #
    # -- run `node_type` to get the nodes, and their info
    # NOTE:  c_in_p : includes c_in_p, c_eq_p, c_eq_x
    #        p_in_c : includes p_in_c, p_eq_c, p_eq_x
    args = node_type(pInc, cInp, poly, clp, x_pnts)
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    #
    # -- quick bail 3
    # if len(p_in_c) == poly.shape[0]:
    #    print("\nAll `poly` points are within the `clp` polygon.\n")
    # if len(c_in_p) == clp.shape[0]:
    #    print("\nAll `clp` points are within the `poly` polygon.\n")
    #     return  # -- uncomment when done
    # -- `_order_pnts_` : order intersection points on clip lines
    #    `_x_mkr_`      : produce `xChk` array and intersection pairs
    # ** doesn't work in all cases, check order as well later
    x_pnts_on = _order_pnts_(whr_on, x_pnts_on, poly, clp)
    xChk, x0x1 = _x_mkr_(whr_on, x_pnts_on, p_in_c, c_in_p)  # poly
    #
    dC_dP = np.abs(np.vstack((xChk[:, 1] - xChk[:, 0],
                              xChk[:, 3] - xChk[:, 2]))).T
    c_prev, p_prev = [-1, -1]  # -- define previous clip/poly segments
    p_seen, c_seen, tot_ = [[], [], []]  # create empty lists
    Np_seq, p_seq, Nc_seq, c_seq = [0, [], 0, []]
    #
    # -- determine the sequences of poly in clp and clp in poly.  The sequences
    #    are split at breaks and if there are points `in` the other, then
    #    they should equal p
    if p_in_c:  # at least 1 poly pnt in clp
        Np_seq, p_seq = split_seq(p_in_c, p_last)  # p_last  ??? or p_st_en
    if c_in_p:  # at least 1 clp pnt in poly
        Nc_seq, c_seq = split_seq(c_in_p, c_last)
    # --
    fr_col = xChk[:, 2]  # check for multiple crossings using poly-from col
    #
    for cnt, xC in enumerate(xChk):  # xChk is the crossings list
        c0_fr, c1_fr, p0_fr, p1_fr = xC[:4]
        c0_to, c1_to, p0_to, p1_to = xC[:4] + 1
        c0_in, c1_in, p0_in, p1_in = np.asarray(xC[-4:], bool)
        in_case = "{} {} {} {}".format(*np.asarray(xC[-4:]))
        x0, x1 = x0x1[cnt]           # -- segment pair intersection points
        diff_C, diff_P = dC_dP[cnt]  # difference in indices
        bf_c, btw_c = [], []         # Initial values for `before`, `between`
        bf_p, btw_p = [], []         # for both poly and clp
        c0c1_ = in_case[:3]
        p0p1_ = in_case[4:]
        sub_ = []
        #
        seq_p = [] if cnt + 1 > Np_seq else p_seq[cnt]  # poly pnt sequences
        # seq_c = [] if cnt + 1 > Nc_seq else c_seq[cnt]  # clp pnt sequences
        #
        # -- multi-crossing check along poly segments
        multi_cross = np.count_nonzero(fr_col == p0_fr)  # count p0_fr vals
        # --
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
            # print(f"{cnt} {x0}, {x1}  xC {xC}")
            # --
            break  # break out
        #
        # -- pre section
        # -- Either `bf_c` or `bf_p` will have values, sometimes both if
        #    point 0 for both are intersection points.
        if cnt == 0:
            # -- before_c `bf_c`
            if c0_in or diff_C == 0:
                if c0_fr == 0:  # -- last and first must be in c_in_p
                    if c0_fr in c_in_p and c_last in c_in_p:
                        bf_c = [c_last, 0]
                else:
                    bf_c = _f0f1_(c0_fr, c1_fr, c_last, c_st_en,
                                  c_in_p, c_eq_x, c_seen)
            if bf_c:  # sometimes the last clip point before clip point 0
                sub_.extend(clp[bf_c])  # add the points before point 0
                c_seen.extend(bf_c)
            # -- before_p `bf_p`
            if p0_in or diff_P == 0:
                if multi_cross > 1:  # -- multiple crossing check
                    bf_p = []
                elif p0_fr == 0 and p_last in p_in_c:
                    bf_p = [p_last, 0]
                else:  # -- could be a duplicate start/end point eg. 0, 4
                    bf_p = _f0f1_(p0_fr, p1_fr, p_last, p_st_en,
                                  p_in_c, p_eq_x, p_seen)
            if bf_p:  # repeat for poly points before point 0
                sub_.extend(poly[bf_p])  # ** had to fix above Mar 4 2022
                p_seen.extend(bf_p)
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
                # -- quirk last check for final crossing line
                if bf_p:
                    if bf_p[-1] + 1 == p_last:
                        bf_p.append(p_last)
                    bf_p = [i for i in bf_p if i not in p_eq_x]
                # -- include any other poly pnts inside
                if bf_p:
                    sub_.extend(poly[bf_p])
                    p_seen.extend(bf_p)
            # --
        # -- end `before`
        # -- begin `between` section
        if diff_C != 0:
            btw_c = _btw_chk_(c0_fr, c1_fr, c_in_p, c_seen)
            if btw_c:
                if c1_fr - c0_fr == 1:  # differ by 1 and both inside
                    c_seen.extend(btw_c)
                elif c1_fr - c0_fr == 2:  # was == 2
                    btw_c = [i for i in btw_c if i in c_in_p]  # & i in c_eq_x]
                else:
                    btw_c = [i for i in btw_c if i in c_in_p
                             and i not in c_seen]  # c_eq_x]
                c_seen.extend(btw_c)  # should this be indented or remove if..
        if diff_P != 0:  # E, b0 : first clip line intersects E lines 0 and 3
            btw_p = _btw_chk_(p0_fr, p1_fr, p_in_c, p_seen)
            if btw_p:
                if p1_fr - p0_fr == 1:  # differ by 1 and both inside
                    p_seen.extend(btw_p)
                elif p1_fr - p0_fr == 2:
                    btw_p = [i for i in btw_p if i in p_in_c]  # & i in p_eq_x]
                else:
                    btw_p = [i for i in btw_p if i in p_in_c
                             and i not in p_seen]  # p_eq_x]
                    if seq_p:  # -- K, A example
                        btw_p = [i for i in btw_p if i in seq_p]
                p_seen.extend(btw_p)
        print(f"{cnt} x0,x1  {x0}, {x1}\n  xC     {xC}")
        print(f"  c0c1_, p0p1_:   {c0c1_} {p0p1_}")
        print(f"  bfc:bfp {bf_c}  {bf_p}\n  btc:btp {btw_c}  {btw_p}")
        print(f"  sub :  {sub_}\n")
        #
        # -------- '0 0' prep --------------------------------------------
        if c0c1_ == '0 0':
            # -- Clip points are out and cross poly.  Poly pnts may be inside.
            #  Order points on clipping line.
            x0, x1 = x0x1[cnt]
            if diff_C != 0:
                if c0_fr + 1 in c_in_p:
                    sub_.append(clp[c0_fr + 1])
            else:
                ln_pnts = clp[[c0_fr, c0_to]]
                _, x0, x1, _ = _sort_on_line_(ln_pnts, [x0, x1])
            # -----------------------------------
            # -- poly checks
            # --
            if p0p1_ == '0 0':                           # 0,  0,  0,  0
                if p0_fr == 0:  # check when p0==0 and change its value
                    if p1_to == p_st_en:  # see if you are back at the start
                        sub_.extend([x1, x0])
                    else:
                        sub_.extend([x0, x1])
                else:  # -- distance check, needed p02, c01
                    sub_.extend([x0, x1])
                if p1_to in p_in_c and cnt != 0:
                    sub_.extend([poly[p1_to]])
                if c1_to in c_in_p:
                    sub_.extend([clp[c1_to]])  # c1,b3 add last clp
            # --
            elif p0p1_ == '0 1':                           # 0,  0,  0,  1
                if p0_fr < p1_fr:  # -- this isn't always the case
                    sub_.extend([x0, x1])
                else:
                    sub_.extend([x1, x0])
                if not sub_:  # -- last ditch effort
                    sub_.extend([x0, x1])
            # --
            elif p0p1_ == '1 0':                           # 0,  0,  1,  0
                if (p0_fr < p1_fr) or (p1_fr in p_in_c):
                    if btw_p:  # may have been set to [] when bf_p btw_p equal
                        sub_.extend(poly[btw_p])
                    sub_.extend([x0, x1])
                else:
                    sub_.extend([x1, x0])
            # --
            elif p0p1_ == '1 1':                           # 0,  0,  1,  1
                # x0,x1 on the clipping line, and is sorted
                sub_.extend([x0, x1])  # or btw_p, s00,t0
        #
        # -------- '0 1' prep --------------------------------------------
        elif c0c1_ == '0 1':  # example b0,c1 b1,c0 A,B
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
            # --
            elif p0p1_ == '0 1':  # p1_seen:  # not done  # 0,  1,  0,  1
                sub_.extend([poly[p1_fr]])
            # --
            elif p0p1_ == '1 0':   # example b0,c1         # 0,  1,  1,  0
                if p0_fr < p1_fr:  # b4,c1  p1_to is > p0_fr & p1_fr
                    if p0_to not in btw_p and p1_to not in btw_p:  # edgy1,ecl
                        if btw_p:
                            sub_.extend(poly[btw_p])
                    sub_.append(x1)
                else:
                    sub_.append(x1)
                    if btw_p and p1_to not in btw_p:
                        sub_.extend(poly[btw_p])
                if p1_to in p_in_c and p1_to not in p_seen + p_eq_x:  # c2,K
                    if len(seq_p) <= 1:
                        sub_.append(poly[p1_to])
                    elif len(seq_p) > 1:  # b4, c1
                        sub_.extend(poly[seq_p])
                    p_seen.extend([p1_to])
            # --
            elif p0p1_ == '1 1':  # 2 clp on same poly     # 0,  1,  1,  1
                if c1_fr in c_in_p and c1_fr not in c_seen:
                    sub_.extend([clp[c1_fr]])
                sub_.append(x1)
                if diff_P == 0 and p0_to in c_in_p:  # diff_P == 0
                    sub_.extend([poly[p0_to]])
        #
        # -------- '1 0' prep --------------------------------------------
        elif c0c1_ == '1 0':  # c0, b0  header is the reverse of '0 1'
            #
            if c0_fr in c_seen and cnt != 0:  # check for last clipping line
                if c0_fr in c_eq_x and c1_to == c_st_en:
                    break  # last clipping line,
            if c0_fr in btw_c:  # added for c1, b5 and C, D
                sub_.extend(clp[btw_c])
                sub_.append(x0)
            elif c0_fr <= c1_fr:
                sub_.append(x0)
                if btw_c:
                    sub_.extend(clp[btw_c])
            elif c0_fr > c1_fr:
                if btw_c:
                    sub_.extend(clp[btw_c])
                # sub_.append(x1)  # 2023-02-28
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
                    sub_.append(x1)
                    tmp = np.array(sub_)
                    un, idx = np.unique(tmp, return_index=True, axis=0)
                    sub_ = tmp[idx]
                else:
                    sub_.append(x1)  # ** changed to try and fix edgy1/eclip
                if p1_to in p_in_c and p1_to in p_eq_x:
                    p_seen.extend([p1_to])
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
                # -- close
                if p1_to in p_in_c and p1_to not in p_eq_x:
                    sub_.extend([poly[p1_to]])
                if (c1_to in c_in_p) and (c1_to not in c_eq_x):
                    sub_.extend([clp[c1_to]])
                # finally
                sub_.append(x1)  # -- or add here in case the above are before
        #
        # ---------- '1 1' prep ------------------------------------------
        elif c0c1_ == '1 1':                               # 1,  1,  ?,  ?
            # -- begin checks  * this is a mess!!!
            #
            if diff_C == 0 or not btw_c:  # check for clip line reordered
                ln_pnts = clp[[c0_fr, c0_to]]
                cross_pnts = x0x1[cnt]
                _, x0, x1, _ = _sort_on_line_(ln_pnts, cross_pnts)
                #
                if p0_fr in p_in_c:  # -- may be before clip point like in
                    sub_.append(poly[p0_fr])  # pl, cl step 3
                sub_.append(x0)
                p_seen.append(p0_fr)
            elif (diff_C == 1) and (c0_to in c_eq_x):  # added E,b0 second line
                sub_.append(x0)  # probably c0_to is an intersection point
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
            elif c0_to in c_in_p and c0_to not in c_eq_x:
                sub_.extend([clp[c0_to]])
                c_seen.extend([c0_fr, c0_to])
            #
            # -----------------------------------
            # -- poly checks
            # --
            if p0p1_ == '0 0':                             # 1,  1,  0,  0
                # -- a,C leaves 3 points dangling and were included in c0c1_
                if p0_fr not in p_in_c or p0_fr not in p_eq_x:
                    sub_.extend([x1])
                elif p0_fr not in p_in_c and p0_fr == p_last:
                    sub_ = []
                    break  # "unsupported segment, reset sub_ for 1 1 0 0"
            # --
            elif p0p1_ == '0 1':                           # 1,  1,  0,  1
                if p0_fr <= p1_fr:
                    if p1_fr in p_in_c:
                        sub_.append(poly[p1_fr])  # E b0 2nd loop
                    to_add = [i for i in btw_p if i not in p_seen]
                    if to_add:  # btw_p could be 1 or more points
                        sub_.extend(poly[btw_p])
                        p_seen.extend(to_add)
                    sub_.append(x1)  # add x1, not other points in between
                else:
                    sub_.append(x1)
            # --
            elif p0p1_ == '1 0':                           # 1,  1,  1,  0
                # add points between p0_fr and p1_fr
                if p0_fr in p_in_c and p0_fr not in p_seen:
                    sub_.append(poly[p0_fr])
                if btw_p or diff_P > 1:
                    sub_.append(x1)
                    to_add = [i for i in btw_p if i in p_in_c
                              and i not in p_eq_x]
                    if to_add:
                        sub_.extend(poly[to_add])  # -- had issues before
                        p_seen.extend(to_add)
                    btw_p = []  # empty btw_p
                # add the second intersection
                else:
                    sub_.append(x1)
                # -- now check the end point/next start point for inclusion
                #    but btw_p must be empty
                if p1_to not in p_eq_x and p1_to in p_in_c:
                    sub_.extend([poly[p1_to]])
                p_seen.extend([p1_to])  # **** added for edgy1 3rd clip
            # --
            elif p0p1_ == '1 1':                           # 1,  1,  1,  1
                if p0_fr in p_in_c:
                    sub_.append(poly[p0_fr])
                if btw_p and p0_fr != 0:  # E,b0 to prevent 0,3 adding 2,3
                    sub_.extend(poly[btw_p])  # add all the between points
                    p_seen.extend(btw_p)  # ??? keep second part of E, b0
                # check to make sure that p0_fr != x1
                if (poly[p1_fr] != x1).any():  # -- not equal to x1
                    sub_.append(x1)
                # check to make sure that p1_to != x1
                if p1_to in p_in_c and not (poly[p1_to] == x1).any():
                    sub_.extend([poly[p1_to]])
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
    #
    # -- clean up the outputs
    if (new_[0] != new_[-1]).any():
        new_ = np.concatenate((new_, new_[0][None, :]), axis=0)
    new_ = del_seq_dups(new_, poly=True)
    return new_, tot_, xChk, x0x1, p_in_c, c_in_p
    # new_, tot_, xChk, x0x1, p_in_c, c_in_p = clip(
    # plot_polygons([s00, t00, t01, t02, t03])


def assemble(poly, clp):
    """Assemble intersection points on two intersecting polygons.

    Parameters
    ----------
    poly, clp : array_like
        `poly` is the polygon being clipped by `clp`.  Coordinates are Nx2
        shaped.

    Requires
    --------
    `roll_arrays`, `_wn_clip_`
    """
    def _srt_pnts_(p):
        """Order intersection points on a line, from the start.

        `_sort_on_line_` is the full version.
        """
        if len(p) == 2:  # -- only start and end point
            return p
        dxdy = np.abs(p[0] - p[1:])
        if dxdy.sum(axis=0)[0] == 0:  # -- vertical line check
            order = np.argsort(dxdy[:, 1])  # sort ascending on y-values
        else:
            order = np.argsort(dxdy[:, 0])
        p[1:] = p[1:][order]
        return p

    def _chk_(poly, clp, x_pnts, whr):
        """Return input arrays with intersections added to their lines."""
        p_ = np.concatenate((poly[:-1], poly[1:]), axis=1).reshape(-1, 2, 2)
        p_ = list(p_)  # [i for i in p_]
        c_ = np.concatenate((clp[:-1], clp[1:]), axis=1).reshape(-1, 2, 2)
        c_ = list(c_)  # [i for i in c_]
        for cnt, cp in enumerate(whr):
            cl, pl = cp
            x = x_pnts[cnt][None, :]
            c_[cl] = np.concatenate((c_[cl], x), axis=0)
            p_[pl] = np.concatenate((p_[pl], x), axis=0)
        # --
        for cnt, p in enumerate(p_):
            if len(p) > 2:
                p_[cnt] = _srt_pnts_(p)
        for cnt, c in enumerate(c_):
            if len(c) > 2:
                c_[cnt] = _srt_pnts_(c)
        return p_, c_

    def _chk2_(poly, clp, x_pnts, whr):
        """Variant of above. but it isn't faster, maybe.

        https://stackoverflow.com/questions/53423924/why-is-python-
        in-much-faster-than-np-isin
        """
        p_ = np.concatenate((poly[:-1], poly[1:]), axis=1).reshape(-1, 2, 2)
        p_ = list(p_)  # [i for i in p_]
        c_ = np.concatenate((clp[:-1], clp[1:]), axis=1).reshape(-1, 2, 2)
        c_ = list(c_)  # [i for i in c_]
        for cnt, cp in enumerate(whr):
            cl, pl = cp
            x = x_pnts[cnt]
            chk0 = (x == c_[cl]).all(-1).any(-1)  # correct but slow
            chk1 = (x == p_[pl]).all(-1).any(-1)  # correct but slow
            if not chk0:
                c_[cl] = np.concatenate((c_[cl], x[None, :]), axis=0)
            if not chk1:
                p_[pl] = np.concatenate((p_[pl], x[None, :]), axis=0)
        for cnt, p in enumerate(p_):
            if len(p) > 2:
                p_[cnt] = _srt_pnts_(p)
        for cnt, c in enumerate(c_):
            if len(c) > 2:
                c_[cnt] = _srt_pnts_(c)
        return p_, c_

    # -- main
    if hasattr(poly, "IFT"):
        poly = poly.XY
    if hasattr(clp, "IFT"):
        clp = clp.XY
    # -- roll towards LL.  `_wn_clp_` gets pnts inside, on, outside each other
    poly, clp = roll_arrays([poly, clp])
    vals = _wn_clip_(poly, clp, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals
    # -- remove duplicate start/end index
    args = node_type(pInc, cInp, poly, clp, x_pnts)
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    # -- run _chk_ and del_seq_dups
    pl_new, cl_new = _chk_(poly, clp, x_pnts, whr)
    pl_new = del_seq_dups(np.concatenate((pl_new), axis=0), poly=True)
    cl_new = del_seq_dups(np.concatenate((cl_new), axis=0), poly=True)
    # ----
    # Below is the index creation, commented out for now
    # --
    # --build the index array, ids, clp, poly is that order
    # max_shp = np.max([i[0] for i in
    #                   [poly.shape, pl_new.shape, clp.shape, cl_new.shape]])
    # idxs = np.full((max_shp, 7), fill_value=-1, dtype='int')
    # idxs[:, 0] = np.arange(0, max_shp, 1)
    # clpl_eq = np.argwhere((pl_new == cl_new[:, None]).all(-1))
    # # -- add where they are equal, to determine which are in or out
    # i0, i1 = clpl_eq.T
    # idxs[i0, 1:3] = clpl_eq  # -- idxs array now includes ids of intersections
    # # --
    # # -- construct the columns and find and extract indices
    # # headers : ids, clpl_eq, plcl_eq, c_in_p_new, p_in_c_new, wc0, wc1
    # cl_oldnew = np.argwhere((cl_new == clp[:, None]).all(-1))
    # if c_in_p:
    #     c_in_p_new = np.in1d(cl_oldnew[:, 0], c_in_p).nonzero()[0]
    # idxs[c_in_p_new, 3] = c_in_p_new
    # whr0 = idxs[:, 1][idxs[:, 1] != -1]
    # whr0a = idxs[:, 3][idxs[:, 3] != -1]
    # wc0 = sorted(list(set(np.concatenate((whr0, whr0a)))))
    # # -- these have to be sorted
    # pl_oldnew = np.argwhere((pl_new == poly[:, None]).all(-1))
    # p_in_c_new = np.in1d(pl_oldnew[:, 0], p_in_c).nonzero()[0]
    # idxs[p_in_c_new, 4] = p_in_c_new
    # whr1 = idxs[:, 2][idxs[:, 2] != -1]   # eg. np.where(idxs[:, 2] != -1)[0]
    # whr1a = idxs[:, 4][idxs[:, 4] != -1]  # eg. np.where(idxs[:, 4] != -1)[0]
    # wc1 = sorted(list(set(np.concatenate((whr1, whr1a)))))
    # # -- ready to piece together
    # idxs[wc0, 5] = wc0  # -- clp in/on
    # idxs[wc1, 6] = wc1  # -- poly in/on
    # cl_final = cl_new[wc0]  # -- not done yet, but close need to add p_in_c
    # pl_final = pl_new[wc1]
    #
    # new_shared = np.argwhere((pl_new == cl_new[:, None]).all(-1))
    # final_shared = np.argwhere((pl_final == cl_final[:, None]).all(-1))
    # diffs = final_shared[1:] - final_shared[:-1]  # -- sequential differences
    # rows = final_shared.tolist()
    # out = [rows[0]]
    # for cnt, row in enumerate(rows[1:]):
    #     prev = rows[cnt]
    #     d0, d1 = diffs[cnt]
    #     if d0 == 1 and (d1 == 1):  # or cnt == 0):
    #         vals = [row]
    #     elif d0 > 1 and d1 == 1:
    #         vals = [[i, -9] for i in range(prev[0] + 1, row[0] + 1)]
    #     elif d0 == 1 and d1 > 1:
    #         vals = [[-9, i] for i in range(prev[1] + 1, row[1] + 1)]
    #     out.extend(vals)
    #     # print("{} : {} {}, {} {}: {}".format(cnt, prev, row, d0, d1, vals))
    # ps = []
    # for i in out:
    #     if i[0] != -9:
    #         ps.append(cl_final[i[0]])
    #     else:
    #         ps.append(pl_final[i[1]])
    # ps = np.asarray(ps)
    # splts = np.array_split(new_shared, brks0 + 1, 0)
    return cl_new, pl_new  # cl_final, pl_final, ps


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
            extent = common_extent(poly, clp)  # either an extent or `None`
            shared_extent.append([i, j, extent])
            if extent:
                in_ = clip(poly, clp)
                if in_ is not None:
                    if len(in_) > 3:
                        to_keep.append(in_)
    return to_keep, shared_extent


# ---- Extras
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
    `_wn_clip_`, `node_type` are used to determine whether each point meets the
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


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))

    # z0 = r"C:\Arc_projects\Test_29\Test_29.gdb\clp"
    # z1 = r"C:\Arc_projects\Test_29\Test_29.gdb\poly"
