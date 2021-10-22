# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E1101,E1121
# pylint: disable=F401
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0621
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# noqa: D205, D400, F401, F403, C0103
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
    2021-10_21

Purpose
-------
Functions for clipping polygons.

Call sequence::

    # _clp_ and _wn_clip_ can be used without poly_clip
    # inouton and _wn_clip have no other dependencies.
    poly_clip
    ...   |__  _clp_
    ...        |  _wn_clip_
    ...

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
import math
# from textwrap import dedent
import numpy as np

# -- optional numpy imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

from npg import npg_plots
from npg.npg_plots import plot_polygons

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['common_extent', 'uniq_1d', 'uniq_2d', 'poly_clip']
__helpers__ = ['_onseg_', '_is_on_', '_side_', '_cut_', '_wn_clip_']


# ----------------------------------------------------------------------------
# ---- (1) helpers
#
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


def uniq_1d(arr):  # *** remove when publishing clip
    """Return mini 1D unique for sorted values."""
    mask = np.empty(arr.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = arr[1:] != arr[:-1]
    return arr[mask]


def uniq_2d(arr, return_sorted=False):
    """Return mini `unique` for 2D coordinates.  Derived from np.unique."""

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
        ar.sort()
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
    """
    Return structured array for intersections.

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


# ---- (2) standalone functions
#
def _wn_clip_(pnts, poly, all_info=True):
    """Return points in polygon using `winding number`.  See `inouton`.

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
    """
    def _w_(a, b, all_info):
        """Return winding number and other values."""
        x0, y0 = a[:-1].T  # point `from` coordinates
        x1, y1 = a[1:].T   # point `to` coordinates
        x1_x0, y1_y0 = (a[1:] - a[:-1]).T
        #
        x2, y2 = b[:-1].T  # polygon `from` coordinates
        x3, y3 = b[1:].T   # polygon `to` coordinates
        x3_x2, y3_y2 = (b[1:] - b[:-1]).T
        # reshape poly deltas
        x3_x2 = x3_x2[:, None]
        y3_y2 = y3_y2[:, None]
        # deltas between pnts/poly x and y
        x0_x2 = (x0 - x2[:, None])
        y0_y2 = y0 - y2[:, None]
        #
        a_0 = y0_y2 * x3_x2
        a_1 = x0_x2 * y3_y2
        b_0 = x1_x0 * y0_y2
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
            z0 = np.logical_and(u_a >= 0., u_a <= 1.)  # equal to `id_vals`
            z1 = np.logical_and(u_b >= 0., u_b <= 1.)
            both = z0 & z1
            xs = (u_a * x1_x0 + x0)[both]
            ys = (u_a * y1_y0 + y0)[both]
        x_pnts = []
        if xs.size > 0:
            x_pnts = np.concatenate((xs[:, None], ys[:, None]), axis=1)
        # eq = pnts[eq_]
        whr = np.array(np.nonzero(both)).T
        return whr, x_pnts
    #
    # Use `_w_` and `_xsect_` to determine pnts in poly
    wn_, denom, x0, y0, x1_x0, y1_y0, a_num, b_num = _w_(pnts, poly, True)
    whr, x_pnts = _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0)
    #
    p_in_c = np.nonzero(wn_)[0]
    p_out_c = np.nonzero(wn_ + 1)[0]
    p_in = pnts[p_in_c]    # pnts in poly
    p_out = pnts[p_out_c]  # pnts outside poly
    x_type = np.concatenate((wn_[:-1, None], wn_[1:, None]), axis=1)
    #
    wn2_ = _w_(poly, pnts, False)  # get poly points in other geometry
    c_in_p = np.nonzero(wn2_)[0]
    c_out_p = np.nonzero(wn2_ + 1)[0]
    c_in = poly[c_in_p]
    vals = [p_in, p_out, c_in, x_pnts, p_in_c, p_out_c, c_in_p, c_out_p,
            x_type, wn_, whr]
    return vals


# ---- (3) poly_clip
#
def _clp_(poly, splitter):
    """Return the result of a polygon clip.

    - * works for
    - * (b0, b1, ... c0), (b3, b4, b5 ... c1)
    - * (b0, ... c2),
    - * (K, ... c0, c2) and reverse
    - * (d0, b0) and reverse
    - * (E, CC, ... b0)
    - * not for
    - ** plot_polygons([A, B, C, E, K, c2])

    Notes
    -----
    >>> z0 = (splitter[:, None] == poly).any(1).all(1)
    >>> # or (poly == splitter[:, None]).any(1).all(1)  # working
    >>> z1 = (poly[:, None] ==  splitter).all(-1).any(1)
    >>> # or (splitter == poly[:, None]).all(-1).any(1)  # ** works
    >>>   # returns the points that are equal in both.
    >>> # splitter[z0]            poly[z1]
    ... array([[  0.00,   0.00],  array([[  0.00,   0.00],
    ...        [  2.00,   8.00],         [ 10.00,  10.00],
    ...        [ 10.00,  10.00],         [ 10.00,   8.00],
    ...        [ 10.00,   8.00],         [  2.00,   8.00],
    ...        [  0.00,   0.00]])        [  0.00,   0.00]])
    >>> v0, v1 = splitter[0] - poly[1]  # fast check for equality
    >>> is_ = math.isclose(v0, v1)
    >>>
    >>> eq = np.nonzero(np.isclose(A, B).all(1))[0]

    Split the `whr` array where the indices differ by 1

    >>> w = np.where(np.diff(whr[:, 0], prepend=whr[0][0]))[0]
    >>> np.array_split(whr, w)
    """
    def _concat_(x):
        """Concatenate array pieces."""
        z1 = [np.atleast_2d(i) for i in x if len(i) > 0]
        return np.concatenate(z1, axis=0)

    def _dist_(a, b):
        """Return squared distance.  Add math.sqrt() for actual distance.

        Can't really be sped up, even if doing 2 comparisons.
        """
        return (b[0] - a[0])**2 + (b[1] - a[1])**2

    def _in_checks_(c_0, c_1, p_0, p_1):
        """Return point in poly/splitter results."""
        chk_c0 = c_0 in c_in_p  # clipper pnt in poly
        chk_c1 = c_1 in c_in_p  # next clipper pnt in poly
        chk_p0 = p_0 in p_in_c  # 1st poly pnt in clipper
        chk_p1 = p_1 in p_in_c  # next poly pnt in clipper
        return chk_c0, chk_c1, chk_p0, chk_p1

    def _eq_(p_0, c_0):
        """Return whether the splitter and poly points are equal."""
        v0, v1 = splitter[c_0] - poly[p_0]
        return math.isclose(v0, v1)

    # def _spl_eq_x0x1_(p_0, c_0):
    #     """Return whether the splitter and poly points are equal."""
    #     return (splitter[[c_0, c_0 + 1]] == pair_lst[cnt]).all()

    def _a_eq_b_(a, b, atol=1.0e-8, rtol=1.0e-5, return_pnts=False):
        """Return where `a` and `b` points are equal.

        See Also
        --------
        See `npg_helpers.a_eq_b` for details.
        """
        b = b[:, None]
        w = np.less_equal(np.abs(a - b), atol + rtol * abs(b)).all(-1).any(0)
        if return_pnts:
            return a[w]
        return w

    def _before_chk_(ply_seen, p_0):
        """Return the point which are between and within."""
        lst = ply_seen + [p_0]
        bf = [x for x in range(lst[0], lst[-1] + 1) if x not in lst]
        if len(bf) > 0:
            pnts = [poly[i] for i in bf if i in p_in_c]
            if pnts:
                return bf, pnts
        return bf, None

    def _between_chk_(a, b, poly, p_in_c):
        """Return whether any points are between and within."""
        if a < b:
            btw = [x for x in range(a, b + 1) if x not in [a, b]]
        else:
            btw = list(range(a, poly.shape[0])) + list(range(1, b + 1))
        if len(btw) > 0:
            pnts = [poly[i] for i in btw if i in p_in_c]
            if pnts:
                return pnts
        return None

    def _after_chk_(ply_seen, remain):
        """Return whether any points are between and within."""
        brake = (np.diff(remain) - 1).nonzero()[0]
        aft = None
        if len(brake) > 0:
            lst = p_in_c[:brake[0] + 1]
            aft = [x for x in range(lst[0], lst[-1] + 1)]
            remain = remain[brake[0]:]
            if len(aft) > 0:
                pnts = [poly[i] for i in aft]
                if pnts:
                    return remain, aft, pnts
        return remain, aft, None

    def _final_chk_(p_1, last_):
        """Return final check for closing line. last_ = poly.shape[0] -1."""
        lst = [p_1, last_]
        btw = [x for x in range(lst[0], lst[-1] + 1) if x not in lst]
        if len(btw) > 0:
            pnts = [poly[i] for i in btw if i in p_in_c]
            if pnts:
                return pnts
        return None

    vals = _wn_clip_(poly, splitter)
    p_in, p_out, c_in, x_pnts = vals[:4]
    p_in_c, p_out_c, c_in_p, c_out_p, x_type, wn_, whr = vals[-7:]
    remain = np.copy(p_in_c)  # used for special cases like E, c0
    # -- point equality check
    whr_eq_ = np.nonzero(_a_eq_b_(poly, splitter))[0]
    whr_eq2_ = np.nonzero(_a_eq_b_(splitter, poly))[0]
    # -- poly, splitter point equal
    if whr_eq_.size > 0:
        whr_eq_ = set(whr_eq_)
        p_in_c = set(p_in_c)
        p_out_c = set(p_out_c)
        p_in_c = p_in_c.difference(whr_eq_)
        p_out_c = p_out_c.difference(whr_eq_)  # out/equal difference
    # -- splitter, poly point equal
    if whr_eq2_.size > 0:
        whr_eq2_ = set(whr_eq2_)
        c_in_p = set(c_in_p)
        c_out_p = set(c_out_p)
        c_in_p = c_in_p.difference(whr_eq2_)
        c_out_p = c_out_p.difference(whr_eq2_)
    # --
    p_in_c = sorted(list(p_in_c) + list(whr_eq_))
    p_out_c = sorted(list(p_out_c))
    c_in_p = sorted(list(c_in_p) + list(whr_eq2_))
    c_out_p = sorted(list(c_out_p))
    #
    if len(p_out_c) == 0:
        p_out_c = [poly.shape[0]]
    s, r = divmod(whr.shape[0], 2)  # check for even pairing
    if r == 0:
        x = (whr.reshape(-1, 4)).copy()
        x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
    else:
        x = (whr[:s * 2].reshape(-1, 4)).copy()
        x[:, 1], x[:, 2] = x[:, 2], x[:, 1].copy()
        lstx, lsty = whr[s * 2:][0]
        x = np.concatenate((x, np.array([[lstx, -1, lsty, -1]])), axis=0)
    crossings = x.copy()
    pair_lst = [x_pnts[i: i + 2] for i in range(0, len(whr), 2)]
    ply_seen = []
    clp_seen = []
    tot_ = []
    go_on = True
    # --
    for cnt, cross in enumerate(crossings):
        if cross[1] == -1:
            go_on = False
            continue
        c_0, c_1, p_0, p_1 = cross
        x0, x1 = pair_lst[cnt]  # intersection points on that segment
        c0_in, c1_in, p0_in, p1_in = _in_checks_(c_0, c_1, p_0, p_1)
        sub_ = []
        p0_seen = p_0 in ply_seen
        c0_seen = c_0 in clp_seen
        #
        # splitter -> intersection point check and reverse check
        spl_eq_x0x1 = (splitter[[c_0, c_0 + 1]] == pair_lst[cnt]).all()
        x0x1_eq_spl = (splitter[[c_0, c_0 + 1]] == pair_lst[cnt][::-1]).all()
        #
        if (c_0 == c_1):
            # -- splitter crosses 2 poly segments splitter-x0x1 are equal
            if spl_eq_x0x1:
                if not c0_seen:
                    clp_seen.append(c_0)
                    sub_.extend(splitter[[c_0, c_0 + 1]])
                    tot_.append([cross.ravel(), sub_])
                ply_seen.extend([p_0])
                if p_0 + 1 in p_out_c:
                    ply_seen.extend([p_0 + 1])
                    # if not c0_seen:
                    #    clp_seen.append(c_0)
            # -- splitter crosses 2 poly segments splitter-x0x1 reverse equal
            elif x0x1_eq_spl:
                if not c0_seen:
                    clp_seen.append(c_0)
                    sub_.extend(splitter[[c_0, c_0 + 1]])
                    tot_.append([cross.ravel(), sub_])
                ply_seen.extend([p_0])
            # -- splitter crosses 2 poly segments
            elif not spl_eq_x0x1:
                # -- before checks
                before_ = None
                if p_1 - p_0 == 1:
                    btw_ = None
                    bf_ids, before_ = _before_chk_(ply_seen, p_0)
                    if before_ is not None:
                        ply_seen.extend(bf_ids)
                # -- between checks
                else:
                    if p0_seen:
                        btw_ = None
                    elif cnt == 0 and p_0 >= 0 and (p_1 == poly.shape[0] - 2):
                        btw_ = None  # first/last point check
                    elif cnt > 0:
                        mx_ = max(ply_seen)
                        vals = _between_chk_(mx_, p_1 + 1, poly, p_in_c)
                        if vals is not None:
                            sub_.extend(vals)
                    else:
                        btw_ = _between_chk_(p_0, p_1, poly, p_in_c)
                # -- after check, on previous loop
                if len(ply_seen) > 0:
                    # if cnt == len(crossings):
                    if p_0 - ply_seen[-1] > 1:  # neg values indicate closing
                        remain, aft, pnts = _after_chk_(ply_seen, remain)
                        if aft is not None:
                            ply_seen.extend(aft)
                        if pnts:
                            tot_[-1][-1].extend(pnts)
                #
                # -- individual point checks
                # (1) p_0 and p_1 in
                if p0_in and p1_in:  # ** A, B0 bad
                    if not c0_seen:
                        sub_.extend([poly[p_1 + 1]])
                        # sub_.extend(splitter[[c_0, c_0 + 1]])  # insert clip
                    else:
                        sub_ = [i for i in poly[p_0:p_1 + 1]]
                    ply_seen.extend([p_0, p_1])
                # (2) p_0 in, p_1 out
                elif p0_in and (not p1_in):
                    for i in [before_, [poly[p_0]], [x0], btw_, [x1]]:
                        if i is not None:
                            sub_.extend(i)
                    ply_seen.extend([p_0, p_1])
                # (3) p_0 out, p_1 in
                elif (not p0_in) and p1_in:
                    sub_.extend([poly[p_1]])
                    # distance check
                    if _dist_(splitter[c_0], x0) < _dist_(splitter[c_0], x1):
                        sub_.extend([x0])
                        if btw_ is not None:
                            sub_.extend(btw_)
                        sub_.extend([x1])
                    else:
                        sub_.extend([x1])
                        if btw_ is not None:
                            sub_.extend(btw_)
                        sub_.extend([x0])
                    if p_0 == 0:
                        ply_seen.extend([p_0])
                    else:
                        ply_seen.extend([p_0, p_1])
                # (4) p0 out, p1 out, hence all 4 are outside
                elif (not p0_in) and (not p1_in):
                    # distance check
                    if _dist_(splitter[c_0], x0) < _dist_(splitter[c_0], x1):
                        sub_ = [x0]
                        if btw_ is not None:
                            sub_.extend(btw_)
                        sub_.extend([x1])
                    else:
                        sub_ = [x1]
                        if btw_ is not None:
                            sub_.extend(btw_)
                        sub_.extend([x0])
                    if p_0 == 0:
                        ply_seen.extend([p_0])
                    else:
                        ply_seen.extend([p_0, p_1])
                clp_seen.append(c_0)
                tot_.append([cross.ravel(), sub_])
                #
                # -- final check
                if cnt == len(crossings) - 1:
                    last_ = poly.shape[0] - 1
                    pnts = _final_chk_(p_1, last_)
                    if pnts is not None:
                        tot_[-1][-1].extend(pnts)
        # --
        # -- splitter consists of 2 lines --
        elif c_0 != c_1:
            # --
            # (1) c0 in/on poly
            if c0_in and c1_in:
                mx_ = max(ply_seen)
                if (p_0 - mx_) > 0:
                    sub_.extend(_between_chk_(mx_, p_0 + 1, poly, p_in_c))
                    sub_.extend([x0, splitter[c_1], x1])
            elif c0_in:
                if p0_in and p1_in:  # (a) both p_0 and p_1 in
                    if p_0 not in ply_seen:
                        sub_.extend([x0])  # poly[p_0] == x0  assert
                    if p_1 not in ply_seen:
                        sub_.extend([x1])  # poly[p_1] == x1  assert
                elif p0_in:  # (b) p_0 in
                    if p_0 not in ply_seen:
                        sub_.extend([x0])  # poly[p_0] == x0  assert
                    if c1_in:  # check c1 as well  **** not needed
                        sub_.extend([splitter[c_1], x1])  # **** not needed
                elif p1_in:  # (c) p_1 in
                    if cnt == 0:
                        sub_ = [x0, poly[p_1]]
                    else:
                        sub_ = [x0, poly[p_1]]
            # (3) p0 in clipper, c1 in poly
            elif c1_in:
                if p0_in:
                    if cnt == 0:
                        pre_ = [i for i in p_in_c if i < p_0]
                        if pre_:
                            sub_.extend([poly[i] for i in pre_])
                    if p0_seen:
                        sub_.extend([x0, splitter[c_1], x1])
                        ply_seen.extend([p_1])
                    else:
                        sub_.extend([poly[p_0], x0, splitter[c_1], x1])
                        ply_seen.extend([p_0])  # p_1])
                    # -- below needed for b4, c1 when the last cutter is used
                    s_idx = [i for i in range(min(p_0, p_1), p_out_c[-1])
                             if i not in ply_seen]
                    in_idx = [i for i in s_idx if i in p_in_c]
                    if s_idx:
                        ply_seen += s_idx
                    if in_idx:
                        to_add = [poly[i] for i in in_idx]
                        sub_.extend(to_add)
            # (4) p_1 in clipper, end in poly
                if p1_in:
                    if c0_seen:  # seen
                        if p1_in:
                            sub_ = [x0, poly[p_1]]
                        else:
                            sub_ = [x0]
                    else:  # not seen
                        sub_ = [c_0, x0]
                        pieces = [poly[i] for i in p_in_c]
                        sub_.extend(pieces)
                    ply_seen.extend([p_0])  # ** redundant??
            # (5) both poly and clipper points outside of each other
            elif (not p0_in) and (not p1_in):
                if c0_in:
                    sub_.extend([splitter[c_0], x0])
                elif c_0 + 1 in c_in_p:
                    sub_.extend([x0, splitter[c_0 + 1]])
                btw_ = _between_chk_(c_0, c_1 + 1, splitter, c_in_p)
                if btw_ is not None:
                    sub_.extend(btw_)
                sub_.extend([x1])
                # elif c_0 + 1 in c_in_p:
                #     sub_.extend([x0, splitter[c_0 + 1]])
                #     if c_1 - c_0 == 1:
                #         sub_.extend([splitter[c_1]])
                #     else:
                #         btw_ = _between_chk_(c_0, c_1 + 1, splitter, c_in_p)
                #         if btw_ is not None:
                #             sub_.extend(btw_)
                #     sub_.extend([x1])
            ply_seen.extend([p_0])  # p_1])
            clp_seen.extend([c_0])  # c_1])
            tot_.append([cross.ravel(), sub_])
        if not go_on:
            break
    new_ = []
    for i in tot_:
        new_.extend([j for j in i[1] if len(j) > 0])
    new_ = np.asarray(new_)
    return new_, crossings, tot_


# ---- (5) multi shape version  ** not final
#
def poly_clip(clippers, polys, inside=True):
    """Return the inside or outside clipped geometry.

    Parameters
    ----------
    clippers, polys : list or array_like
        The geometries representing the clipping features and the polygon
        geometry.  A list or array with ndim > 2 is expected for both
    inside : boolean
        True, keeps the inside geometry.  False for outside, the clip extents.

    Requires
    --------
    `common_extent`, `_in_out_`

    Returns
    -------
    The shared_extent between the clipping and poly features.  The geometry
    inside/outside depending on the `inside` boolean.

    Example
    -------
    >>> poly_clip([c0], [E])  # lists provided
    """
    shared_extent = []
    to_keep = []
    for i, clp in enumerate(clippers):
        for j, poly in enumerate(polys):
            shared, extent = common_extent(clp, poly)
            if shared:
                shared_extent.append([i, j, extent])
                inout = _clp_(clp, poly, keep_in=inside)  # ** check
                if inout is None:
                    pass
                in_, out_ = inout
                if inside:
                    if len(in_) > 0:
                        to_keep.extend(in_)
                else:
                    if len(out_) > 0:
                        to_keep.extend(out_)
    # remember that shared_extent and out2 should be sliced prior to plotting
    # ie. out2[0] is the result of clipping one polygon by one clipper
    return shared_extent, to_keep


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
