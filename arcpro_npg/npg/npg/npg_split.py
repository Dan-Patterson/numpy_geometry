# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E1101,E1121
# pylint: disable=F401
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0621
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# noqa: D205, D400, F401, F403, C0103
r"""
---------
npg_split
---------

----

Script :
    npg_clip.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2022-06-25

Purpose
-------
Functions for splitting polygons.

from line_side
    if line is None:
        A, B = line = np.array(left_right_pnts(pnts))
    else:
        A, B = line
    BAx, BAy = line[1] - line[0]
    XAx = pnts[:, 0] - A[0]
    YAy = pnts[:, 1] - A[1]
    return np.sign(BAx * YAy - BAy * XAx).astype('int')
"""

import sys
import numpy as np
from npg import npg_plots  # noqa
from npg.npg_plots import plot_polygons  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]


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


def _a_eq_b_(a, b, atol=1.0e-8, rtol=1.0e-5, return_pnts=False):
    """
    Return the points in `b` that are equal to the points in `a`.

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


def _w_(a, b):  # input arrays
    """Abbreviated winding number and intersections. See `npg.wn_clip`."""

    def _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0):
        """Return the intersection."""
        with np.errstate(all="ignore"):  # ignore all errors
            u_a = a_num / denom
            u_b = b_num / denom
            z0 = np.logical_and(u_a >= 0., u_a <= 1.)
            z1 = np.logical_and(u_b >= 0., u_b <= 1.)
            both = z0 & z1
            xs = (u_a * x1_x0 + x0)[both]
            ys = (u_a * y1_y0 + y0)[both]
        x_pnts = []
        if xs.size > 0:
            x_pnts = np.concatenate((xs[:, None], ys[:, None]), axis=1)
        whr = np.array(np.nonzero(both)).T
        return whr, x_pnts
    #
    x0, y0 = a[:-1].T  # line `from` coordinates
    x1, y1 = a[1:].T   # line `to` coordinates
    x1_x0, y1_y0 = (a[1:] - a[:-1]).T
    x2, y2 = b[:-1].T  # polygon `from` coordinates
    x3, y3 = b[1:].T   # polygon `to` coordinates
    x3_x2, y3_y2 = (b[1:] - b[:-1]).T
    # reshape poly deltas
    x3_x2 = x3_x2[:, None]
    y3_y2 = y3_y2[:, None]
    # deltas between pnts/poly x and y
    x0_x2 = x0 - x2[:, None]
    y0_y2 = y0 - y2[:, None]
    a_num = x3_x2 * y0_y2 - y3_y2 * x0_x2  # signed distance is a_0 + a_1
    b_num = x1_x0 * y0_y2 - y1_y0 * x0_x2
    # -- check sign check on `a_num`, and get intersections
    denom = (x1_x0 * y3_y2) - (y1_y0 * x3_x2)
    rgt_eq_lft = np.sign(a_num).astype(np.int32)
    whr, x_pnts = _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0)
    return rgt_eq_lft, whr, x_pnts


def split_seq(seq, last):  # sec_last
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
    N = 0
    if len(seq) == 0:
        return N, []
    if isinstance(seq, (list, tuple)):
        seq = np.asarray(seq)
    if seq.ndim > 1 or seq.shape[0] <= 1:
        return seq.size, seq.tolist()
    if seq[0] == 0 and last == seq[-1]:  # piece first to end if present
        tmp = np.concatenate((seq[1:], [last + 1]))
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


def _lineside_(p, strt, end):
    """Determine if a point (p) is `inside` a line segment (strt-->end).

    left (> 0), on (==0), right (< 0)
    """
    x, y, x0, y0, x1, y1 = *p, *strt, *end
    return (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)


def _dist_(a, b):
    """Return squared distance.  Add math.sqrt() for actual distance.

    Useage
    ------
    >>> _dist_(x_pnts[0], line[0]) < _dist_(x_pnts[1], line[0])
    ... if True, the first intersection point is closest to the line start
    """
    return (b[0] - a[0])**2 + (b[1] - a[1])**2


def sort_on_line(ln_pnts, x_pnts):
    """Order intersection points on a straight line, from the start."""
    p = np.concatenate((ln_pnts, x_pnts), axis=0)
    dxdy = np.abs(p[0] - p[1:])
    order = np.argsort(dxdy[:, 0])
    if dxdy.sum(axis=0)[0] == 0:
        order = np.argsort(dxdy[:, 1])
    tmp = p[1:][order]
    tmp = np.concatenate(np.atleast_2d(p[0], tmp))
    return uniq_2d(tmp)


def split_arr(poly, line, keep_right=True):
    """Return the result of a poly* split.

    Notes
    -----
    Intersection points along a straight line can be in a different order
    than their distance from the first point.  This checks the order.

    poly : A, C, D
    poly = np.array([[0., 0], [0, 10.], [10, 10], [10, 0], [0, 0]])
    poly = np.array([[0., 0], [0, 10.], [10, 10], [10, 8], [2, 8],
                     [2, 2], [10, 2], [10, 0], [0, 0]])
    poly = np.array([[0., 5.], [5., 10.], [10., 5.], [5., 0.], [0., 5.]])
    line = np.array([[0., 2], [10., 8.]])  # simple
    line = np.array([[0., 2], [2, 2], [5, 5], [10., 8.]])
    line = np.array([[10., 8.], [0., 2]])
    line = np.array([[0., 5.], [5., 10.]])
    line = np.array([[5., 10.], [10., 5.]])
    line2 = np.array([[  0.00,   2.00], [  2.00,   3.20]])
    """

    def sort_on_line(ln_pnts, x_pnts):
        """Order intersection points on a straight line, from the start."""
        p = np.concatenate((ln_pnts, x_pnts), axis=0)
        dxdy = np.abs(p[0] - p[1:])
        order = np.argsort(dxdy[:, 0])
        if dxdy.sum(axis=0)[0] == 0:
            order = np.argsort(dxdy[:, 1])
        tmp = p[1:][order]
        tmp = np.concatenate(np.atleast_2d(p[0], tmp))
        return uniq_2d(tmp)

    def l_r(pair, poly, line, whr, p_i, p_o):  # p_eq
        """Split polygon."""
        p0, p1 = pair
        st, en = whr[:, 0][:2]
        if abs(st - en) != 0:
            clp = np.concatenate(np.atleast_2d(p0, line[1:-1]), axis=0)
        else:
            clp = np.atleast_2d(p0)  # p0
        left_ = np.concatenate(
            np.atleast_2d(poly[p_o], p1, clp[::-1]), axis=0)
        rght_ = np.concatenate(
            np.atleast_2d(clp, p1, poly[p_i], p0), axis=0)
        return left_, rght_

    # -- get the intersections and point positions and clean up the pairs
    rgt_eq_lft, whr, x_pnts = _w_(poly, line)
    #
    pairs = sort_on_line(line, x_pnts)
    x0x1 = [pairs[i: i + 2] for i in range(0, len(pairs), 2)]
    if x0x1[-1].shape == (1, 2):
        pad = np.concatenate((x0x1[-1], np.array([[np.nan, np.nan]])), axis=0)
        x0x1[-1] = pad
    # --
    p_in = np.nonzero(rgt_eq_lft[0] < 0)[0]
    p_eq = np.nonzero(rgt_eq_lft[0] == 0)[0]
    p_out = np.nonzero(rgt_eq_lft[0] > 0)[0]
    poly_shp = poly.shape[0]
    last = poly_shp - 2
    end_ = poly_shp - 1 if 0 in p_out else poly_shp
    ids = set(list(np.arange(end_)))
    # -- split sequences if necessary
    Ni, p_in = split_seq(p_in, last)
    Ne, p_eq = split_seq(p_eq, last)
    No, p_out = split_seq(p_out, last)
    # --
    out_, in_ = [], []
    P = np.copy(poly)
    L = np.copy(line)
    for cnt, pair in enumerate(x0x1):  # [:-1]):
        if Ni == 0:
            p_o = list(ids.difference(set(p_in)))
        elif cnt <= Ni:
            bit = p_in[cnt]
            if last in bit:
                bit.append(0)
            p_o = list(ids.difference(set(bit)))  # use the first set
            # Ne, p_eq = split_seq(p_eq, last)
            # No, p_out = split_seq(p_out, last)
        chk = _dist_(pair[0], line[0]) > _dist_(pair[0], line[1])
        if chk:
            pair = pair[::-1]
        left_, rght_ = l_r(pair, P, L, whr, p_in[cnt], p_o)
        out_.append(left_)
        in_.append(rght_)
    return out_, in_


def wv(poly, line):
    """Return polygon parts split by a polyline.

    Parameters
    ----------
    poly : array-like
        Single-part polygons are required.  Holes are not addressed.
    line : array-like
        The line can be a pair of points or a polyline.  Multipart polylines
        (not spatially connected) are not addressed.

    Returns
    -------
    Polygon parts.  Two or more parts are returned.
    Subsequent treatment can address whether the polygons should be considered
    a multipart polygon or new polygons that share a common origin.
    """
    def sort_on_line(ln_pnts, x_pnts):
        """Order intersection points on a straight line, from the start."""
        p = np.concatenate((ln_pnts, x_pnts), axis=0)
        dxdy = np.abs(p[0] - p[1:])
        order = np.argsort(dxdy[:, 0])
        if dxdy.sum(axis=0)[0] == 0:
            order = np.argsort(dxdy[:, 1])
        tmp = p[1:][order]
        tmp = np.concatenate(np.atleast_2d(p[0], tmp))
        return uniq_2d(tmp)

    def spl_seq(seq, last):
        """Split sequence simple."""
        N = 0
        if len(seq) == 0:
            return N, []
        if isinstance(seq, (list, tuple)):
            seq = np.asarray(seq)
        if seq.ndim > 1 or seq.shape[0] <= 1:
            return N, [seq.tolist()]
        whr = np.nonzero(np.abs(seq[1:] - seq[:-1]) != 1)[0]
        if whr.size > 0:
            N += whr.size + 1
            return N, [s.tolist() for s in np.array_split(seq, whr + 1)]
        return N, [seq.tolist()]

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
    #
    # -- get the line side and where the intersections occur
    # -- The splitter may be a 2 point line or consist of many points.
    # -- This needs checking and a fix for a multipoint splitter line.
    # reorganize x_pnts
    rgt_eq_lft, whr, x_pnts = _w_(poly, line)
    uni, idx = np.unique(x_pnts, return_index=True, axis=0)
    x0x1 = x_pnts[np.sort(idx)]
    # if (len(x0x1) == 2) and (len(line) == 2):  # line equals x_pnts
    #     first, second = x0x1[0], x0x1[-1]
    # elif len(line) > len(x0x1) and len(x0x1) == 2:
    #     st, en = line[0], line[-1]
    #     first_end = _is_pnt_on_line_(st, en, x0x1[0], tolerance=1.0e-12)
    #     second_end = _is_pnt_on_line_(st, en, x0x1[1], tolerance=1.0e-12)
    #     if first_end:
    #         first = x0x1[0]
    #         second = np.concatenate(
    #                     (np.atleast_2d(line[1:-1], x0x1[-1])), axis=0)
    #         x0x1 = [first, second]
    #     elif not first_end:
    #         first = x0x1[0]
    #         second = np.concatenate(
    #                     (np.atleast_2d(line[1:-1], x0x1[-1])), axis=0)
    #         x0x1 = [first, second]
    #         print("check for min whr[:, 1] ")
    # -- determine if the line needs to be flipped by checking the polygon
    #   segment intersection
    # s = whr[0, 1]  # first row, poly segment
    # fr, to = poly[s:s+2]
    # chk0 = _is_pnt_on_line_(fr, to, x_pnts[0], tolerance=1.0e-12)
    # assume 1 pair of intersection points
    rel = rgt_eq_lft[0]
    rel = np.concatenate((rel, np.array([rel[0]])))
    p_in = np.nonzero(rel < 0)[0]
    p_eq = np.nonzero(rel == 0)[0]
    p_out = np.nonzero(rel > 0)[0]
    # last = poly.shape[0] - 1
    ids = np.arange(poly.shape[0]).tolist()
    # lft, rgt = [], []
    # prev = True if 0 in p_in else False
    #
    # Note:
    # pull the segments crossed by the line from `_w_`'s `whr` values
    # The following code does what the whr slice does
    # diff = (p_in[1:] - p_in[:-1]) - 1
    # delta = diff.max()
    # strt = np.nonzero(diff)[0]
    # end = strt + delta
    n = len(whr)
    if n == 2:
        bits = np.atleast_2d(x0x1[0], line[1:-1], x0x1[-1])
        line_ = np.concatenate(bits, axis=0)
        fr, to = whr[:, 1][[0, -1]]  # polygon segments crossed
        chk_order = _dist_(x_pnts[0], line[0]) < _dist_(x_pnts[0], line[1])
        if chk_order:  # x_pnts[0] is closest to start
            in_bits = np.concatenate(
                (poly[0:fr + 1], line_, poly[to + 1:]), axis=0
                )
            out_bits = np.concatenate(
                np.atleast_2d(poly[fr + 1: to + 1], line_[::-1]), axis=0
                )
        else:  # x_pnts are reversed, probably closing
            in_bits = np.concatenate((x_pnts[::-1], poly[fr + 1:to]), axis=0)
            out_bits = np.concatenate(
                np.atleast_2d(poly[0:fr], x_pnts, poly[to:], axis=0)
                )
    elif n == 3:
        ids = whr[:, 1].tolist()
        fr, to = ids[1], ids[n - 1]
        # vals = poly[[fr, to]]  # should be equal to `line`
        in_bits = np.concatenate(
            np.atleast_2d(poly[0:fr], line_, poly[to:]),
            axis=0
            )
        out_bits = np.concatenate(
            np.atleast_2d(poly[fr + 1: to + 1], line_[::-1]), axis=0
            )
    elif n == 4:
        ids = whr[:, 1].tolist()
        fr, to = ids[1], ids[n - 1]
        # vals = poly[[fr, to]]  # should be equal to `line`
        in_bits = np.concatenate((poly[0:fr], line, poly[to + 1:]), axis=0)
        out_bits = np.concatenate(
            np.atleast_2d(line[0], poly[fr + 1: to + 1], line[1:][::-1]),
            axis=0
            )
    return in_bits, out_bits  # lft, rgt


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
