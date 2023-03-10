# -*- coding: utf-8 -*-
# noqa: D205, D400, F403, F401
r"""
---------
npg_split
---------

----

Script :
    npg_split.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2023-03-01

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

Notes
-----
split
clip
union
erase
merge
difference
"""

import sys
import numpy as np
from npg.npGeo import roll_arrays
from npg.npg_helpers import uniq_2d, del_seq_dups
from npg.npg_clip import _wn_clip_, node_type
from npg.npg_plots import plot_polygons  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]


# ---- (1) private helpers
#
def _sort_on_line_(ln_pnts, cross_pnts):
    """Order intersection points on a straight line, from the start."""
    p = np.concatenate((ln_pnts, cross_pnts), axis=0)
    dxdy = np.abs(p[0] - p[1:])
    order = np.argsort(dxdy[:, 0])
    if dxdy.sum(axis=0)[0] == 0:
        order = np.argsort(dxdy[:, 1])
    tmp = p[1:][order]
    tmp = np.concatenate(np.atleast_2d(p[0], tmp))
    return uniq_2d(tmp), order


def _add_pnts_(poly, line, x_pnts, whr):
    """Return input arrays with intersections added to their lines.

    Parameters
    ----------
    poly, line : array_like
        N-2 arrays of clockwise ordered points
    x_pnts : array_like
        The intersection points.
    whr : array_like
        The id locations where the line points intersect the polygon segments.
    """
    def _srt_pnts_(p):
        """Order intersection points on a line, from the start/first point.

        `_sort_on_line_` is the full version, `p` is the combined point list.
        """
        if len(p) == 2:  # -- only start and end point
            return p
        dxdy = np.abs(p[0] - p[1:])  # difference from first
        if dxdy.sum(axis=0)[0] == 0:  # -- vertical line check
            order = np.argsort(dxdy[:, 1])  # sort ascending on y-values
        else:
            order = np.argsort(dxdy[:, 0])
        p[1:] = p[1:][order]
        return p
    # --
    p_ = np.concatenate((poly[:-1], poly[1:]), axis=1).reshape(-1, 2, 2)
    p_ = list(p_)  # [i for i in p_]
    c_ = np.concatenate((line[:-1], line[1:]), axis=1).reshape(-1, 2, 2)
    c_ = list(c_)  # [i for i in c_]
    for cnt, cp in enumerate(whr):
        cl, pl = cp  # print(f"cnt {cnt}  cp {cp}") add below to see order
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


def find_overlap_segments(arr, is_poly=True, return_all=True):
    """Locate and remove overlapping segments in a polygon boundary.

    Notes
    -----
    The idx, cnts and uni are for the frto array, so the indices will be out
    by 1.  Split `tmp` using the `idx_dup + 1`
    """
    tmp = del_seq_dups(np.asarray(arr), poly=is_poly)  # keep dupl. last point
    # -- create from-to points
    frto = np.concatenate((tmp[:-1], tmp[1:]), axis=1)
    frto_idx = np.arange(frto.shape[0])
    # sort within the row, not by column!!!
    sr = np.sort(frto, axis=1)
    # determine the `unique` properties of the row-sorted array
    uni, idx, cnts = np.unique(
        sr, return_index=True, return_counts=True, axis=0)
    if arr.shape[0] == uni.shape[0]:  # -- all are unique, no duplicates
        return arr
    # identify where, if any, the duplicates occur, get the indices of the rest
    whr = np.nonzero(cnts > 1)[0]
    dups = uni[whr]
    idx_dup = np.nonzero((dups == sr[:, None]).all(-1).any(-1))[0]
    idx_final = sorted(list(set(frto_idx).difference(idx_dup)))  # faster
    # idx_final2 = frto_idx[np.isin(frto_idx, idx_dup, invert=True)]  # slower
    dups = frto[idx_dup]
    subs = np.array_split(tmp, idx_dup + 1)  # for testing, added + 1
    final = tmp[idx_final]
    if return_all:
        return final, subs, idx_dup, dups
    return final


def find_seq_dups(arr, reverse=False, poly=True):
    """Find sequential duplicates in a Nx2 array of points.

    Parameters
    ----------
    arr : array_like
        An Nx2 of point coordinates.
    reverse : boolean
        True, to keep non-duplicates.  False to return them.
    poly : boolean
        True if the points originate from a polygon boundary, False otherwise.

    Notes
    -----
    See `np.npg_helpers.del_seq_dups` for the original that deletes the
    duplicates as well.
    """
    # -- like np.unique but not sorted
    shp_in, dt_in = arr.shape, arr.dtype
    # ar = np.ascontiguousarray(ar)
    dt = [('f{i}'.format(i=i), dt_in) for i in range(arr.shape[1])]
    tmp = arr.view(dt).squeeze()  # -- view data and reshape to (N,)
    # -- mask and check for sequential equality.
    mask = np.empty((shp_in[0],), np.bool_)
    if reverse:
        mask[0] = True
        mask[1:] = tmp[:-1] != tmp[1:]  # returns 1 for non-duplicates
    else:
        mask[0] = False
        mask[1:] = tmp[:-1] == tmp[1:]
    return np.nonzero(mask)[0]


def _prepare_overlay_(arrs, roll=True, polygons=[True, True]):
    """Prepare arrays for overlay analysis.

    Parameters
    ----------
    arrs : list/tuple
        The first geometry is the one being acted upon and the second is the
        one being used to overlay the first for operations such as clipping,
        splitting, intersection.
    polygons : list/tuple
        True, the input geometry is a polygon, False otherwise.
        Some operations permit polygon and polyline inputs, so you can alter

    Requires
    --------
    This script compiles the common functions::

        - `roll_array`
        - `_wn_clip_`
        - `node_type`
        - `_add_pnts_`
        - `del_seq_dups`

    Notes
    -----
    The sequence is as follows::

      - Roll the arrays so that their first coordinate is the closest to the
        lower left of the geometry extent.
      - Determine points inside each other`s geometry.
      - Add intersection points both geometries.
      - Delete sequential duplicates if any exist
    """
    #
    # -- roll towards LL.  `_wn_clp_` gets pnts inside, on, outside each other
    if len(arrs) != 2:
        print("Two poly* type geometries expected.")
        # return None
    a0, a1 = arrs
    is_0, is_1 = polygons
    if roll:
        a0, a1 = roll_arrays(arrs)
    vals = _wn_clip_(a0, a1, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals
    #
    # -- insert intersections, delete duplicates (`_add_pnts_`, `del_seq_dups`)
    args = node_type(pInc, cInp, a0, a1, x_pnts)
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    a0_new, a1_new = _add_pnts_(a0, a1, x_pnts, whr)
    x_pnts = del_seq_dups(x_pnts, poly=False)
    a0_new = del_seq_dups(np.concatenate((a0_new), axis=0), poly=is_0)
    a1_new = del_seq_dups(np.concatenate((a1_new), axis=0), poly=is_1)
    return x_pnts, a0_new, a1_new, args


# ---- (2) main function
#
def split_poly(poly, line):
    """Return polygon parts split by a polyline.

    Parameters
    ----------
    poly : array-like
        Single-part polygons are required.  Holes are not addressed.
    line : array-like
        The line can be a pair of points or a polyline.  Multipart polylines
        (not spatially connected) are not addressed.

    Requires
    --------
    `roll_arrays`, `_wn_clip_`, `node_type`, `_add_pnts_`, `del_seq_dups`

    Returns
    -------
    Polygon split into two parts.  Currently only two parts are returned.
    Subsequent treatment will address multiple polygon splits.
    """
    #
    # -- Prepare for splitting
    result = _prepare_overlay_([poly, line],
                               roll=True,
                               polygons=[True, False])  # -- (1)
    x_pnts, pl_new, cl_new, args = result
    #
    if len(x_pnts) > 2:
        msg = "Only 2 intersection points permitted, {} found"
        print(msg.format(len(x_pnts)))
        return poly, line
    #
    if len(cl_new) == 2:  # split points are not at an intersection
        new_line = cl_new
    elif len(cl_new) > 2:  # keep next 2 lines in case I want to do multiple
        st_en = np.nonzero((x_pnts == cl_new[:, None]).all(-1).any(-1))[0]
        st, en = st_en[:2]
        if abs(st - en) == 1:
            new_line = cl_new[[st, en]]
        else:
            new_line = cl_new[st:en+1]
    # -- order the clip line to match the intersection points
    # check to see if start equals the first x_pnt
    st_en = new_line[[0, -1]]
    up = new_line
    down = new_line[::-1]
    if (st_en[0] == x_pnts[0]).all(-1):  # equal?
        up, down = down, up
    # at least 1 split point is an intersection
    # -- find the intersection point indices on pl_new
    st_en_ = np.nonzero((x_pnts == pl_new[:, None]).all(-1).any(-1))[0]
    st, en = st_en_[0], st_en_[-1]
    rgt = [np.atleast_2d(i) for i in [pl_new[:st], down, pl_new[en:]]]
    rgt = np.concatenate(rgt, axis=0)
    lft = [np.atleast_2d(i) for i in [pl_new[st:en], up]]
    lft = np.concatenate(lft, axis=0)
    if (rgt[0] != rgt[-1]).all(-1):
        rgt = np.concatenate((rgt, rgt[0][None, :]), axis=0)
    if (lft[0] != lft[-1]).all(-1):
        lft = np.concatenate((lft, lft[0][None, :]), axis=0)
    return lft, rgt

    # line = np.array([[0., 5.], [4., 4.], [6., 8.], [10.0, 9.0]])
    # line = np.array([[0., 5.], [4., 4.], [6., 8.], [12.5, 10.0]])
    # line = np.array([[6., 0.], [10., 12.]])
    # line = np.array([[6., 0.], [12., 10.]])


def clip_poly(poly, clp):
    """Clip a polygon `poly` with another polygon `clp`.

    Parameters
    ----------
    poly, clp : array_like
        `poly` is the polygon being clipped by polygon `clp`

    Requires
    --------
    `roll_arrays`, `_wn_clip_`, `node_type`, `_add_pnts_`, `del_seq_dups
    """
    def prePC(i0_, i1_, j0_, j1_):
        """Determine pre `p` and `c` points."""
        preP, preC = [], []
        if j0_ > 0 and j1_ < j0_:
            preP = [m for m in range(j1_, j0_ + 1) if m in pinside]
        # -- add preceeding cinside points
        if i0_ > 0 and i1_ < i0_:
            preC = [m for m in range(i1_, i0_ + 1) if m in cinside]
        return preP, preC

    # -- (1) prepare the arrays for clipping
    # - roll, wn_clp, node_type, del
    result = _prepare_overlay_([poly, clp],
                               roll=True,
                               polygons=[True, True])
    x_pnts, pl, cl, args = result
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    #
    # -- (2) clip
    vals = _wn_clip_(pl, cl, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals
    args = node_type(pInc, cInp, pl, cl, x_pnts)
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    #
    # -- issues with singleton intersections at a point
    pl_new, cl_new = _add_pnts_(pl, cl, x_pnts, whr)  # -- 2023-02-27
    x_pnts = del_seq_dups(x_pnts, poly=False)
    pl_new = del_seq_dups(np.concatenate((pl_new), axis=0), poly=True)
    cl_new = del_seq_dups(np.concatenate((cl_new), axis=0), poly=True)
    #
    # -- assemble the `cl` points that are in `pl`
    z0 = np.nonzero((x_pnts == cl_new[:, None]).all(-1).any(-1))[0]
    cinp = cl[c_in_p]  # cinp = cl[cInp]
    z1 = np.nonzero((cinp == cl_new[:, None]).all(-1).any(-1))[0]
    idx0 = sorted(list(set(np.concatenate((z0, z1)))))
    cl_n = cl_new[idx0]
    #
    # -- assemble the `pl` points that are in `cl`
    z2 = np.nonzero((x_pnts == pl_new[:, None]).all(-1).any(-1))[0]
    pinc = pl[p_in_c]  # pinc = pl[pInc]
    z3 = np.nonzero((pinc == pl_new[:, None]).all(-1).any(-1))[0]
    idx1 = sorted(list(set(np.concatenate((z2, z3)))))
    pl_n = pl_new[idx1]
    #
    # probably redundant the next 4 lines
    if (cl_n[0] != cl_n[-1]).all(-1):
        cl_n = np.concatenate((cl_n, cl_n[0][None, :]), axis=0)
    if (pl_n[0] != pl_n[-1]).all(-1):
        pl_n = np.concatenate((pl_n, pl_n[0][None, :]), axis=0)
    cN = len(cl_n) - 1
    pN = len(pl_n) - 1
    # --
    i0, i1 = np.where((pl_n == cl_n[:, None]).all(-1))
    i0i1 = np.concatenate((i0[None, :], i1[None, :])).T
    cinside = np.nonzero((pl_n != cl_n[:, None]).any(-1).all(-1))[0]
    pinside = np.nonzero((cl_n != pl_n[:, None]).any(-1).all(-1))[0]
    # ----
    i0_, j0_ = i0i1[0]
    i1_, j1_ = i0i1[-1]
    # -- add preceeding pinside points
    out = []
    preP, preC = prePC(i0_, i1_, j0_, j1_)
    # -- assemble
    if preP:
        out.extend(pl_n[preP])
    if preC:
        out.extend(cl_n[preC])
    out.append(cl_n[0])  # -- make sure first intersection is added
    prev = i0i1[0]
    prev = [-1 if i in [cN, pN] else i for i in prev]
    #
    ic = cinside.tolist() + i0.tolist()
    ip = pinside.tolist() + i1.tolist()
    ic = sorted(list(set(ic)))
    ip = sorted(list(set(ip)))
    print("previous current    diff")
    print("   i_p,j_p   i_c,j_c   d0,d1")
    print('-'*24)
    # -- assemble
    for cnt, p in enumerate(i0i1[1:], 1):
        # diff = p - prev
        i_c, j_c = p
        i_p, j_p = prev
        d0, d1 = p - prev
        sub = []
        if d0 == 0:
            sub.append(cl_n[i_c])
        elif d0 == 1:
            if d1 == 1:
                sub.append(cl_n[i_c])
            elif d1 < 0:
                sub.append(cl_n[i_c])
                # out.append(cl_n[i_c])
                # to_add = [m for m in range(i_p, i_c + 1) if m in ic]
                # out.append(cl_n[i_c])
                # out.extend(cl_n[to_add])
                # print("  to_add: cl", to_add)
                # if j_p in ip:
                #     sub.append(pl_n[j_p])  # not sure
            elif d1 > 1:
                # out.append(pl_n[j_p])  # -- intersection point
                to_add = [m for m in range(j_p, j_c + 1) if m in ip]
                sub.extend(pl_n[to_add])
        elif d0 > 1:
            if d1 == 1:
                # sub.append(cl_n[i_p])
                to_add = [m for m in range(i_p, i_c + 1) if m in ic]
                sub.extend(cl_n[to_add])
                # sub.append(pl_n[j_c])
            elif d1 < 0:
                to_add = [m for m in range(i_p, i_c + 1) if m in ic]
                sub.extend(cl_n[to_add])
                sub.append(cl_n[i_c])
            elif d1 > 1:
                # sub.append(cl_n[i_p])
                # out.append(pl_n[j_p])  # -- intersection point
                # to_add = [m for m in range(j_p, j_c + 1) if m in ip]]
                # out.extend(pl_n[to_add])
                # print("  to_add: pl  ", to_add)
                to_add = [m for m in range(i_p, i_c + 1) if m in ic]
                sub.extend(cl_n[to_add])
                # sub.append(cl_n[i_c])
        else:
            sub.append(cl_n[i_c])
        print(f"  d0 d1= {[d0, d1]}\n  {sub}")
        print(cnt, ": ", prev, p)
        out.extend(sub)
        prev = p
    #
    out = np.asarray(out)
    # -- clean out segments that cross back on themselves
    #
    final, subs, idx_dup, dups = find_overlap_segments(out,
                                                       is_poly=True,
                                                       return_all=True)
    # -- try piecing the bits together
    if len(subs) > 1:
        zz = []
        tmp = [i for i in subs]
        ids = []
        for cnt, s in enumerate(tmp):
            chk = (s[0] == s[-1]).all(-1)
            if chk and len(s) > 1:
                zz.append(tmp.pop(cnt))
                ids.append(cnt)
        if len(tmp) > 0:
            the_rest = np.concatenate((tmp), axis=0)
            zz.append(the_rest)

    if dups is not None:
        return final, dups
    return out, final, subs, pl_n, cl_n
    # work on E, d0 next


def erase_poly(poly, clp):
    """Clip a polygon `poly` with another polygon `clp`.

    Parameters
    ----------
    poly, clp : array_like
        `poly` is the polygon being clipped by polygon `clp`

    Requires
    --------
    `roll_arrays`, `_wn_clip_`, `node_type`, `_add_pnts_`, `del_seq_dups
    """
    pl = roll_arrays(poly)
    cl = roll_arrays(clp)
    vals = _wn_clip_(pl, cl, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals
    args = node_type(pInc, cInp, pl, cl, x_pnts)
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    pl_new, cl_new = _add_pnts_(pl, cl, x_pnts, whr)  # -- 2023-02-27
    x_pnts = del_seq_dups(x_pnts, poly=False)
    pl_new = del_seq_dups(np.concatenate((pl_new), axis=0), poly=True)
    cl_new = del_seq_dups(np.concatenate((cl_new), axis=0), poly=False)
    #
    # -- assemble the `cl` points that are in `pl`
    z0 = np.nonzero((x_pnts == cl_new[:, None]).all(-1).any(-1))[0]
    cinp = cl[c_in_p]  # cinp = cl[cInp]
    z1 = np.nonzero((cinp == cl_new[:, None]).all(-1).any(-1))[0]
    idx0 = sorted(list(set(np.concatenate((z0, z1)))))
    cl_n = cl_new[idx0]
    cN = len(cl_n) - 1
    #
    # -- assemble the `pl` points that are in `cl`
    z2 = np.nonzero((x_pnts == pl_new[:, None]).all(-1).any(-1))[0]
    pinc = pl[p_in_c]  # pinc = pl[pInc]
    z3 = np.nonzero((pinc == pl_new[:, None]).all(-1).any(-1))[0]
    idx1 = sorted(list(set(np.concatenate((z2, z3)))))
    pl_n = pl_new[idx1]
    pN = len(pl_n) - 1
    # --
    i0, i1 = np.where((pl_n == cl_n[:, None]).all(-1))
    i0i1 = np.concatenate((i0[None, :], i1[None, :])).T
    cinside = np.nonzero((pl_n != cl_n[:, None]).any(-1).all(-1))[0]
    pinside = np.nonzero((cl_n != pl_n[:, None]).any(-1).all(-1))[0]
    #
    coutside = np.setdiff1d(np.arange(0, len(cl_n)), cinside)
    poutside = np.setdiff1d(np.arange(0, len(pl_n)), pinside)
    #
    # tomorrow  Get the output from clip_poly first
    """
    z5 = np.nonzero((poly == pl_n[:, None]).all(-1).any(-1))[0]
    z6 = np.nonzero((pl_n == poly[:, None]).all(-1).any(-1))[0]
    z5z6 = np.concatenate((z5[None, :], z6[None, :])).T
    z7 = np.nonzero((pl_n == out[:, None]).all(-1).any(-1))[0]
    z8 = np.nonzero((out == pl_n[:, None]).all(-1).any(-1))[0]
    z7z8 = np.concatenate((z7[None, :], z8[None, :])).T
    z9 = np.nonzero((cl_n == out[:, None]).all(-1).any(-1))[0]
    z10 = np.nonzero((out == cl_n[:, None]).all(-1).any(-1))[0]
    z9z10 = np.concatenate((z9[None, :], z10[None, :])).T  # ziplongest
    """
    return cinside, pinside, coutside, poutside


# ---- (3) not used ----------------------------------------------------------
#
def _whr_sort_(whr, pl_pnts, ln_pnts, x_pnts):
    """Order intersection points on a straight line, from the start.

    Parameters
    ----------
    whr : array
        Segment `whr` crossings array.  Where line crosses poly features.
    ln_pnts, x_pnts : arrays
        Poly, line and intersection point arrays.

    Returns
    -------
    split_keys : Indices needed to split the `whr` array on first column.
    c0c1_ids : Line-poly intersection ids.
    split_pnts : Intersection points split by `split_keys`.
    ft_pnts : `split_pnts` reshaped as from-to points.
    """
    c0, c1 = whr.T  # transpose to split along columns
    split_keys = np.nonzero(c0[:-1] - c0[1:])[0] + 1
    crossings = np.array_split(whr, split_keys)
    split_pnts = np.array_split(x_pnts, split_keys)
    ft_pnts = np.concatenate((ln_pnts[:-1], ln_pnts[1:]), axis=1)
    # ft_pnts.reshape(ft_pnts.shape[0], 2, 2)
    return crossings, split_keys, split_pnts, ft_pnts


#  the next two are from overlay
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
    """Return the side of a two-point line that the points lie on.

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

    `<https://stackoverflow.com/questions/26782038/how-to-eliminate-the-extra
    -minus-sign-when-rounding-negative-numbers-towards-zer>`_.

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


def split_seq(seq, last):  # sec_last
    """Return a sequence of point ids split at its numeric gaps.

    The defined gap is 1, since we are working with sequences of points.

    Parameters
    ----------
    seq : array_like
        A sorted ndarray is expected, but will be converted if needed.  Sorting
        is your task.
    last : integers
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


def _w_(pnts, poly, all_info=False):
    """Return winding number and/or the points inside."""
    # pnts in poly
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    x, y = pnts.T         # point coordinates
    x_x0 = x[:, None] - x0
    y_y0 = y[:, None] - y0
    diff_ = ((x1 - x0) * y_y0 - (y1 - y0) * x_x0) + 0.0  # einsum originally
    chk1 = (y_y0 >= 0.0)
    chk2 = np.less(y[:, None], y1)
    chk3 = np.sign(diff_).astype(np.int32)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn_vals = pos - neg
    wn_ = np.concatenate((wn_vals, np.array([wn_vals[0]])))
    non_0 = np.nonzero(wn_)[0]
    if all_info:
        return non_0, pnts[non_0]  # if you need interior points
    return non_0


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
