# -*- coding: utf-8 -*-
# noqa: D205, D400, F403, F401
r"""
---------
npg_boolean
---------

----

Script :
    npg_boolean.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2023-03-09

Purpose
-------
Functions for boolean operations on polygons:

    - clip
    - difference
    - erase
    - merge
    - split
    - union
 A and B, A not B, B, not A
 A union B (OR)
 A intersect B (AND)
 A XOR B


"""
import sys
import numpy as np
from npg.npg_helpers import uniq_2d, a_eq_b
from npg.npg_plots import plot_polygons  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]

__all__ = ['clip_poly', 'find_overlap_segments', 'split_poly']
__helpers__ = ['_add_pnts_', 'del_seq_pnts', '_roll_', '_sort_on_line_',
               '_wn_clip_']


# ---- (1) private helpers
#
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


def _del_seq_pnts_(arr, poly=True):
    """Remove sequential duplicates in a Nx2 array of points.

    Parameters
    ----------
    arr : array_like
        An Nx2 of point coordinates.
    poly : boolean
        True if the points originate from a polygon boundary, False otherwise.

    Notes
    -----
    This largely based on numpy.arraysetops functions `unique` and `_unique1d`.
    See the reference link in the script header.

    The method entails viewing the 2d array as a structured 1d array, then
    checking whether sequential values are equal.  In np.unique, the values
    are initially sorted to determine overall uniqueness, not sequential
    uniqueness.

    See Also
    --------
    `uniq_2d` above, which can be used in situations where you genuine
    uniqueness is desired.
    """
    # -- like np.unique but not sorted
    shp_in, dt_in = arr.shape, arr.dtype
    # ar = np.ascontiguousarray(ar)
    dt = [('f{i}'.format(i=i), dt_in) for i in range(arr.shape[1])]
    tmp = arr.view(dt).squeeze()  # -- view data and reshape to (N,)
    # -- mask and check for sequential equality.
    mask = np.empty((shp_in[0],), np.bool_)
    mask[0] = True
    mask[1:] = tmp[:-1] != tmp[1:]
    # wh_ = np.nonzero(mask)[0]
    # sub_arrays = np.array_split(arr, wh_[wh_ > 0])
    tmp = arr[mask]  # -- slice the original array sequentially unique points
    if poly:  # -- polygon source check
        if (tmp[0] != tmp[-1]).all(-1):
            arr = np.concatenate((tmp, tmp[0, None]), axis=0)
            return arr
    return tmp


def _roll_(arrs):
    """Roll point coordinates to a new starting position.

    Parameters
    ----------
    arrs : list of arrays or a single array

    Notes
    -----
    Rolls the coordinates of the Geo array or ndarray to put the start/end
    points as close to the lower-left of the ring extent as possible.

    If a single array is passed, a single array is returned otherwise a list
    of arrays.
    """
    # --
    def _closest_to_LL_(a, p, sqrd_=False):
        """Return point distance closest to the `lower-left, LL`."""
        diff = a - p[None, :]
        if sqrd_:
            return np.einsum('ij,ij->i', diff, diff)
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    # --
    if not isinstance(arrs, (list, tuple)):
        arrs = [arrs]
    out = []
    for ar in arrs:
        LL = np.min(ar, axis=0)
        dist = _closest_to_LL_(ar, LL, sqrd_=False)
        num = np.argmin(dist)
        out.append(np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0))
    if len(out) == 1:
        return out[0]
    return out


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


def _node_type_(p_in_c, c_in_p, poly, clp, x_pnts):
    """Return node intersection data. clipper polygon intersection`.

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
    # -- build the output
    p_in_c = list(set(p_in_c))
    c_in_p = list(set(c_in_p))
    if p_eq_c or p_eq_x:  # -- non-empty lists check
        #  p_in_c = reduce(np.union1d, [p_in_c, p_eq_c, p_eq_x])  # slow equiv.
        p_in_c = sorted(list(set(p_in_c + p_eq_c + p_eq_x)))
    if c_eq_p or c_eq_x:  # c_in_p + (p_eq_c, c_eq_x)
        c_in_p = sorted(list(set(c_in_p + c_eq_p + c_eq_x)))
    return p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x


# ---- (1) prepare for boolean operations
#
def _prepare_(arrs, roll=True, polygons=[True, True]):
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
        - `_node_type_`
        - `_add_pnts_`
        - `_del_seq_pnts_`

    Notes
    -----
    The sequence is as follows::

      - Roll the arrays so that their first coordinate is the closest to the
        lower left of the geometry extent.
      - Determine points inside each other`s geometry.
      - Classify the points.
      - Add intersection points both geometries.
      - Delete sequential duplicates if any exist.
    """
    #
    # -- roll towards LL.  `_wn_clp_` gets pnts inside, on, outside each other
    if len(arrs) != 2:
        print("Two poly* type geometries expected.")
        # return None
    a0, a1 = arrs
    is_0, is_1 = polygons
    if roll:
        a0, a1 = _roll_(arrs)
    vals = _wn_clip_(a0, a1, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals
    args = _node_type_(pInc, cInp, a0, a1, x_pnts)
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    a0_new, a1_new = _add_pnts_(a0, a1, x_pnts, whr)
    x_pnts = _del_seq_pnts_(x_pnts, poly=False)
    a0_new = _del_seq_pnts_(np.concatenate((a0_new), axis=0), poly=is_0)
    a1_new = _del_seq_pnts_(np.concatenate((a1_new), axis=0), poly=is_1)
    return x_pnts, a0, a1, a0_new, a1_new, args


# ---- (2) find segment overlaps
#
def find_overlap_segments(arr, is_poly=True, return_all=True):
    """Locate and remove overlapping segments in a polygon boundary.

    Notes
    -----
    The idx, cnts and uni are for the frto array, so the indices will be out
    by 1.  Split `tmp` using the `idx_dup + 1`
    """
    tmp = _del_seq_pnts_(np.asarray(arr), poly=is_poly)  # keep dupl last point
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


# ---- (3) clip polygons
#
def clip_poly(poly, clp):
    """Clip a polygon `poly` with another polygon `clp`.

    Parameters
    ----------
    poly, clp : array_like
        `poly` is the polygon being clipped by polygon `clp`

    Requires
    --------
    `npg_helpers` : `a_eq_b`

    `_roll_`, `_wn_clip_`, `_node_type_`, `_add_pnts_`, `_del_seq_pnts_
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

    # -- quick bail 1
    bail = a_eq_b(poly, clp).all()  # from npg_helpers
    if bail:
        print("\nInput polygons are equal.\n")
        return poly
    #
    # -- (1) prepare the arrays for clipping
    #        - roll, wn_clp, _node_type_, del
    result = _prepare_([poly, clp],
                       roll=True,
                       polygons=[True, True]
                       )
    # -- Returns the intersections, the rolled input polygons, the new polygons
    #    and how the points in both relate to one another.
    x_pnts, pl, cl, pl_new, cl_new, args = result
    p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    #
    # -- quick bail 2
    if len(x_pnts) == 0:
        print("No intersection between `poly` and `clp`.")
        return poly
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
    ic = cinside.tolist() + i0.tolist()  # clipper inside ids
    ip = pinside.tolist() + i1.tolist()  # poly inside ids
    ic = sorted(list(set(ic)))
    ip = sorted(list(set(ip)))
    print("previous current    diff")
    print("   i_p,j_p   i_c,j_c   d0,d1")
    print('-'*24)
    # -- assemble
    for cnt, p in enumerate(i0i1[1:], 1):
        i_c, j_c = p       # current ids
        i_p, j_p = prev    # previous ids
        d0, d1 = p - prev  # differences in ids
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
    result = find_overlap_segments(out, is_poly=True, return_all=True)
    final, subs, idx_dup, dups = result
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
    if (final[0] != final[-1]).all(-1):
        final = np.concatenate((final, final[0, None]), axis=0)
    if dups is not None:
        return final, dups
    return out, final, subs, pl_n, cl_n
    # work on E, d0 next


# ---- (4) split polygon
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
    `_roll_`, `_wn_clip_`, `_node_type_`, `_add_pnts_`, `_del_seq_pnts_`

    Returns
    -------
    Polygon split into two parts.  Currently only two parts are returned.
    Subsequent treatment will address multiple polygon splits.
    """
    #
    # -- (1) Prepare for splitting
    result = _prepare_([poly, line],
                       roll=True,
                       polygons=[True, False]
                       )
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
# ---- Final main section ----------------------------------------------------


if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
