# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
-----------
npg_boolean
-----------

** Boolean operations on poly geometry.

----

Script :
    npg_boolean.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2023-04-12

Purpose
-------
Functions for boolean operations on polygons:

    - clip
    - difference
    - erase
    - merge
    - split
    - union
 A and B, A not B, B not A
 A union B (OR)
 A intersect B (AND)
 A XOR B

"""
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E1101,E1121
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0621
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915

import sys
import numpy as np
import npg  # noqa
from npg.npg_helpers import a_eq_b
from npg_bool_hlp import _del_seq_pnts_, prep_overlay
from npg.npg_plots import plot_polygons  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]

__all__ = ['clip_poly', 'find_overlap_segments', 'split_poly']
__helpers__ = ['del_seq_pnts', '_roll_', 'prep_overlay']


# ---- (1) clip polygons
#
def clip_poly(poly, clp, as_geo=False):
    """Clip a polygon `poly` with another polygon `clp`.

    Parameters
    ----------
    poly, clp : array_like
        `poly` is the polygon being clipped by polygon `clp`
    as_geo : boolean
        True to return a Geo array.  False for an ndarray.

    Requires
    --------
    `npg_helpers` : `a_eq_b`

    `_roll_`, `_del_seq_pnts_
    """

    def _bits_(i0, i1, in_, seen_):
        """Return indices which are in `in_` and not in `seen_`.

        Parameters
        ----------
        i0, i1 : integers
        in_, seen_ : list / array
            These are the values in an `inclusion` being checked if they
            have not been `seen` yet.

        Notes
        -----
        >>> p_add = [m for m in range(j_p, j_c + 1)
        ...          if m in ip and m not in p_seen]
        """
        r = set(range(i0, i1 + 1))
        ids = sorted(list(r.intersection(in_).difference(seen_)))
        return ids

    # -- quick bail 1
    bail = a_eq_b(poly, clp).all()  # from npg_helpers
    if bail:
        print("\nInput polygons are equal.\n")
        return None
    #
    # -- (1) prepare the arrays for clipping
    #
    # -- Returns the intersections, the rolled input polygons, the new polygons
    #    and how the points in both relate to one another.
    result = prep_overlay([poly, clp], roll=False, polygons=[True, True])
    x_pnts, pl, cl, pl_new, cl_new, args = result
    px_in_c, p_in_c, p_eq_c, p_eq_x, cx_in_p, c_in_p, c_eq_p, c_eq_x = args
    #
    # -- locate first intersection and roll geometry to it.
    r0 = np.nonzero((x_pnts[0] == pl_new[:, None]).all(-1).any(-1))[0]
    r1 = np.nonzero((x_pnts[0] == cl_new[:, None]).all(-1).any(-1))[0]
    fix_out = []
    out2 = []
    nums = [r0[0], r1[0]]
    # --
    fixes = [[px_in_c, p_in_c, p_eq_c, p_eq_x],
             [cx_in_p, c_in_p, c_eq_p, c_eq_x]]
    for cnt, ar in enumerate([pl_new, cl_new]):
        num = nums[cnt]
        fix = fixes[cnt]
        tmp = np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0)
        fix_out.append(tmp)
        new = []
        for i in fix:
            if i:
                v = [j + num for j in i]
                v = [i - 1 if i < 0 else i for i in v]
                new.append(v)
            else:
                new.append(i)
        out2.append(new)
    #
    # -- rolled output and point locations fixed
    pl_r, cl_r = fix_out  # temporary, renamed down further
    px_in_c_1, p_in_c_1, p_eq_c_1, p_eq_x_1 = out2[0]
    cx_in_p_1, c_in_p_1, c_eq_p_1, c_eq_x_1 = out2[1]
    # --
    z0 = np.nonzero((x_pnts == cl_r[:, None]).all(-1).any(-1))[0]
    z1 = np.nonzero((cl[c_in_p] == cl_r[:, None]).all(-1).any(-1))[0]
    z1a = np.nonzero((cl[c_eq_x] == cl_r[:, None]).all(-1).any(-1))[0]
    idx0 = sorted(list(set(np.concatenate((z0, z1, z1a)))))
    # --
    z2 = np.nonzero((x_pnts == pl_r[:, None]).all(-1).any(-1))[0]
    z3 = np.nonzero((pl[p_in_c] == pl_r[:, None]).all(-1).any(-1))[0]
    z3a = np.nonzero((pl[p_eq_x] == pl_r[:, None]).all(-1).any(-1))[0]
    idx1 = sorted(list(set(np.concatenate((z2, z3, z3a)))))
    #
    # -- mask those that are in, out or on
    cl_n = cl_r[idx0]  # cl, with just the intersection and `in` points
    pl_n = pl_r[idx1]
    #
    cN = len(cl_n) - 1
    pN = len(pl_n) - 1
    # --
    # inside points for both
    inC, inP = np.where((pl_n == cl_n[:, None]).all(-1))
    inCinP = np.concatenate((inC[None, :], inP[None, :])).T
    cinside = np.nonzero((pl_n != cl_n[:, None]).any(-1).all(-1))[0]
    pinside = np.nonzero((cl_n != pl_n[:, None]).any(-1).all(-1))[0]
    #
    # -- make sure first intersection is added
    #
    whr0 = np.nonzero(inCinP[:, 0] == cN)[0]
    whr1 = np.nonzero(inCinP[:, 1] == pN)[0]
    inCinP[whr0, 0] = 0
    inCinP[whr1, 1] = 0
    inCinP = inCinP[1:-1]  # strip off one of the duplicate start/end 0`s
    #
    prev = inCinP[0]      # -- set the first `previous` for enumerate
    ic = sorted(cinside)  # -- use the equivalent of p_in_c, c_in_p
    ip = sorted(pinside)
    close = False
    null_pnt = np.array([np.nan, np.nan])
    out, p_seen, c_seen = [], [], []
    for cnt, p in enumerate(inCinP[1:-1], 1):  # enumerate from inCinP[1:]
        i_c, j_c = p       # current ids, this is an intersection point
        i_p, j_p = prev    # previous ids
        d0, d1 = p - prev  # differences in ids
        sub, p_a, c_a = [], [], []
        # --
        if i_p == 0:
            out.append(cl_n[i_p])  # already added, unless i_c is first, 0
        # --
        if d0 == 0:
            if j_c not in p_seen:
                out.append(cl_n[i_p])
        elif d0 == 1:
            if d1 == 1:
                if j_c not in p_seen:  # same point
                    sub.append(cl_n[i_c])  # in original
            elif d1 < 0:  # negative so can't use `_bits_`
                if j_c not in p_seen:
                    sub.append(pl_n[j_c])
            elif d1 > 1:  # this may close a sub-polygon
                if j_c not in p_seen:  # and cnt < 2 :   # cludge
                    p_a = _bits_(j_p, j_c, pinside, p_seen)  # !!!!!
                    if d1 > 2 and cnt > 2 and len(p_a) == 0:  # check
                        sub.append(null_pnt)  # this works with E, d0_
                    if p_a:  # needed for edgy1, eclip
                        sub.extend(pl_n[p_a])
                    sub.append(cl_n[i_c])  # append cl_n[i_c] or pl_n[j_c]
                if j_c + 1 in p_seen:  # if the next point was seen, close out
                    sub.extend([pl_n[j_c + 1], null_pnt])
                    p_seen.append(j_c)  # add the index before decreasing
                    j_c -= 1
                    close = True
                elif j_c + 1 in ip:
                    # same as
                    # _bits_(j_c + 1, pN, ip, p_seen)
                    p_a = []  # poly points to add
                    st = j_c
                    for i in ip:
                        if i - st == 1:
                            p_a.append(i)
                            st = i
                        else:
                            break
                    if p_a:
                        sub.extend(pl_n[p_a])
                        nxt = p_a[-1] + 1
                        if nxt in inCinP[:, 1]:
                            sub.append(pl_n[nxt])
                            p_a.append(nxt)
                        close = True
                if i_c + 1 in ic:
                    sub.append(cl_n[i_c + 1])
                    c_a.append(i_c + 1)
                # --
                if close:
                    sub.append(null_pnt)
                    close = not close
        # --
        elif d0 > 1:
            if d1 == 1:
                c_a = _bits_(i_p, i_c, ic, c_seen)
                # sub.append(cl_n[i_c])  # only if d0 previous == 1 ????
                sub.extend(cl_n[c_a])
                sub.append(cl_n[i_c])  # edgy1, eclip
                # sub.append(pl_n[j_c])
            elif d1 < 0:
                c_a = _bits_(i_p, i_c, ic, c_seen)
                sub.extend(cl_n[c_a])
                sub.append(cl_n[i_c])  # in clip_poly2
            elif d1 > 1:
                c_a = _bits_(i_p, i_c, ic, c_seen)
                if c_a:
                    sub.extend(cl_n[c_a])
                sub.append(cl_n[i_c])
        # --
        elif d0 < 0:  # needs fixing with d1==1, d1<0, d1>0
            if i_c == 0 and i_p == cN - 1:  # second last connecting to first
                p_a = _bits_(j_p, pN, pinside, p_seen)
                sub.extend(pl_n[p_a])
                # sub.append(pl_n[pN])  # commented out 2023-03-19 for E, d0_
            elif i_c == 0 and cnt == len(inCinP) - 1:  # last slice was -1
                c_a = _bits_(i_p, cN, cinside, c_seen)
                sub.extend(cl_n[c_a])
                sub.append(cl_n[i_c])
        else:
            if i_c not in c_seen:
                sub.append(cl_n[i_c])
            # if i_c not in c_seen:  # not in clip_poly2
            #     sub.append(cl_n[i_c])
        #
        c_seen.extend([i_c, i_p])
        p_seen.extend([j_c, j_p])
        c_seen.extend(c_a)
        p_seen.extend(p_a)
        # --
        out.extend(sub)  # add the sub array if any
        prev = p         # swap current point to use as previous in next loop
        #
        # print("cnt {}\nout\n{}".format(cnt, np.asarray(out)))  # uncomment
    # --
    # -- post cleanup for trailing inside points
    inC_0, inP_0 = prev
    c_missing = list(set(cinside).difference(c_seen))
    p_missing = list(set(pinside).difference(p_seen))
    if p_missing or c_missing:
        # out.extend(pl_n[p_missing])
        # out.extend(cl_n[c_missing])
        msg = "\nMissed during processing clip {} poly {}"
        print(msg.format(c_missing, p_missing))
    #
    final = np.asarray(out)
    #
    # -- check for sub-arrays created during the processing
    whr = np.nonzero(np.isnan(final[:, 0]))[0]  # split at null_pnts
    if len(whr) > 0:
        ft = np.asarray([whr, whr + 1]).T.ravel()
        subs = np.array_split(final, ft)
        final = [i for i in subs if not np.isnan(i).all()]  # dump the nulls
        tmp = []
        for f in final:
            if not (f[0] == f[-1]).all(-1):
                f = np.concatenate((f, f[0][None, :]), axis=0)
                tmp.append(f)
        final = tmp
    # --
    # if as_geo:
    #     return npg.arrays_to_Geo(final, kind=2, info=None, to_origin=False)
    # return final, [out, subs, dups, pl_n, cl_n, xtras]
    return final

    # out, final = clip_poly(
    # all work as of 2023-03-19
    # out, final = clip_poly(edgy1, eclip)
    # out, final = clip_poly(E, d0_)
    # out, final = clip_poly(pl_, cl_)
    # out, final = clip_poly(p00, c00)


# ---- (2) split polygon
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
    `_roll_`, `_del_seq_pnts_`

    Returns
    -------
    Polygon split into two parts.  Currently only two parts are returned.
    Subsequent treatment will address multiple polygon splits.
    """
    #
    # -- (1) Prepare for splitting
    result = prep_overlay([poly, line], roll=False, polygons=[True, False])
    # -- intersection points, arrays rolled to first intersection,
    #    rolled with intersections added on, optional arguments
    x_pnts, pl_roll, cl_roll, pl_, cl_, args = result
    # -- quick bail
    # if len(x_pnts) > 2:
    #     msg = "Only 2 intersection points permitted, {} found"
    #     print(msg.format(len(x_pnts)))
    #     return poly, line
    #
    px_in_c, cx_in_p, p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    #
    r0 = np.nonzero((x_pnts[0] == pl_[:, None]).all(-1).any(-1))[0]
    r1 = np.nonzero((x_pnts[0] == cl_[:, None]).all(-1).any(-1))[0]
    r0, r1 = [r0[0], r1[0]]
    if r0 != 0:
        pl_ = np.concatenate((pl_[r0:-1], pl_[:r0], [pl_[r0]]), axis=0)
    if r1 != 0:
        if r1 == cl_.shape[0] - 1:
            cl_ = cl_[::-1]
        else:
            cl_ = np.concatenate((cl_[r1:-1], cl_[:r1], [cl_[r1]]), axis=0)
    #
    if len(cl_) == 2:  # split points are not at an intersection
        new_line = cl_
    elif len(cl_) > 2:  # keep next 2 lines in case I want to do multiple
        # get the new line values where they match pl_ eg intersections
        st_en = np.nonzero((pl_ == cl_[:, None]).all(-1).any(-1))[0]
        st, en = st_en[:2]
        if abs(st - en) == 1:
            new_line = cl_[[st, en]]
        else:
            new_line = cl_[st:en + 1]
    # -- order the clip line to match the intersection points
    # check to see if start equals the first x_pnt
    # st_en = new_line[[0, -1]]
    rev = new_line[::-1]
    # at least 1 split point is an intersection
    # -- The first intersection is point 0 in both poly and line
    st_en_ = np.nonzero((new_line == pl_[:, None]).all(-1).any(-1))[0]
    # st is always zero, so you want en to collect pl_ points
    st, en = st_en_[0], st_en_[1]  # the last one will be pl_.shape[0] - 1
    rgt = np.concatenate((pl_[:en], rev), axis=0)
    lft = np.concatenate((new_line, pl_[en + 1:]), axis=0)
    return lft, rgt

    # line = np.array([[0., 5.], [4., 4.], [6., 8.], [10.0, 9.0]])
    # line = np.array([[0., 5.], [4., 4.], [6., 8.], [12.5, 10.0]])
    # line = np.array([[6., 0.], [10., 12.]])
    # line = np.array([[6., 0.], [12., 10.]])


# ---- Extras section --------------------------------------------------------
def find_overlap_segments(arr, is_poly=True, return_all=True):
    """Locate and remove overlapping segments in a polygon boundary.

    Notes
    -----
    The idx, cnts and uni are for the frto array, so the indices will be out
    by 1.  Split `tmp` using the `idx_dup + 1`

    See Also
    --------
    See `simplify` in `npg_geom`.
    """
    tmp = _del_seq_pnts_(np.asarray(arr), poly=is_poly)  # keep dupl last point
    # -- create from-to points
    frto = np.concatenate((tmp[:-1], tmp[1:]), axis=1)
    frto_idx = np.arange(frto.shape[0])
    # sort within the row, not by column!!!
    sr = np.sort(frto, axis=1)
    # determine the `unique` properties of the row-sorted array
    uni, _, cnts = np.unique(
        sr, return_index=True, return_counts=True, axis=0)
    if arr.shape[0] == uni.shape[0]:  # -- all are unique, no duplicates
        return arr, []
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
        return final, [subs, idx_dup, dups]
    return final, []

# i0_, j0_ = inCinP[0]   # i0i1[0]   first    with old i0i1 = to new inCinP
# i1_, j1_ = inCinP[-1]  # i0i1[-1]  last
# preP, preC = prePC(i0_, i1_, cN, j0_, j1_, pN)  # changed 2023-03-15
# if preP and preC:
#     print("\nBoth have preceeding points. \n")
# elif preP:
#     out.extend(pl_n[preP])
#     out.append(pl_n[j0_])
#     p_seen.extend(preP + [j0_, j1_])
#     c_seen.append(i0_)
# elif preC:
#     out.extend(cl_n[preC])
#     out.append(cl_n[i0_])
#     c_seen.extend(preC + [i0_, i1_])
#     p_seen.append(j0_)
# else:
#     # c_seen.append(i1_)
#     # p_seen.append(j1_)
#     c_seen.extend([i0_, i1_])
#     p_seen.extend([j0_, j1_])


def prePC(i0_, i1_, cN, j0_, j1_, pN, pinside, cinside):
    """Determine pre `p` and `c` points."""
    preP, preC = [], []
    i1_ = 0 if i1_ in [i1_, cN] else i1_  # clp first/last point check
    j1_ = 0 if j1_ in [j1_, pN] else j1_  # poly first/last point check
    #
    # -- add preceeding pinside points
    if j0_ > 0 and j1_ < j0_:
        preP = [m for m in range(j1_, j0_ + 1) if m in pinside]
    # -- add preceeding cinside points
    if i0_ > 0 and i1_ < i0_:
        preC = [m for m in range(i1_, i0_ + 1) if m in cinside]
    return preP, preC


def postPC(inC_0, cN, inP_0, pN, cinside, pinside):
    """Determine pre `p` and `c` points."""
    preC, preP = [], []
    # -- add trailing cinside points
    if inC_0 != 0:
        preC = [m for m in range(inC_0, cN + 1) if m in cinside]
    # -- add trailing pinside points
    if inP_0 != 0:

        preP = [m for m in range(inP_0, pN + 1) if m in pinside]
    return preC, preP


def _bits2_(i0, i1, in_, seen_):
    """Return indices version 2."""
    r = set(range(i0, i1 + 1))
    ids = sorted(list(r.intersection(in_)))
    return ids


def _bits3_(i0, i1, in_=None, seen_=None):
    """Return indices which are in `in_` and not in `seen_`."""
    rev = False
    step = 1 if i1 >= i0 else -1
    rev = rev if step > 0 else ~rev
    r = set(range(i0, i1 + step, step))
    ids = sorted(list(r), reverse=rev)
    r = set(r)
    ids = sorted(list(r.intersection(in_).difference(seen_)))
    return ids


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")
