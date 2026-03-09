# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
-----------
npg_clip_split
-----------

** Boolean operations on poly geometry.

----

Script :
    npg_clip_split.py

Author :
    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-12-21

Purpose
-------
Functions for boolean operations on polygons:

    - clip
    - split

"""
# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121

import sys
import numpy as np
import npg  # noqa
# from npg.npGeo import roll_arrays
from npg.npg_geom_hlp import a_eq_b
from npg.npg_bool_hlp import (_add_pnts_, _del_seq_dupl_pnts_, _wn_clip_,
                              prep_overlay)  # _node_type_)
from npg.npg_plots import plot_polygons  # noqa

#  --- alter or use below
fmt_ = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.3f}'.format}
np.set_printoptions(precision=3, threshold=100, edgeitems=10, linewidth=80,
                    suppress=True,
                    formatter=fmt_,
                    floatmode='maxprec_equal',
                    legacy='1.25')  # legacy=False or legacy='1.25'
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['clip_poly', 'split_poly', 'find_overlap_segments']


# ---- ---------------------------
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
    `npg_geom_hlp` : `a_eq_b`

    `npg.npg_bool_hlp` : `_del_seq_dupl_pnts_`
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
    bail = a_eq_b(poly, clp).all()  # from npg_geom_hlp
    if bail:
        print("\nInput polygons are equal.\n")
        return None
    #
    # -- (1) prepare the arrays for clipping
    #
    # -- Returns the intersections, the rolled input polygons, the new polygons
    #    and how the points in both relate to one another.
    result = prep_overlay([poly, clp], roll=False, p0_pgon=True, p1_pgon=True)
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
    # doesn't work with C, A or A, C


# ---- ---------------------------
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
    `npg.npg_bool_hlp` : `_wn_clip_`, `_del_seq_dupl_pnts_`

    Returns
    -------
    Polygon split into two parts.  Currently only two parts are returned.
    Subsequent treatment will address multiple polygon splits.

    Notes
    -----
    A good reference is:

    `<https://geidav.wordpress.com/2015/03/21/splitting-an-arbitrary-polygon-
    by-a-line/>`_.
    """

    def _prep_(poly, line):
        """Prep the array and line. mini `prepare_overlay`.

        Parameters
        ----------
        poly, line : arrays
            `poly` is the polygon being split by `line`.

        Returns
        -------
        Intersection points `x_pnts` and new versions of the polygon and line
        with intersection points added to them.
        """
        vals = _wn_clip_(poly, line, all_info=True)
        x_pnts, pInc, cInp, x_type, whr = vals
        if len(x_pnts) < 2:
            return [], None, None
        _p, _l = _add_pnts_(poly, line, x_pnts, whr)  # -- temporary poly, clp
        x_pnts = _del_seq_dupl_pnts_(x_pnts, poly=False)
        pl_ = _del_seq_dupl_pnts_(np.concatenate((_p), axis=0), poly=True)
        cl_ = _del_seq_dupl_pnts_(np.concatenate((_l), axis=0), poly=False)
        return x_pnts, pl_, cl_

    def _side_(pnts, line):
        """Return the line side that points are on.

        Parameters
        ----------
        pnts : array
            The points being tested.
        line : array
            The line to compare to.

        Notes
        -----
        The variant keeps the points on the line in both the left and right
        side ids since reconstructing both halves of a split will require those
        on as well.

        See _is_right_side in npg_pip
        x, y, x0, y0, x1, y1 = *p, *strt, *end  # p point, strt/end line point
        (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        """
        x0, y0, x1, y1 = line.ravel()
        v = (x1 - x0) * (pnts[:, 1] - y0) - (y1 - y0) * (pnts[:, 0] - x0)
        v = np.round(v, 6)
        lft_ids = np.nonzero(np.sign(v) >= 0.)[0]  # keep on and left of line
        rgt_ids = np.nonzero(np.sign(v) <= 0.)[0]  # keep on and right
        eq_ids = np.nonzero(np.sign(v) == 0.)[0]   # get separate ids for equal
        return lft_ids, rgt_ids, eq_ids

    #
    # -- (1) Prepare for splitting
    # Line direction must be increasing in x, swap if necessary.
    # Determine intersection points, and add to the arrays.  Roll to the
    # first intersection.
    #  
    if line[0][0] > line[1][0]:
        line = line[::-1]
    x_pnts, pl_, cl_ = _prep_(poly, line)  # -- `_prep_` stage
    #
    if len(x_pnts) < 2:
        print("\nNot enough intersection points to split.")
        return None, None
    # -- (2) remove any extraneous bits from cl_ and fix `line` as `dup_line`
    if cl_.shape[0] != x_pnts.shape[0]:
        whr_ = np.nonzero((cl_ == x_pnts[:, None]).all(-1))[1]
        dup_line = cl_[np.sort(whr_[[0, -1]])]
        cl_ = cl_[whr_]
    else:
        dup_line = np.copy(line)  #x_pnts[np.sort(whr_[[0, -1]])]
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
    # -- (3) run `_side_` to get the points to the left and right of the line
    pl_lft_ids, pl_rgt_ids, eq_ids = _side_(pl_, cl_[[0, -1]])  # line)  # --
    #
    # -- (4) check for sorting using pl_ and cl_.
    cl_pl_ids = np.nonzero((pl_[:-1] == cl_[:, None]).all(-1))
    arr = cl_pl_ids[1]  # -- new cl_ids
    _is_sorted_ = (arr[:-1] <= arr[1:]).all()  # if True,
    # 
    # -- (5) Locate gaps in pl_lft_ids and pl_rgt_ids.  Gaps in the ids can
    #  indicate that there may be more than 1 piece on one or both sides.
    #  Pair the gaps, slice the points and concatenate with the slicing line.
    lft_out = pl_[pl_lft_ids]  # -- left of the line, should only be 1 poly
    #
    _lft_ = np.nonzero(np.diff(pl_lft_ids) > 1)[0] + 1
    _l = np.array_split(pl_lft_ids, _lft_)
    lft_splits = [i for i in _l if len(i) > 1]
    #
    _rgt_ = np.nonzero(np.diff(pl_rgt_ids) > 1)[0] + 1
    _r = np.array_split(pl_rgt_ids, _rgt_)  # -- used below
    rgt_splits = [i for i in _r if len(i) > 1]
    #
    # -- (6) Now process the possibilities
    # -- simple split -- only one split on left and right
    N_l = len(lft_splits)
    N_r = len(rgt_splits)
    if N_l == 1 and N_r == 1:
        r_ = rgt_splits[0].tolist()
        r_ += [r_[0]]
        l_ = lft_splits[0].tolist()
        l_ += [l_[0]]
        return pl_[l_], pl_[r_]
    #
    # -- multiple intersections within polygon like E. `eq_ids` used here
    if N_l == 1 and N_r > 1:  # -- vertical line intersectiong multiple pnts
        l_ = pl_lft_ids[:-1]  # first and last are the same point
        # -- eq_ids from _side_
        _tmp = np.array(list(zip(eq_ids[:-1], eq_ids[1:])))
        _dif = (_tmp[:, -1] - _tmp[:, 0])
        _w = np.nonzero(_dif > 1)[0]
        keep_ = _tmp[_w]
        lft_final = []
        for i in keep_:
            f, t = i[0], i[1]
            ftf = l_[f: t + 1].tolist() + [f]
            if len(ftf) > 3:
                lft_final.append(pl_[ftf])
        rgt_final = pl_[pl_rgt_ids]  # -- or pl_[np.concatenate(_r)]
        if (dup_line[[0, 1]] == cl_[[-1, 0]]).all(-1).all():  # -- swapped line
            lft_final, rgt_final = rgt_final, lft_final
        return lft_final, rgt_final
    #
    # -- multiple splits on both sides
    if N_l == N_r:  # ---- carry on, multiple splits
        lft_final = []
        rgt_final = []  # [rgt_splits[0]]  # add the first split from cl_ segs
        keep_ = []
        for cnt, i in enumerate(lft_splits):  # [:-1]):
            i = lft_splits[cnt]               
            if i[-1] + 1 in pl_rgt_ids:
                lft_final.append(i)  # left piece
                rgt_final.append(rgt_splits[cnt])  # may not be needed !!!
            elif i[-1] == pl_.shape[0] - 1:  # only one clip outside
                # lft_final.append(i)
                # rgt_final.append(rgt_splits[cnt])
                keep_.append(i)
    # -- assemble the appropriate bits
    if _is_sorted_:  # splits are paired
        rgt_ids = np.concatenate(rgt_splits).tolist()  # equal to pl_rgt_ids !!
        rgt_ids += [rgt_ids[0]]
        rgt_out = pl_[rgt_ids]
        lft_ids = [i.tolist() + [i[0]] for i in lft_final if len(i) > 1]
        lft_out = [pl_[i] for i in lft_ids]    
    else:  # split orders differ, eg. last clp pairs with 1st ply
        lft_ids = np.concatenate(lft_splits).tolist()  # equal to pl_lft_ids !!
        lft_ids += [lft_ids[0]]
        lft_out = pl_[lft_ids]
        rgt_ids = [i.tolist() + [i[0]] for i in rgt_final if len(i) > 1]
        rgt_out = [pl_[i] for i in rgt_ids]
    # -- note line may have been swapped when intersecting, check the reverse
    if (dup_line == cl_[[-1, 0]]).all(-1).all():  # -- swapped line
       lft_out, rgt_out =  rgt_out, lft_out
    both_ = []
    for i in [lft_out, rgt_out]:
        if isinstance(i, list):
            both_.append(i)
        else:
            both_.append([i])
    l_, r_ = both_
    return l_, r_  # , pl_, cl_
#
    # sample geometry                                       swap srted lft  rgt
    # poly = C; line = np.array([[0., 5.], [10.0, 9.0]])  # works
    # poly = E; line = np.array([[0., 2.], [10.0, 9.0]])  # no   no    3    3
    # poly = E; line = np.array([[0., 9.], [8.0, 0.0]])   # no   yes   3    3
    # poly = E; line = np.array([[1., 0.], [2.0, 10.0]])  # no   -     1    1
    # poly = E; line = np.array([[2., 0.], [2.0, 10.0]])  # vertical   1    3
    # poly = E; line = np.array([[6., 0.], [10.0, 10.0]]) # no   yes   2    2
    # poly = aoi; line = np.array([[0., 2.], [10.0, 8.0]]) #no   yes   1    1
    # -- process, then plot
    # lft_out, rgt_out = split_poly(poly, line)
    # plot_polygons([pl_, cl_], True, True, True)

    # --------


    # line = np.array([[0., 5.], [4., 4.], [6., 8.], [10.0, 9.0]])
    # line = np.array([[0., 5.], [4., 4.], [6., 8.], [12.5, 10.0]])
    # line = np.array([[6., 0.], [10., 12.]])
    # line = np.array([[6., 0.], [12., 10.]])


# ---- ---------------------------
# ---- Extras section --------------------------------------------------------
#
def find_overlap_segments(arr, is_poly=True, return_all=True):
    """Locate and remove overlapping segments in a polygon boundary.

    Notes
    -----
    The idx, cnts and uni are for the frto array, so the indices will be out
    by 1.  Split `tmp` using the `idx_dup + 1`

    See Also
    --------
    See `simplify` in `npg_geom_ops`.
    """
    tmp = _del_seq_dupl_pnts_(np.asarray(arr), poly=is_poly)  # keep dupl pnt
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


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")
