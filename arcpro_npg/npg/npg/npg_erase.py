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
    2023-04-15

Purpose
-------
Functions for boolean operations on polygons:

    - erase

"""
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E1101,E1121
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0621
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915

import sys
import copy
import numpy as np
import npg  # noqa
from npg_bool_hlp import add_intersections  # prep_overlay
from npg.npg_plots import plot_polygons  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]

__all__ = ['erase_poly']
__helpers__ = []


# ---- (1) difference polygons
#
def erase_poly(poly, clp, as_geo=True):
    """Return the symmetrical difference between two polygons, `poly`, `clp`.

    Parameters
    ----------
    poly, clp : array_like
        `poly` is the polygon being differenced by polygon `clp`

    Requires
    --------
    `npg_helpers` : `a_eq_b`

    `_roll_`, `_wn_clip_`, `_node_type_`, `_add_pnts_`, `_del_seq_pnts_
    """

    def _out_c_(i_p, i_c, c_seen, c_outside):
        """Return sub lists."""
        if len(c_outside) > 0:
            if i_c in c_outside[0]:
                vals = c_outside.pop(0)
                c_seen.extend(vals)
                return cl_n[vals]
        return []

    def _out_p_(j_p, j_c, p_seen, p_outside):
        """Return sub lists."""
        if len(p_outside) > 0:
            if j_c in p_outside[0]:
                vals = p_outside.pop(0)
                p_seen.extend(vals)
                return pl_n[vals]
        return []

    def _in_c_(i_p, i_c, c_seen, c_in):
        """Return sub lists."""
        if len(c_in) > 0:
            if i_c in c_in[0]:
                vals = c_in.pop(0)
                c_seen.extend(vals)
                return cl_n[vals]
        return []

    def _in_p_(j_p, j_c, p_seen, p_in):
        """Return sub lists."""
        if len(p_in) > 0:
            if j_c in p_in[0]:
                vals = p_in.pop(0)
                p_seen.extend(vals)
                return pl_n[vals]
        return []

    def last_chk(j_p, j_c, p_seen, p_outside):
        """Last ditch check in case j_p and j_c are separated by a segment."""
        bits = []
        for i in [j_p, j_c]:
            for c0, lst in enumerate(p_outside):
                if i in lst:
                    vals = p_outside.pop(c0)
                    p_seen.extend(lst)
                    bits.append(pl_n[vals])
        return bits

    def last_chk2(_n, _p, _c, _seen, _outside, _inside):
        """Last ditch check in case j_p and j_c are separated by a segment.

        Parameters
        ----------
        inputs::

        - _n, _p, _c, _seen, _outside, _inside
        - pl_n, j_p, j_c, p_seen, p_outside, p_inside
        - cl_n, i_p, i_c, c_seen, c_outside, c_inside
        """
        out_bits = []
        in_bits = []
        v_seen = []
        N = _n.shape[0] - 1
        for i in [_p, _c]:
            for cnt, out_ in enumerate(_outside):
                if i in out_:  # and N not in out_:
                    vals = _outside.pop(cnt)  # [cnt]
                    out_bits.append(_n[vals])
                    v_seen.extend(vals)
            for cnt, in_ in enumerate(_inside):
                if i in in_:
                    vals = _inside.pop(cnt)  # [cnt]
                    in_bits.append(_n[vals])
                    v_seen.extend(vals)
        return out_bits, in_bits, v_seen

    def on_pairs(col):
        """Return sequential ids from the intersections not in or out."""
        segs = []
        for cn, v in enumerate(col[1:], 0):
            prev = col[cn]
            dff = v - prev
            if dff == 1:
                segs.append([prev, v])
        return segs

    # -- Returns the intersections, the rolled input polygons, the new polygons
    #    and how the points in both relate to one another.
    result = add_intersections(poly, clp,
                               roll_to_minX=True,
                               polygons=[True, True],
                               class_ids=True)
    pl_n, cl_n, id_plcl, x_pnts, p_out, p_in, c_out, c_in = result
    # pN = pl_n.shape[0] - 1
    # cN = cl_n.shape[0] - 1
    # --
    # Get the intersections, new polys, points inside and outside and
    # x_pnt ids from `add_intersections`.  Swap the order of the last.
    w0 = np.argsort(id_plcl[:, 1])  # get the order and temporarily sort
    # -- cut lines, where one crosses the other
    onConP = id_plcl[:, [1, 0]][w0]  # slice to rearrange the columns
    # -- two point cut lines, which cross the other polygon
    p_cut = on_pairs(id_plcl[:, 0])  # use id_plcl col 0 to save a sort
    c_cut = on_pairs(onConP[:, 0])   # use onConP col 0 since it is now sorted
    # -- cut lines that are more than two points and and are either inside or
    #    outside the other polygon
    p_outside = copy.deepcopy(p_out)
    p_inside = copy.deepcopy(p_in)
    c_outside = copy.deepcopy(c_out)
    c_inside = copy.deepcopy(c_in)
    #
    #  Determine preceeding points to first clip.
    out = []  # p_seen, c_seen = [], [], []
    prev = onConP[0]  # -- set the first `previous` for enumerate
    p_seen, c_seen = [], []
    in_segs = []  # collect `clipping` segments to use for clip.
    kind_ = []  # -1 symmetrical diff, 0 erase, 1 clip, 2 hole
    for cnt, p in enumerate(onConP[1:4], 1):  # enumerate fromonConP[1:]
        i_c, j_c = p       # current ids, this is an intersection point
        i_p, j_p = prev    # previous ids
        d0, d1 = p - prev  # differences in ids
        sub, bits, sub0 = [], [], []
        # --
        # if False, False then both ids are intersections
        chk0, chk1, chk2, chk3 = [False, False, False, False]
        if len(c_outside) > 0:
            chk0 = set([i_p, i_c]).issubset(set(c_outside[0]))
        if len(c_inside) > 0:
            chk1 = set([i_p, i_c]).issubset(set(c_inside[0]))
        if len(p_outside) > 0:
            chk2 = set([j_p, j_c]).issubset(set(p_outside[0]))
        if len(p_inside) > 0:
            chk3 = set([j_p, j_c]).issubset(set(p_inside[0]))
        # d0, d1, chk0, chk1, chk2, chk3
        # --
        # -- d0 clp ids are sequential and are on the densified clp line
        # -- d0 should never be <= 0 since you are following clp sequence
        #
        # -- When d0 == 1, this is a shared edge between the two polygons
        if d0 == 1:  # this is a `cutting` segment traversing the other polygon
            sub0 = cl_n[[i_p, i_c]]
            in_segs.append(sub0)
            # --
            if d1 < 0:  # not common, but accounted for (eg. E, d0_ polys)
                a0_, b0_ = sorted([j_p, j_c])
                bits = _out_p_(j_p, j_c, p_seen, p_outside)
                if len(bits) > 0:
                    sub = np.concatenate((bits, bits[0][None, :]), axis=0)
                    kind_.append(-1)
                if j_p in p_seen:  # or [j_p - 1, j_p] in p_on
                    sub = np.concatenate((sub0, pl_n[[j_p - 1, j_p]]), axis=0)
                # -- could add clip bit here
            # --
            elif d1 == 1:  # unexpected, but accounted for
                sub = []
            # --
            elif d1 > 1:  # poly inside and outside check
                if chk2:  # poly segment outside clp
                    sub1 = _out_p_(j_p, j_c, p_seen, p_outside)
                    in_segs.append(sub0)
                    sub = np.concatenate((sub1, sub0[::-1]), axis=0)
                    kind_.append(0)
                elif chk3:  # ply segment inside clp
                    sub1 = _in_p_(j_p, j_c, p_seen, p_inside)
                    in_segs.append(sub1)
                    sub = np.concatenate((sub0, sub1[::-1]), axis=0)
                    kind_.append(-1)
                if not chk2 and not chk3:  # poly pnts inside and out
                    # sub1 = last_chk(j_p, j_c, p_seen, p_outside)
                    returned = last_chk2(pl_n, j_p, j_c,
                                         p_seen, p_outside, p_inside)
                    out_bits, in_bits, v_seen = returned
                    pp = pl_n[j_p][None, :]  # previous point
                    n0, n1 = len(out_bits), len(in_bits)
                    if n0 > 1:  # -- construct first outside bits
                        bt = np.concatenate(out_bits, axis=0)
                        sub1 = np.concatenate((bt, pp), axis=0)
                        out.append(sub1)
                        kind_.append(-1)
                    elif n0 == 1:
                        sub1 = np.concatenate((bits[0][0], pp), axis=0)
                        out.append(sub1)
                        kind_.append(-1)
                    if n1 > 1:  # -- inside bits
                        bt = np.concatenate(in_bits, axis=0)
                        sub1 = np.concatenate((bt, pp), axis=0)
                        out.append(sub1)
                        kind_.append(1)
                    elif n1 == 1:
                        sub1 = np.concatenate((pp, in_bits[0], pp), axis=0)
                        out.append(sub1)
                        kind_.append(1)
                    #
                    # in_segs.append(sub0)
                    p_seen.extend(v_seen)
                # --
                # do an outside check for the last poly point being inside
                # if j_c in p_inside[0]:
                #     in_bits = pl_n[p_inside.pop(0)]
                #     sub0 = np.concatenate((sub0, in_bits), axis=0)
                #     kind_.append(1)
                #     out.append(sub0)  # -- extra append
        # --
        # -- Note: clip can be inside or outside
        elif d0 > 1:
            if chk0:  # clp seg is outside
                sub0 = _out_c_(i_p, i_c, c_seen, c_outside)
            elif chk1:  # clp seg is inside
                sub0 = _in_c_(i_p, i_c, c_seen, c_inside)
            # --
            if d1 < 0:  # -- applies to E, d0_ because of wrapping crosses
                if chk0:  # clip segment outside ply
                    in_segs.append(sub0)
                    sub0 = sub0[::-1] if len(sub0) > 0 else []
                chk_seen = set([j_p, j_c]).issubset(set(p_seen))  # check seen
                if chk_seen:
                    sub1 = pl_n[j_c:j_p + 1, :][::-1]
                if len(sub0) > 0 and len(sub1) > 0:
                    # print(sub0, sub1)
                    sub = np.concatenate((sub1, sub0), axis=0)
                    kind_.append(2)  # a hole between the two
                else:
                    sub = []  # or _out_bits_
            # --
            if d1 == 1:  # poly ids are sequential, clp is inside or outside
                sub1 = pl_n[[j_p, j_c]]
                if chk0:
                    in_segs.append(sub1)
                    sub = np.concatenate((sub0, sub1[::-1]), axis=0)
                elif chk1:
                    in_segs.append(sub0)
                    sub = np.concatenate((sub1, sub0[::-1]), axis=0)
                kind_.append(-1)
            # --
            elif d1 > 1:  # clp inside and outside check
                if chk0:  # clip segment outside ply, chk3==True
                    sub1 = _in_p_(j_p, j_c, p_seen, p_inside)
                    in_segs.append(sub1)
                    sub1 = sub1[::-1] if len(sub1) > 0 else []
                    kind_.append(-1)
                elif chk1:  # clip segment inside ply, chk2==True?
                    sub1 = _out_p_(j_p, j_c, p_seen, p_outside)
                    in_segs.append(sub0)
                    sub0 = sub0[::-1] if len(sub0) > 0 else []
                    kind_.append(0)
                if len(sub0) > 0 and len(sub1) > 0:
                    # print(sub0, sub1)
                    sub = np.concatenate((sub0, sub1), axis=0)
        if len(sub) > 0:
            out.append(sub)
        # else:
        #     kind_ = kind_[:-1]  # no sub, hence drop last `kind_` assignment
        #
        prev = p
        p_seen.append(j_c)
        c_seen.append(i_c)
        # # --
    final = np.asarray(out, dtype='O')
    clp_poly = np.concatenate([i for i in in_segs if len(i) > 0], axis=0)
    idx = np.array(kind_)
    erase_idx = np.nonzero(idx)[0]
    symm_idx = np.nonzero(idx + 1)[0]
    erase_poly = final[erase_idx]
    symm_poly = final[symm_idx]
    # --
    if as_geo:
        return npg.arrays_to_Geo(final, kind=2, info=None, to_origin=False)
    # return final, [out, subs, dups, pl_n, cl_n, xtras]
    return final, clp_poly, erase_poly, symm_poly


# preP, preC = prePC_out(i0_, i1_, cN, j0_, j1_, pN)  # changed 2023-03-15
# # ---- end from working copy
# # -- assemble
# out, p_seen, c_seen = [], [], []
# #
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
#
# -- make sure first intersection is added and end points renumbered to 0
#
# whr0 = np.nonzero(inCinP[:, 0] == cN)[0]
# whr1 = np.nonzero(inCinP[:, 1] == pN)[0]
# inCinP[whr0, 0] = 0
# inCinP[whr1, 1] = 0


# def _out_bits_(j_p, j_c, p_outside):
#     """Return sub lists."""
#     sub = []
#     for i in [j_p, j_c]:
#         for c0, lst in enumerate(p_outside):
#             if i in lst:
#                 j = p_outside.pop(c0)
#                 sub.append(pl_n[j])
#     return sub

# def prePC_out(i0_, i1_, cN, j0_, j1_, pN):
#     """Determine pre `p` and `c` points."""
#     preP, preC = [], []
#     i1_ = 0 if i1_ in [i1_, cN] else i1_  # clp first/last point check
#     j1_ = 0 if j1_ in [j1_, pN] else j1_  # poly first/last point check
#     #
#     # -- add preceeding pinside points
#     if j0_ > 0 and j1_ < j0_:
#         preP = [m for m in range(j1_, j0_ + 1) if m in p_outside]
#     # -- add preceeding cinside points
#     if i0_ > 0 and i1_ < i0_:
#         preC = [m for m in range(i1_, i0_ + 1) if m in c_outside]
#     return preP, preC


# def _bits_(i0, i1, in_, seen_):
#     """Return indices which are in `in_` and not in `seen_`.

#     Parameters
#     ----------
#     i0, i1 : integers
#     in_, seen_ : list / array
#         These are the values in an `inclusion` being checked if they
#         have not been `seen` yet.

#     Notes
#     -----
#     >>> p_add = [m for m in range(j_p, j_c + 1)
#     ...          if m in ip and m not in p_seen]

#     """
#     r = set(range(i0, i1 + 1))
#     ids = sorted(list(r.intersection(in_).difference(seen_)))
#     return ids


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


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")
