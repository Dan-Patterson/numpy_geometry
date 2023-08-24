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

    Notes
    -----
    The notations `p_p, p_c` refer to the previous and current poly points
    during iterations.  Similarily `c_p, c_c` denote the previous and current
    clipper poly points.
    """
    """
    Create dictionary from list using first value as key

    ky = [i[0] for i in p_out]
    dct = dict(list(zip(ky, p_out)))
    """
    def _in_c_(c_c, c_seen, c_inside):
        """Return sub lists."""
        if len(c_inside) > 0:
            if c_c in c_inside[0]:
                vals = c_inside.pop(0)
                c_seen.extend(vals)
                return cl_n[vals]
        return []

    def _out_c_(c_c, c_seen, c_outside):
        """Return sub lists."""
        if len(c_outside) > 0:
            if c_c in c_outside[0]:
                vals = c_outside.pop(0)
                c_seen.extend(vals)
                return cl_n[vals]
        return []

    def _in_p_(p_c, p_seen, p_inside):
        """Return sub lists."""
        if len(p_inside) > 0:
            if p_c in p_inside[0]:
                vals = p_inside.pop(0)
                p_seen.extend(vals)
                return pl_n[vals]
        return []

    def _out_p_(p_c, p_seen, p_outside):
        """Return sub lists."""
        if len(p_outside) > 0:
            if p_c in p_outside[0]:
                vals = p_outside.pop(0)
                p_seen.extend(vals)
                return pl_n[vals]
        return []

    def _chk_(_p, _c, _case):
        """Boolean check of poly or clip points.

        Parameters
        ----------
        _p, _c : integer
            Previous or current point id values.
        _case : list
            Inside or outside point lists.

        Notes
        -----
        This function is used to see if the previous (`_p`) or current (`_c`)
        poly or clip points are inside or outside their counterpart.  The same
        function can be used for either case.
        """
        for lst in _case:
            if _p in lst and _c in lst:
                return True, lst
        return False, []

    def in_out_chk(_n, _p, _c, _seen, _outside, _inside):
        """Last ditch check in case p_p and p_c are separated by a segment.

        Parameters
        ----------
        parameter meanings

        +------+-------+-----+-----+--------+-----------+----------+
        |      |  _n   | _p  | _c  | _seen  | _outside  | _inside  |
        +======+=======+=====+=====+========+===========+==========+
        | poly |  pl_n | p_p | p_c | p_seen | p_outside | p_inside |
        +------+-------+-----+-----+--------+-----------+----------+
        |clip  | cl_n  | c_p | c_c | c_seen | c_outside | c_inside |
        +------+-------+-----+-----+--------+-----------+----------+

        """
        out_bits = []
        in_bits = []
        pc_max = max([_p, _c]) + 1
        for i in [_p, _c]:
            for cnt_, out_ in enumerate(_outside):
                if i in out_ and pc_max not in out_:  # only take the first out
                    vals = _outside.pop(cnt_)
                    out_bits.append(vals)
            for cnt_, in_ in enumerate(_inside):
                if i in in_ and pc_max not in in_:  # only take the first in
                    vals = _inside.pop(cnt_)
                    in_bits.append(vals)
        return out_bits, in_bits

    def on_pairs(col):
        """Return sequential ids from the intersections not in or out."""
        segs = []
        for cn, v in enumerate(col[1:], 0):
            prev = col[cn]
            dff = v - prev
            if dff == 1:
                segs.append([prev, v])
        return segs

    def cut_pairs(arr):
        """Return cut lines from `onConP` or `id_plcl`."""
        c_segs = []
        p_segs = []
        for cn, v in enumerate(arr[1:, 0], 0):
            prev = arr[cn, 0]
            dff = v - prev
            if dff == 1:
                vals = [arr[cn, 1], arr[cn + 1, 1]]
                c_segs.append([prev, v])
                p_segs.append(vals)
        return c_segs, p_segs

    # -- Returns the intersections, the rolled input polygons, the new polygons
    #    and how the points in both relate to one another.
    result = add_intersections(poly, clp,
                               roll_to_minX=True,
                               polygons=[True, True],
                               class_ids=True)
    pl_n, cl_n, id_plcl, x_pnts, p_out, p_in, c_out, c_in = result
    # --
    # Get the intersections, new polys, points inside and outside and
    # x_pnt ids from `add_intersections`.  Swap the order of the last.
    w0 = np.argsort(id_plcl[:, 1])  # get the order and temporarily sort
    # -- cut lines, where one crosses the other
    onConP = id_plcl[:, [1, 0]][w0]  # slice to rearrange the columns
    # -- two point cut lines, which cross the other polygon
    c_cut0, p_cut0 = cut_pairs(onConP)   # use onConP col 0 since it issorted
    p_cut1, c_cut1 = cut_pairs(id_plcl)  # use id_plcl col 0 to save a sort
    c_cut = c_cut0 + c_cut1  # sorted(c_cut0 + c_cut1, key=lambda l:l[0])
    p_cut = p_cut0 + p_cut1  # sorted(p_cut0 + p_cut1, key=lambda l:l[0])
    # -- cut lines that are more than two points and are either inside or
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
    for cnt, p in enumerate(onConP[1:9], 1):  # enumerate fromonConP[1:]
        c_c, p_c = p       # current ids, this is an intersection point
        c_p, p_p = prev    # previous ids
        d0, d1 = p - prev  # differences in ids
        sub, bits, sub0, sub1 = [], [], [], []
        # --
        chk0, chk1, chk2, chk3 = [False, False, False, False]
        c_out_f = sum(c_outside, [])  # flatten list of sub lists
        c_in_f = sum(c_inside, [])
        p_out_f = sum(p_outside, [])
        p_in_f = sum(p_inside, [])
        if len(c_outside) > 0:
            chk0 = set([c_p, c_c]).issubset(set(c_out_f))
        if len(c_inside) > 0:
            chk1 = set([c_p, c_c]).issubset(set(c_in_f))
        if len(p_outside) > 0:
            chk2 = set([p_p, p_c]).issubset(set(p_out_f))
        if len(p_inside) > 0:
            chk3 = set([p_p, p_c]).issubset(set(p_in_f))
        # d0, d1, chk0, chk1, chk2, chk3
        # --
        # -- d0 clp ids are sequential and are on the densified clp line
        # -- d0 should never be <= 0 since you are following clp sequence
        # -- When d0 == 1, this is a shared edge between the two polygons
        #    it is equivalent to `[c_p, c_c] in c_cut`
        # --
        if d0 == 1:  # this is a `cutting` segment inside `c_cut`
            _clp_ = cl_n[[c_p, c_c]]
            in_segs.append(_clp_)
            # --
            if d1 > 1:  # poly inside and outside check
                r_ = in_out_chk(pl_n, p_p, p_c, p_seen, p_outside, p_inside)
                out_bits, in_bits = r_
                if len(out_bits) > 0:  # -- construct outside bits
                    tmp = sum(out_bits, [])
                    bts = [pl_n[tmp]] + [_clp_[::-1]]
                    out.append(np.concatenate(bts, axis=0))
                    p_seen += tmp
                    kind_.append(0)
                if len(in_bits) > 0:  # -- inside bits
                    tmp = sum(in_bits, [])
                    if p_p == tmp[-1]:  # check to see if it is start or end
                        bts = [pl_n[tmp]] + [_clp_] + [pl_n[tmp[0]][None, :]]
                    else:
                        bts = [_clp_] + [pl_n[tmp[::-1]]]
                    out.append(np.concatenate(bts, axis=0))
                    p_seen += tmp
                    kind_.append(1)
                # if chk2:  # poly segment outside clp
                #     sub1 = _out_p_(p_c, p_seen, p_outside)
                #     # in_segs.append(_clp_)
                #     sub = np.concatenate((sub1, _clp_[::-1]), axis=0)
                #     kind_.append(0)
                # elif chk3:  # ply segment inside clp
                #     sub1 = _in_p_(p_c, p_seen, p_inside)
                #     # in_segs.append(sub1)
                #     sub = np.concatenate((_clp_, sub1[::-1]), axis=0)
                #     kind_.append(-1)
            # --
            elif d1 < 0:  # not common, but accounted for (eg. E, d0_ polys)
                # fix this section
                if p_c + 1 in p_seen:  # closes possible double cut triangle
                    if [p_c, p_c + 1] in p_cut:
                        to_add = pl_n[[p_c, p_c + 1, p_p, p_c]]
                        out.append(to_add)
                bits = _out_p_(max([p_p, p_c]), p_seen, p_outside)
                if len(bits) > 0:
                    s0 = np.concatenate((bits, bits[0][None, :]), axis=0)
                    kind_.append(-1)
                    out.append(s0)
                # -- try this for cnt = 7
                if min([p_p, p_c]) in p_seen:  # or [p_p - 1, p_p] in p_on
                    s1 = np.concatenate((_clp_, pl_n[[p_p - 1, p_p]]), axis=0)
                    out.append(s1)
            # --
            elif d1 == 1:  # unexpected, but accounted for
                sub = []
            # --
        # -- Note: clip can be inside or outside
        elif d0 > 1:
            if chk0:  # clp seg is outside
                sub0 = _out_c_(c_c, c_seen, c_outside)
                in_segs.append(sub0)  # 5-9
            elif chk1:  # clp seg is inside
                sub0 = _in_c_(c_c, c_seen, c_inside)
                in_segs.append(sub0)
            # --
            if d1 < 0:  # -- applies to E, d0_ because of wrapping crosses
                if chk0:  # clip segment outside ply
                    sub0 = sub0[::-1] if len(sub0) > 0 else []  # ???
                    sub1 = pl_n[p_c:p_p + 1, :][::-1]
                if len(sub0) > 0 and len(sub1) > 0:
                    sub = np.concatenate((sub1, sub0), axis=0)
                    kind_.append(2)  # a hole between the two
                else:
                    sub = []  # or _out_bits_
            # --
            if d1 == 1:  # poly ids are sequential, clp is inside or outside
                sub1 = pl_n[[p_p, p_c]]
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
                    sub1 = _in_p_(p_c, p_seen, p_inside)
                    if len(sub1) > 0:
                        in_segs.append(sub1)
                        sub1 = sub1[::-1] if len(sub1) > 0 else []
                        kind_.append(-1)
                elif chk1:  # clip segment inside ply, chk2==True?
                    sub1 = _out_p_(p_c, p_seen, p_outside)
                    # in_segs.append(sub0)
                    if len(sub1) > 0:
                        sub0 = sub0[::-1] if len(sub0) > 0 else []
                        kind_.append(0)
                if len(sub0) > 0 and len(sub1) > 0:
                    sub = np.concatenate((sub0, sub1), axis=0)
        if len(sub) > 0:
            out.append(sub)
        # else:
        #     kind_ = kind_[:-1]  # no sub, hence drop last `kind_` assignment
        #
        val_lst = [cnt, prev, p, d0, d1, chk0, chk1, chk2, chk3, out]
        print("cnt {}: {} {} {} {} {} {} {} {} \n   {}".format(*val_lst))
        prev = p
        p_seen.append(p_c)
        c_seen.append(c_c)
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
    """Determine post `p` and `c` points."""
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
