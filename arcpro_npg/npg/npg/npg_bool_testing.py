# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
"""
Created on Sun Jan 14 16:52:22 2024

@author: dan_p
"""
# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121

import sys
import copy
import numpy as np
import npg  # noqa
from npg import npGeo
# from npg.npg_pip import np_wn
from npg.npg_bool_hlp import add_intersections, _del_seq_pnts_  # p_ints_p
from npg.npg_geom_hlp import _bit_check_  # radial_sort
from npg.npg_prn import prn_  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)


def _cut_pairs_(arr):
    """Return cut lines from `onConP` or `id_plcl`."""
    c_segs = []
    p_segs = []
    idxs = arr[:, 0]
    prev = idxs[0]
    for cn, v in enumerate(idxs[1:], 0):
        dff = v - prev
        if dff == 1:
            vals = [arr[cn, 1], arr[cn + 1, 1]]
            c_segs.append([prev, v])
            p_segs.append(vals)
        prev = v
    return c_segs, p_segs


def overlay_ops(poly, clp, as_geo=True):
    """Return various overlay results between two polygons, `poly`, `clp`.

    Parameters
    ----------
    poly, clp : array_like
        `poly` is the polygon being differenced by polygon `clp`

    Returns
    -------
    erase, reverse erase, symmetrical difference, holes, clip (both orders)

    Requires
    --------
    `npg_bool_hlp` : `add_intersections`, `_del_seq_pnts_`

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
                return vals
        return []

    def _out_c_(c_c, c_seen, c_outside):
        """Return sub lists."""
        if len(c_outside) > 0:
            if c_c in c_outside[0]:
                vals = c_outside.pop(0)
                c_seen.extend(vals)
                return vals
        return []

    def _in_p_(p_c, p_seen, p_inside):
        """Return sub lists."""
        if len(p_inside) > 0:
            if p_c in p_inside[0]:
                vals = p_inside.pop(0)
                p_seen.extend(vals)
                return vals
        return []

    def _out_p_(p_c, p_seen, p_outside):
        """Return sub lists."""
        if len(p_outside) > 0:
            if p_c in p_outside[0]:
                vals = p_outside.pop(0)
                p_seen.extend(vals)
                return vals
        return []

    def _io_(_n, _p, _c, _seen, _outside, _inside):
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
        # pc_max = max([_p, _c]) + 1
        for i in [_p, _c]:
            for cnt_, out_ in enumerate(_outside):
                if i in out_:  # and pc_max not in out_:  # take first out
                    vals = _outside.pop(cnt_)
                    out_bits.append(vals)
                    _seen += vals
                    continue
            for cnt_, in_ in enumerate(_inside):
                if i in in_:  # and pc_max not in in_:  # take the first in
                    vals = _inside.pop(cnt_)
                    in_bits.append(vals)
                    _seen += vals
                    continue
        return out_bits, in_bits

    def _op_(_p, _c, _seen, _out):
        """Return multiple out bits."""
        out_bits = []
        for i in [_p, _c]:
            for cnt_, out_ in enumerate(_out):
                if i in out_ and _c not in _seen:  # conditions  !!!!
                    vals = _out.pop(cnt_)
                    out_bits.append(vals)
                    _seen += vals
                    # continue
        out_bits = sum(out_bits, [])  # flatten subarrays here
        return out_bits

    def _ip_(_p, _c, _seen, _in):
        """Return multiple in bits."""
        in_bits = []
        for i in [_p, _c]:
            for cnt_, in_ in enumerate(_in):
                if i in in_ and _c not in _seen:  # conditions
                    vals = _in.pop(cnt_)
                    in_bits.append(vals)
                    _seen += vals
                    # continue
        in_bits = sum(in_bits, [])  # flatten subarrays here
        return in_bits

    def _in_(_c, _seen, _inside):
        """Return sub lists."""
        if len(_inside) > 0:
            if _c in _inside[0]:
                vals = _inside.pop(0)
                _seen.extend(vals)
                return vals
        return []

    def _out_(_c, _seen, _outside):
        """Return sub lists."""
        if len(_outside) > 0:
            if _c in _outside[0]:
                vals = _outside.pop(0)
                _seen.extend(vals)
                return vals
        return []

    # --
    # -- Get the intersections, new polys, points inside and outside and
    #    the rolled input polygons, the new and how the points in both
    #    relate to one another.  Swap the order of the last.
    #
    testing = True
    #
    result = add_intersections(poly, clp,
                               roll_to_minX=True,
                               p0_pgon=True,
                               p1_pgon=True,
                               class_ids=True)
    pl_n, cl_n, id_plcl, onConP, x_pnts = result[:5]
    p_outside, p_inside, c_outside, c_inside = result[5:]
    # --
    # -- cut lines, where one crosses the other
    # -- two point cut lines, which cross the other polygon
    #    cut lines that are more than two points are either inside or
    #    outside the other polygon
    c_cut0, p_cut0 = _cut_pairs_(onConP[:, :2])  # use onConP, it is sorted
    p_cut1, c_cut1 = _cut_pairs_(id_plcl)  # use id_plcl col 0 to save a sort
    c_cut = c_cut0 + c_cut1  # sorted(c_cut0 + c_cut1, key=lambda l:l[0])
    p_cut = p_cut0 + p_cut1  # sorted(p_cut0 + p_cut1, key=lambda l:l[0])
    #
    c_cut_out = [i for i in c_cut if i[1] - i[0] != 1]
    p_cut_out = [i for i in p_cut if i[1] - i[0] != 1]
    #
    # -- individual points that are outside of their other shape
    # cs = onConP[:, 0]  # clp point ids
    # N_cl = np.arange(0, cl_n.shape[0])
    # co_pnts = sorted(list(set(N_cl).difference(cs)))
    #
    # ps = onConP[:, 1]  # poly point ids
    # N_pl = np.arange(0, pl_n.shape[0])
    # po_pnts = sorted(list(set(N_pl).difference(ps)))
    #
    # -- Copy the outside
    p_out = copy.deepcopy(p_outside)
    p_in = copy.deepcopy(p_inside)
    c_out = copy.deepcopy(c_outside)
    c_in = copy.deepcopy(c_inside)
    #
    #  Determine preceeding points to first clip.
    out = []
    prev = onConP[0, :2]  # -- set the first `previous` for enumerate
    p_seen, c_seen = [], []
    in_clp = []  # collect `clipping` segments to use for clip.
    kind_ = []  # see below
    # nP = N_pl[-1]  # size of poly array
    # nC = N_cl[-1]  # size of clp array
    # -1 clip outside poly :  symmetrical diff = -1 & 0
    #  0 poly outside of clip : erase
    #  1 poly in clip : clip  : both inside
    #  2 hole  : neither
    #  3 concave hull inside clp
    #  9 fix!!
    if testing:
        print("onConP\n{}".format(onConP))
        print("\np_outside : {}".format(p_outside))
        print("p_inside  : {}".format(p_inside))
        print("c_outside : {}".format(c_outside))
        print("c_inside  : {}".format(c_inside))
    msg = []
    fmt = "{:>3}{:>3}{:>3}{:>3} :{:>3}{:>3} {!s:<6}{!s:<6}{!s:<6}{!s:<6}"
    for cnt, row in enumerate(onConP[1:], 1):  # enumerate from onConP[1:]
        # current ids and differences... this is an intersection point
        c_c, p_c, d0, d1 = row  # row[:2], row[2], row[3]
        c_p, p_p = prev    # previous ids
        bts, sub0, sub1 = [], [], []
        # --
        chk0, chk1, chk2, chk3 = [False, False, False, False]
        c_out_f = sum(c_out, [])  # flatten list of sub lists
        c_in_f = sum(c_in, [])
        p_out_f = sum(p_out, [])
        p_in_f = sum(p_in, [])
        #
        # chk0 = c_c in c_out_f
        # chk1 = c_c in c_in_f
        # chk2 = p_c in p_out_f
        # chk3 = p_c in p_in_f
        if len(c_out) > 0:  # also : set([c_p, c_c]).issubset(set(c_out_f))
            chk0 = c_p in c_out_f and c_c in c_out_f  # is faster
        if len(c_in) > 0:
            chk1 = c_p in c_in_f and c_c in c_in_f
        if len(p_out) > 0:
            chk2 = p_p in p_out_f and p_c in p_out_f
        if len(p_in) > 0:
            chk3 = p_p in p_in_f and p_c in p_in_f
        t = [c_p, c_c, p_p, p_c, d0, d1, chk0, chk1, chk2, chk3]
        msg.append(fmt.format(*t))
        # d0, d1, chk0, chk1, chk2, chk3
        # --
        # -- d0 clp ids are sequential and are on the densified clp line
        # -- d0 will never be <= 0 since you are following clp sequence
        # -- When d0 == 1, this is a shared edge between the two polygons
        #    it is equivalent to `[c_p, c_c] in c_cut`
        # --
        if d0 == 1:  # this is a `cutting` segment inside `c_cut`
            _clp_ = cl_n[[c_p, c_c]]
            if chk2:                   # poly outside, use _clp_
                in_clp += [_clp_]
            elif chk3:                 # poly inside _clp_
                in_clp += [pl_n[p_p: p_c + 1]]  # p_cut_out possibly
            elif [c_p, c_c] in c_cut:  # when ch2 and ch3 are False
                in_clp += [_clp_]      # add the cut line
            # --
            # Create the various polygons once clipping segment is created
            if d1 > 1:  # inside and outside check
                if chk2:  # poly bits outside
                    tmp = _out_(p_c, p_seen, p_out)  # -- new  _out_p_
                    bts = [pl_n[tmp]] + [_clp_[::-1]]
                    out.append(np.concatenate(bts, axis=0))
                    kind_.append(0)
                    # check to see if p_c is also inside  E, d0_ first case
                    if len(p_in_f) > 0:
                        tmp = _in_(p_c, p_seen, p_in)
                        if len(tmp) > 0:  # p_p, p_c in p_cut
                            bts = [_clp_] + [pl_n[tmp]] + [_clp_[0][None, :]]
                            out.append(np.concatenate(bts, axis=0))
                            kind_.append(1)
                elif chk3:  # -- poly inside bits
                    # tmp = _in_p_(p_c, p_seen, p_in)
                    tmp = _in_(p_c, p_seen, p_in)  # p01, p02 test
                    if tmp[-1] - tmp[0] == 1:  # edgy1-eclip last  [76,77]
                        bts = []
                    elif p_p == tmp[-1]:  # check if it is start or end
                        bts = [pl_n[tmp]] + [_clp_] + [pl_n[tmp[0]][None, :]]
                    else:
                        bts = [_clp_] + [pl_n[tmp[::-1]]]
                    if len(bts) > 0:  # diff > 1
                        out.append(np.concatenate(bts, axis=0))
                        kind_.append(1)
                elif not chk2 and not chk3:  # empty
                    added = False
                    if len(p_out) > 0:  # try to use p_out's last entry
                        p_chk = p_out[-1]
                        c_chk = sum([i for i in c_cut_out if c_c in i], [])
                        if c_c in c_chk or p_c + 1 in p_chk:
                            # chk = p_out.pop(-1)  # pop or slice
                            seg = pl_n[p_chk]
                            tmp = np.concatenate(in_clp, axis=0)
                            tmp = np.concatenate((tmp, seg), axis=0)
                            p_seen += p_chk
                            kind_.append(1)
                            out.append(tmp)
                            added = True
                    if len(c_out) > 0 and not added:  # try c_out's last
                        c_chk = c_out[-1]
                        p_chk = sum([i for i in p_cut_out if p_c in i], [])
                        if p_c in p_chk or c_c + 1 in c_chk:
                            # chk = c_out.pop(-1)
                            seg = cl_n[c_chk]
                            tmp = np.concatenate(in_clp, axis=0)
                            tmp = np.concatenate((tmp, seg), axis=0)
                            p_seen += c_chk
                            kind_.append(1)
                            out.append(tmp)
            # --
            elif d1 < 0:  # not common, but accounted for (eg. E, d0_ polys)
                # check to see if both are cut lines, may be a triangle
                if [c_p, c_c] in c_cut and [c_c, c_c + 1] in c_cut:
                    bts = cl_n[[c_p, c_c, c_c + 1, c_p]]
                    kind_.append(1)
                    out.append(bts)
                    # -- add c_c + 1 to c_seen???
                # -- below should be the same, so it will be skipped
                elif [p_p, p_c] in p_cut and [p_c, p_c + 1] in p_cut:
                    bts = pl_n[[p_p, p_c, p_c + 1, p_p]]
                    kind_.append(1)
                    out.append(bts)
                else:
                    print("something missed")
                # pick up outside
                if [p_c - 1, p_c] in p_cut:
                    if p_p in p_out_f:
                        # p_p should be in p_out
                        tmp = _out_(p_c, p_seen, p_out)
                        ids = [p_c - 1, p_c] + tmp + [p_c - 1]
                        bts = pl_n[ids]
                        kind_.append(-1)  # outside clp
                        out.append(bts)
                        p_seen += tmp
                # may be other parts to test
            # -- if d0, d1 = 1,1 then both lines are the same
            elif d1 == 1:
                print("cnt {} row {} d0 {}, d1 {}".format(cnt, row, d0, d1))
        # --
        # -- Note: clip can be inside or outside
        elif d0 > 1:
            if chk0:  # clp seg is outside
                # ids = _out_c_(c_c, c_seen, c_out)
                ids = _out_(c_c, c_seen, c_out)  # p01, p02 test
                sub0 = cl_n[ids]
                #  in_clp.append(sub0)  # moved to below
            elif chk1:  # clp seg is inside
                # ids = _in_c_(c_c, c_seen, c_in)
                ids = _ip_(c_p, c_c, c_seen, c_in)  # p01, p02 test
                sub0 = cl_n[ids]
                # add in_clp here ???
            # --
            if d1 < 0:  # -- applies to E, d0_ because of wrapping crosses
                # -- in_clp handled previously
                if chk0:  # clip segment outside ply
                    in_clp.append(sub0)
                    sub0 = sub0[::-1] if len(sub0) > 0 else []
                    ids = np.arange(p_c, p_p + 1).tolist()
                    sub1 = pl_n[ids, :][::-1]
                    p_seen += ids  # pop ids?????
                if len(sub1) > 0:  # len(sub0) > 0
                    bts = np.concatenate((sub1, sub0), axis=0)
                    out.append(bts)
                    kind_.append(2)  # a hole between the two
            # --
            elif d1 == 1:  # poly ids are sequential, clp  inside or outside
                sub1 = pl_n[[p_p, p_c]]
                if chk0:  # clip outside
                    in_clp += [sub1]  # d02, c03
                    bts = np.concatenate((sub0, sub1[::-1]), axis=0)
                    out.append(bts)
                    kind_.append(-1)
                elif chk1:  # clip inside
                    in_clp += [sub0]  # add in_clp here
                    bts = np.concatenate((sub1, sub0[::-1]), axis=0)
                    out.append(bts)
                    kind_.append(0)  # ?? check
            # --
            elif d1 > 1:  # clp : inside and outside check
                if chk0:  # clip segment outside ply. Note may be: chk3==True
                    # ids = _in_p_(p_c, p_seen, p_in))  # p01, p02 test
                    ids = _out_(p_c, p_seen, p_out)
                    if len(ids) > 0:
                        sub1 = pl_n[ids]
                        in_clp += [sub1]  # ply inside and is the clip edge
                        sub1 = sub1[::-1]
                        kind_.append(1)  # clip outside,
                elif chk1:  # clip segment inside ply, chk2==True?
                    in_clp += [sub0]
                    # ids = _out_p_(p_c, p_seen, p_out)
                    ids = _out_(p_c, p_seen, p_out)
                    sub1 = pl_n[ids]
                    if len(sub1) > 0:
                        sub0 = sub0[::-1] if len(sub0) > 0 else []
                        kind_.append(0)
                # -- put the pieces together
                if len(sub1) > 0:  # len(sub0) > 0 and sub0 is always > 0
                    bts = np.concatenate((sub0, sub1), axis=0)
                    out.append(bts)
                    kind_.append(-9)  # -- placeholder !!! fix
                # -- last one is for p02, c03
                elif len(sub1) == 0:  # if the last, sub0 > 0 always
                    # close outside
                    bts = np.concatenate((sub0, sub0[None, 0]), axis=0)
                    out.append(bts)
                    kind_.append(-1)
                    # close inside
                    bts = np.concatenate(in_clp, axis=0)
                    bts = np.concatenate((bts, bts[None, 0]), axis=0)
                    out.append(bts)
                    kind_.append(0)
        #
        prev = [c_c, p_c]
        p_seen.append(p_c)
        c_seen.append(c_c)
        # # --
    # -- keep in_clp_ out of final
    if len(in_clp) > 0:
        clp_ = np.concatenate(in_clp, axis=0)  # intersect as well
        clp_ = _del_seq_pnts_(clp_, True)      # this seems to be
        if (clp_[0] != clp_[-1]).any():
            clp_ = np.concatenate((clp_, clp_[0][None, :]), axis=0)
        # kind_.append(1)  # or 3 if concave hull in c_in check below
        # out.append(clp_)
    #
    final = np.asarray(out, dtype='O')
    #
    idx = np.array(kind_)
    idx_hole = np.nonzero(idx == 2)[0]    # holes, 2
    idx_all = np.nonzero(idx < 2)[0]      # symmetrical difference, -1, 0, 1
    idx_p_out = np.nonzero(idx == 0)[0]   # pairwise erase, 0
    idx_c_out = np.nonzero(idx == -1)[0]  #
    # idx_p_in = np.nonzero(idx == 1)[0]
    idx_c_in = np.nonzero(idx == 1)[0]    # reverse pairwise erase, 1
    #
    hole_ = final[idx_hole] if len(idx_hole) > 0 else []
    symm_ = final[idx_all]
    cin_ = final[idx_c_in] if len(idx_c_in) > 0 else []  # clp times E,d0_
    hull_ = []  # -- hull
    if cin_.dtype == 'O':
        tmp = _del_seq_pnts_(cin_[-1], True)
        if clp_.shape == tmp.shape:
            chk = np.equal(clp_, tmp).all()
            if chk:
                hull_ = tmp
                z = np.asarray(cin_[:-1], dtype='O')
                # print("chk {}\n{}".format(chk, z))
                cin_ = z
                idx[-1] = 3
                idx_c_in = np.nonzero(idx == 1)[0]
                cin_ = final[idx_c_in]
    erase_ = final[idx_p_out] if len(idx_p_out) > 0 else []
    erase_r = final[idx_c_out] if len(idx_c_out) > 0 else []
    # --
    # if as_geo:
    #     return npg.arrays_to_Geo(final, kind=2, info=None, to_origin=False)
    # erase_, clp_0, clp_1, hole_, symm_, erase_r
    if testing:
        print("\nidx : {}\n".format(idx))
        print("cc  cp pc pp   d0 d1 chk0  chk1  chk2  chk3")
        print("\n".join([i for i in msg]))
    return final, idx, clp_, cin_, hole_, symm_, erase_, erase_r, hull_

'''
    def _in_c_(c_c, c_seen, c_in):
        """Return sub lists."""
        if len(c_in) > 0:
            if c_c in c_in[0]:
                vals = c_in.pop(0)
                # c_seen.extend(vals)
                return vals  # cl_n[vals]
        return []

    def _out_c_(c_c, c_seen, c_out):
        """Return sub lists."""
        if len(c_out) > 0:
            if c_c in c_out[0]:
                vals = c_out.pop(0)
                # c_seen.extend(vals)
                return vals  # cl_n[vals]
        return []

    def _in_p_(p_c, p_seen, p_in):
        """Return sub lists."""
        if len(p_in) > 0:
            if p_c in p_in[0]:
                vals = p_in.pop(0)
                # p_seen.extend(vals)
                return vals  # pl_n[vals]
        return []

    def _out_p_(p_c, p_seen, p_out):
        """Return sub lists."""
        if len(p_out) > 0:
            if p_c in p_out[0]:
                vals = p_out.pop(0)
                # p_seen.extend(vals)
                return vals  # pl_n[vals]
        return []
'''
