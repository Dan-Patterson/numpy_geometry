# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
------------
npg_bool_ops
------------

Modified :
    2024-01-24

** Boolean operations on poly geometry.

In npg_clip_split.

- clip
- split

Here.

- clip
- erase
- symmetrical difference

Other ops.

- union

- A and B, A not B, B not A
- A union B (OR)
- A intersect B (AND)
- A XOR B
----

Script :
    npg_bool_ops.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

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

script = sys.argv[0]

__all__ = [
    'clp_', 'erase_', 'symm_diff_', 'overlay_ops',
    'append_',
    'adjacency_matrix',
    'merge_',
    'union_adj'
]   # 'dissolve'

__helpers__ = ['_adjacent_', '_cut_pairs_']
__imports__ = [
    'add_intersections', '_del_seq_pnts_', 'p_ints_p',
    '_bit_check_', 'remove_geom', 'radial_sort'
]


# ---- helpers
#
def _adjacent_(a, b):
    """Check adjacency between 2 polygon shapes on their edges.

    Parameters
    ----------
    a, b : ndarrays
        The arrays of coordinates to check for polygon adjacency.
        The duplicate first/last point is removed so that a count will not
        flag a point as being a meeting point between polygons.

    Note
    ----
    Adjacency is defined as meeting at least 1 point.  Further checks will be
    needed to assess whether two shapes meet at 1 point or 2 non-consequtive
    points.
    """
    s = np.sum((a[:, None] == b).all(-1).any(-1))
    if s > 0:
        return True
    return False


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


# ---- (1) Operations for polygon overlays.
#
def clp_(poly, clp, as_geo=True):
    """Return the symmetrical difference.  See `overlay_ops`."""
    result = overlay_ops(poly, clp)
    final, idx, clipped, cin_, hole_, symm_, erase_, erase_r, hull_ = result
    if as_geo:
        return npg.arrays_to_Geo(clipped, kind=2, info=None, to_origin=False)
    return clipped


def erase_(poly, clp, as_geo=True):
    """Return the symmetrical difference.  See `overlay_ops`."""
    result = overlay_ops(poly, clp)
    final, idx, clipped, cin_, hole_, symm_, erase_, erase_r, hull_ = result
    if as_geo:
        return npg.arrays_to_Geo(erase_, kind=2, info=None, to_origin=False)
    return erase_


def symm_diff_(poly, clp, as_geo=True):
    """Return the symmetrical difference.  See `overlay_ops`."""
    result = overlay_ops(poly, clp)
    final, idx, clipped, cin_, hole_, symm_, erase_, erase_r, hull_ = result
    if as_geo:
        return npg.arrays_to_Geo(symm_, kind=2, info=None, to_origin=False)
    return symm_


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

    def _out_1_(_c, _seen, _out):
        """Return sub lists."""
        if len(_out) > 0:
            if _c in _out[0]:
                vals = _out.pop(0)
                # c_seen.extend(vals)
                return vals  # cl_n[vals]
        return []

    def _in_1_(_c, _seen, _in):
        """Return sub lists."""
        if len(_in) > 0:
            if _c in _in[0]:
                vals = _in.pop(0)
                # c_seen.extend(vals)
                return vals  # cl_n[vals]
        return []

    def trichk(c_p, c_c, p_p, p_c):
        """Check and classify triangles."""
        tchk = False
        ids = []
        knd = 9
        t0 = [c_c, c_c + 1] in c_cut
        t1 = [c_c + 1, c_p] in c_cut
        t2 = [c_p, c_c + 1] in c_cut
        tchk = t0 and (t1 or t2)
        if tchk:
            if t1:
                ids = [c_p, c_c + 1, c_c, c_p]
                knd = 2
            else:
                ids = [c_p, c_c, c_c + 1, c_p]
                knd = 1
        return tchk, ids, knd

    def _op_(_p, _c, _seen, _out):
        """Return multiple out bits."""
        _bits, vals, _pop_, frst = [], [], [], []  # initialize
        if len(_out) == 0:
            return vals, _pop_, _bits
        for i in [_p, _c]:
            for cnt_, out_ in enumerate(_out):
                if i in out_:  # and _c not in _seen:  conditions  !!!!
                    vals = _out[cnt_]  # .pop(cnt_)
                    if vals not in _bits:
                        _bits.append(vals)
        if len(_bits) == 0:
            return [], [], []
        frst = _bits[0]
        if len(_bits) > 1:
            e = _bits[0][-1]
            s = _bits[1][0]
            if e != s:
                vals = sum(_bits, [])  # flatten subarrays here
            else:
                vals = _bits[0]
        if frst in _out:
            idx = _out.index(frst)
            _pop_ = _out.pop(idx)  # _in[0]  # drop the first in out
        return vals, _pop_, _bits

    def _ip_(_p, _c, _seen, _in):
        """Return multiple in bits."""
        _bits, vals, _pop_, frst = [], [], [], []  # initialize
        if len(_in) == 0:
            return vals, _pop_, _bits
        for i in [_p, _c]:
            for cnt_, in_ in enumerate(_in):
                if i in in_:  # and _c not in _seen:  conditions
                    vals = _in[cnt_]  # .pop(cnt_)
                    if vals not in _bits:
                        _bits.append(vals)
        if len(_bits) == 0:
            return [], [], []
        frst = _bits[0]
        if len(_bits) > 1:
            e = _bits[0][-1]
            s = _bits[1][0]
            if e != s:
                vals = sum(_bits, [])  # flatten subarrays here
            else:
                vals = _bits[0]
        if frst in _in:
            idx = _in.index(frst)
            _pop_ = _in.pop(idx)  # _in[0]  # drop the first in out
        return vals, _pop_, _bits

    # -- Returns the intersections, the rolled input polygons, the new polygons
    #    and how the points in both relate to one another.
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
    # -- cut lines that are more than two points are either inside or
    #    outside the other polygon
    p_out = copy.deepcopy(p_outside)
    p_in = copy.deepcopy(p_inside)
    c_out = copy.deepcopy(c_outside)
    c_in = copy.deepcopy(c_inside)
    #
    #  Determine preceeding points to first clip.
    out = []  # p_seen, c_seen = [], [], []
    prev = onConP[0, :2]  # -- set the first `previous` for enumerate
    p_seen, c_seen = [0], [0]
    in_clp = []  # collect `clipping` segments to use for clip.
    kind_ = []  # see below
    # -1 symmetrical diff : feature areas are outside one another
    #  0 erase : poly area outside of clip
    #  1 clip  : poly areas that are clipped by clp
    #  2 hole  : boundary area between clip and poly segments
    #  3 identity : features or parts that overlap
    if testing:
        print("onConP\n{}".format(onConP))
        print("\np_outside : {}".format(p_outside))
        print("p_inside  : {}".format(p_inside))
        print("c_outside : {}".format(c_outside))
        print("c_inside  : {}".format(c_inside))
    msg = []
    fmt = "{:>3}"*4 + ":" + "{:>3}"*2 + "  " + "{!s:<6}"*6
    # --
    # -------------------------------------------------------------------
    for cnt, row in enumerate(onConP[1:], 1):  # enumerate fromonConP[1:]
        # current ids and differences... this is an intersection point
        c_c, p_c, d0, d1 = row  # row[:2], row[2], row[3]
        c_p, p_p = prev    # previous ids
        bts, sub, sub0, sub1, _, __ = [], [], [], [], [], []
        # --
        chk0, chk1, chk2, chk3 = [False, False, False, False]
        c_out_f = sum(c_out, [])  # flatten list of sub lists
        c_in_f = sum(c_in, [])
        p_out_f = sum(p_out, [])
        p_in_f = sum(p_in, [])
        #
        chk0 = c_p in c_out_f and c_c in c_out_f
        chk1 = c_p in c_in_f and c_c in c_in_f
        chk2 = p_p in p_out_f and p_c in p_out_f
        chk3 = p_p in p_in_f and p_c in p_in_f
        #
        chk4 = c_c in c_out_f  # c_p in c_out_f and c_p in c_in_f
        chk5 = p_c in p_out_f
        #
        t = [c_p, c_c, p_p, p_c, d0, d1, chk0, chk1, chk2, chk3, chk4, chk5]
        msg.append(fmt.format(*t))
        # d0, d1, chk0, chk1, chk2, chk3, chk4, chk5
        # --
        # -- d0 clp ids are sequential and are on the densified clp line
        # -- d0 should never be <= 0 since you are following clp sequence
        # -- When d0 == 1, this is a shared edge between the two polygons
        #    it is equivalent to `[c_p, c_c] in c_cut`
        # --
        # -------------------------------------------------------------------
        #
        if d0 == 1:  # this is a `cutting` segment inside `c_cut`
            _clp_ = cl_n[[c_p, c_c]]
            if chk2:                   # poly outside, use _clp_
                in_clp += [_clp_]
            elif chk3:                 # poly inside _clp_
                in_clp += [pl_n[p_p: p_c + 1]]  # p_cut_out possibly
            elif [c_p, c_c] in c_cut:  # when ch2 and ch3 are False
                in_clp += [_clp_]      # add the cut line
            # --
            if d1 > 1:  # poly inside and outside check
                p_chk = [p_p, p_c] in p_cut  # p_p, p_c may be cut line
                # c_p, c_c is a cut, so check for triangular cuts
                tri_chk, ids, knd = trichk(c_p, c_c, p_p, p_c)
                # tri_chk = [c_c, c_c + 1] in c_cut and [c_c + 1, c_p] in c_cut
                #
                # -- check for inside poly points
                if p_chk and chk3:  # poly inside, clip is an edge
                    in_ids, _, __ = _ip_(p_p, p_c, p_seen, p_in)
                    if len(in_ids) > 0:
                        bts = pl_n[in_ids[::-1]]
                        out.append(np.concatenate((_clp_, bts), axis=0))
                        p_seen += in_ids
                        if chk5:             # hole, if p_chk, chk3 and chk5
                            kind_.append(2)
                        else:
                            kind_.append(1)  # inside
                # -- handle possible triangles
                elif p_chk and tri_chk:  # form the triangle
                    ids = [c_p, c_c, c_c + 1, c_p]
                    bts = cl_n[ids]
                    out.append(bts)
                    c_seen += [c_p, c_c]
                    kind_.append(1)
                    # put chk2 here?????
                    if chk2:
                        a_, b_ = min([p_p, p_c]), max([p_p, p_c])
                        out_ids, _, __ = _op_(a_, b_, p_seen, p_out)
                        keep_ = [i for i in out_ids if i >= a_ and i <= b_]
                        ids = sorted(list(set(keep_)))
                        bts = np.concatenate(
                                (pl_n[ids], pl_n[[ids[0]]]), axis=0)
                        out.append(bts)
                        p_seen += ids
                #
                # -- handle in and out bit
                elif chk2 and chk5:  # to handle E d0_ first clip
                    out_ids, _, __ = _op_(p_p, p_c, p_seen, p_out)
                    if len(out_ids) > 0:
                        tmp = pl_n[out_ids + [p_p]]
                        p_seen += out_ids[:-1]
                        out.append(tmp)
                        kind_.append(0)  # poly outside
                    # --
                    if not chk4:  # pl_, cl_ has chk4 and chk5 leading values
                        in_ids, _, __ = _ip_(p_p, p_c, p_seen, p_in)
                        if len(in_ids) > 0:
                            tmp = pl_n[[p_p] + in_ids + [p_p]]
                            p_seen += in_ids[:-1]
                            out.append(tmp)
                            kind_.append(1)
                        # -- update _seen lists first
                    #
                elif chk2:
                    print("{} {} chk2 bit using _out_1_".format(d0, d1))
                    ids = _out_1_(p_c, p_seen, p_out)
                    tmp = pl_n[ids]
                    out.append(np.concatenate((tmp, _clp_), axis=0))
                elif chk3:
                    print("{} {} chk2 bit using in_1_".format(d0, d1))
                    ids = _in_1_(p_c, p_seen, p_in)
                    tmp = pl_n[ids]
                    out.append(np.concatenate((tmp, _clp_[::-1]), axis=0))
            # --
            elif d1 < 0:  # not common, but accounted for (eg. E, d0_ polys)
                # check for inside points, perhaps a triangle
                p_chk = [p_p, p_c] in p_cut  # p_p, p_c may be cut line
                c_chk = [c_c, c_c + 1] in c_cut  # c_p, c_c d0 = 1, is a cut
                if p_chk and c_chk:
                    ids = [c_p, c_c, c_c + 1, c_p]
                    bts = cl_n[ids]
                    out.append(bts)
                    c_seen += [c_p, c_c]
                    kind_.append(1)
                elif p_chk and [c_p, c_c] in c_cut:  # both are cut lines
                    c_seen.append(c_c)
                    p_seen.append(p_c)
                #
                # This will be true:  p_p > p_c:
                # check for seen points
                else:
                    a_, b_ = min([p_p, p_c]), max([p_p, p_c])
                    rng = list(range(a_, b_ + 1))
                    all_seen = set(rng).issubset(set(p_seen))
                    if not all_seen:
                        out_ids, _, __ = _op_(a_, b_, p_seen, p_out)
                        keep_ = [i for i in out_ids if i >= a_ and i <= b_]
                        # keep_ should now equal _pop_
                        if len(keep_) > 0:
                            # in_clp.extend([])  # fix !!!
                            bts = pl_n[keep_]
                            s0 = np.concatenate((bts, bts[0][None, :]), axis=0)
                            kind_.append(0)  # was -1 which is wrong
                            p_seen += keep_
                            out.append(s0)
            # --
            elif d1 == 1:  # unexpected, get in and out at once
                if chk5:
                    # -- get inside bit
                    in_ids, _, __ = _ip_(c_p, c_c, c_seen, c_in)
                    if len(in_ids) > 0:
                        bts = cl_n[in_ids]
                        c_seen += in_ids
                        kind_.append(1)
                        out.append(np.concatenate((_clp_, bts), axis=0))
                    # -- get outside bit
                    out_ids, _, __ = _op_(0, 1, p_seen, p_out)
                    if len(out_ids) > 0:
                        bts2 = pl_n[out_ids]
                        p_seen += out_ids
                        kind_.append(0)
                        out.append(np.concatenate((bts[::-1], bts2), axis=0))
            # --
        # -------------------------------------------------------------------
        # -- Note: clip can be inside or outside
        elif d0 > 1:
            if chk0:  # chk4 as well  ... clp seg outside
                out_ids, _, __ = _op_(c_p, c_c, c_seen, c_out)
                sub0 = cl_n[out_ids]
                c_seen += out_ids
                if out_ids[-1] == cl_n.shape[0] - 1:  # check for last outside
                    out.append(cl_n[out_ids + [out_ids[0]]])
                # in_clp added later
            elif chk1:  # clp seg is inside
                in_ids, _, __ = _ip_(c_p, c_c, c_seen, c_in)
                sub0 = cl_n[in_ids]
                c_seen += in_ids
                in_clp += [sub0]
                # polygons touch at 1 point, collect clip area
                if (sub0[0] == sub0[-1]).any().all():
                    out.append(sub0)
                    kind_.append(1)
            # elif chk2:  # clp is in a poly edge
            #     _clp_ =
            # --
            if d1 < 0:  # -- chk4, chk5  : E, d0_ because of wrapping crosses
                if chk0:  # clip segment outside ply
                    sub0 = sub0[::-1] if len(sub0) > 0 else []
                    out_ids, _, __ = _op_(p_p, p_c, p_seen, p_out)
                    sub1 = pl_n[out_ids[::-1]]
                if len(sub0) > 0 and len(sub1) > 0:
                    # reinsert vals since a hole was formed
                    p_out.insert(0, out_ids)
                    sub = np.concatenate((sub1, sub0), axis=0)
                    out.append(sub)
                    kind_.append(2)  # a hole between the two
                else:
                    sub = []  # or _out_bits_
            # --
            if d1 == 1:  # poly ids are sequential, clp is inside or outside
                sub1 = pl_n[[p_p, p_c]]
                if chk0:
                    in_clp += [sub1]
                    sub = np.concatenate((sub0, sub1[::-1]), axis=0)
                    out.append(sub)
                    kind_.append(-1)
                elif chk1:
                    sub = np.concatenate((sub1, sub0[::-1]), axis=0)
                    out.append(sub)
                    kind_.append(0)  # ?? check
            # --
            elif d1 > 1:  # clp inside and outside check
                if chk0:  # also clip segment outside ply, chk3==True
                    in_ids, _, __ = _ip_(p_p, p_c, p_seen, p_in)
                    if len(in_ids) > 0:
                        # -- in_ids 0 & -1 should be in p_cut
                        sub1 = pl_n[in_ids]
                        in_clp += [sub1]
                        p_seen += in_ids
                        sub = np.concatenate((sub0[::-1], sub1), axis=0)
                        out.append(sub)
                        kind_.append(-1)
                        sub0 = []  # empty sub0 from above since it is outside
                elif chk1:  # poly outside clp and chk2==True?
                    out_ids, _, __ = _op_(p_p, p_c, p_seen, p_out)
                    if len(out_ids) > 0:
                        sub1 = pl_n[out_ids]
                        sub = np.concatenate((sub1, sub0[::-1]), axis=0)
                        out.append(sub)
                        kind_.append(0)
                elif chk2:
                    out_ids, _, __ = _op_(p_p, p_c, p_seen, p_out)
                    if len(out_ids) > 0:
                        p_seen += in_ids
                        out.append(pl_n[out_ids + [out_ids[0]]])
                        kind_.append(-9)
        # --
        print("cnt {} : p_out {}\n      : c_out {}".format(cnt, p_out, c_out))
        msg[-1] = msg[-1] + "{!s:>4}".format(kind_[-1])
        prev = [c_c, p_c]
        p_seen.append(p_c)
        c_seen.append(c_c)
        # # --
    # --
    #
    if isinstance(in_clp, list) and len(in_clp) > 1:
        _c_ = np.concatenate(in_clp, axis=0)  # intersect as well
        clipped = _del_seq_pnts_(_c_, True)      # this seems to be
        if (_c_[0] != _c_[-1]).any():
            clipped = np.concatenate((clipped, clipped[0][None, :]), axis=0)
        kind_.append(1)  # or 3 if concave hull in c_in check below
        out.append(clipped)
    else:
        clipped = np.asarray([])
    #
    final = np.asarray(out, dtype='O')
    hull_ = []
    idx = np.array(kind_)
    idx_hole = np.nonzero(idx == 2)[0]    # holes, 2
    idx_all = np.nonzero(idx < 2)[0]      # symmetrical difference, -1, 0, 1
    idx_p_out = np.nonzero(idx == 0)[0]   # poly outside clip (pairwise erase)
    idx_c_out = np.nonzero(idx == -1)[0]  # clp outside poly
    idx_p_in = np.nonzero(idx == 1)[0]    # clipped
    idx_c_in = np.nonzero(idx == 1)[0]    # reverse pairwise erase, eg clipped
    #
    hole_ = final[idx_hole] if len(idx_hole) > 0 else []
    symm_ = final[idx_all]
    cin_ = final[idx_c_in] if len(idx_c_in) > 0 else []  # clp times E,d0_
    if isinstance(cin_, np.ndarray):
        if len(cin_) == 1:
            cin_ = cin_[0]
            clipped = cin_
        else:
            tmp = _del_seq_pnts_(cin_[-1], True)
            if clipped.shape == tmp.shape:
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
    hull_idx = np.nonzero(idx == 3)[0]
    hull_ = final[hull_idx] if len(hull_idx) > 0 else []
    # --
    # if as_geo:
    #     return npg.arrays_to_Geo(final, kind=2, info=None, to_origin=False)
    # erase_, clp_0, clp_1, hole_, symm_, erase_r
    if testing:
        print("\nidx : {}\n".format(idx))
        print("cc  cp pc pp   d0 d1 chk0  chk1  chk2  chk3  chk4  chk5  kind")
        print("\n".join([i for i in msg]))
    # return final, idx
    return final, idx, clipped, cin_, hole_, symm_, erase_, erase_r, hull_


# ----------------------------------------------------------------------------
# ---- (2) append geometry
#
def append_(this, to_this):
    """Append `this` geometry `to_this` geometry.

    Parameters
    ----------
    this : array(s) or a Geo array
        The geometry to append to the existing geometry (`to_this`).
        `this` can be a single array, a list of arrays or a Geo array.
        If you want to append object array(s) (dtype= 'O'), then convert to a
        list of arrays or a list of lists first.
    to_this : Geo array
        The Geo array to receive the new geometry

    Returns
    -------
    A new Geo array.

    a = np.array([[0, 10.],[5., 15.], [5., 0.], [0., 10]])
    b = a + [5, 0]
    this = [a, b]
    to_this = s0
    """
    if not hasattr(to_this, "IFT"):
        print("\nGeo array required for `to_this`\n")
        return None
    if hasattr(this, "IFT"):
        a_stack = this.XY
        IFT = this.IFT
        if this.K != to_this.K:
            print("\nGeo array `kind` is not the same,\n")
            return None
    else:
        a_stack, IFT, extent = npGeo.array_IFT(this)
    last = to_this.IFT[-1, :]
    add_ = []
    for i, row in enumerate(IFT, 1):
        add_.append([last[0] + i, last[2] + row[1],
                     last[2] + row[2]] + list(row[3:]))
    add_ = np.atleast_2d(add_)
    new_ift = np.vstack((to_this.IFT, add_))
    xys = np.vstack((to_this.XY, a_stack))
    kind = to_this.K
    sr = to_this.SR
    out = npGeo.Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
    return out


# ----------------------------------------------------------------------------
# ---- (3) dissolve shared boundaries
#
def _union_op_(a, b):
    """Union two polygon features with a dissolve of their shared edges.

    Private helper for  `union`.

    Returns
    -------
    Unioned polygon.
    """
    def _rem_geom_(ar, w_, main=True):
        """Remove the geometry."""
        N = ar.shape[0]
        # -- condition the removal points
        ids = np.arange(N)
        rem = sorted(list(set(w_)))
        keep = sorted(list(set(ids).difference(rem)))
        chk = np.isin(rem, [0, N-1])
        if chk.all():  # -- cannot split between last and first point
            return None
        if chk.any():  # -- remove duplicate start/end point
            rem = rem[:-1]
        # -- produce the remove list
        if not (0 in rem):  # start point not in those to remove
            bit = np.concatenate((ar[rem[-1]:], ar[:rem[0] + 1]), axis=0)
            return bit
        else:
            return ar[keep]
    # --
    w0, w1 = np.nonzero((a[:, None] == b).all(-1))
    if len(w0) == 0:
        return None
    z0 = _rem_geom_(a, w0, main=True)
    if z0 is None:
        return None
    z1 = _rem_geom_(b, w1, main=False)
    if z1 is None:
        return None
    out = np.concatenate((z0, z1), axis=0)
    out = npg.roll_arrays(out)  # -- npg.roll_arrays
    out = _del_seq_pnts_(out)  # -- npg.npg_bool_hlp
    return out


def union_adj(a, asGeo=False):
    """Union polygon features with a dissolve of internal shared edges.

    Parameters
    ----------
    a : Geo array
        The polygon geometry to union.  Holes not supported as yet.

    Requires
    --------
    `_union_op_`
        to close polygons and/or sort coordinates in cw or ccw order.

    """
    if not hasattr(a, "IFT"):
        print("\nGeo array required as input.")
        return None
    # --
    # -- get the intersection points
    rings = _bit_check_(a, just_outer=True)
    if len(rings) < 2:
        print(f"\nTwo or more shapes required, {rings} found")
        return None
    a = rings[0]
    last = len(rings) - 2
    out = []
    for i, b in enumerate(rings[1:]):
        ret = _union_op_(a, b)
        is_last = i == last
        if ret is None:  # -- finished sub array
            tmp = npg.roll_arrays(a)  # -- npg.roll_arrays
            tmp = _del_seq_pnts_(tmp)  # -- npg.npg_bool_hlp
            out.append(tmp)
            a = b[:]  # -- assign the last poly as the first and repeat
        elif is_last:
            tmp = npg.roll_arrays(ret)
            tmp = _del_seq_pnts_(tmp)
            out.append(tmp)
        else:
            a = ret[:]
    if asGeo:
        out = npGeo.arrays_to_Geo(out, 2, "dissolved", False)
        out = npGeo.roll_coords(out)
    return out


def union_over(this, on_this):
    """Return union of geometry."""
    result = overlay_ops(this, on_this, False)
    final, idx, clp_, cin_, hole_, symm_, erase_, erase_r = result
    out = merge_(clp_, on_this)
    return out


# ----------------------------------------------------------------------------
# ---- (4) adjacency
#
def adjacency_matrix(a, collapse=True, prn=False):
    """Construct an adjacency matrix from an input polygon geometry.

    Parameters
    ----------
    a : array-like
        A Geo array, list of lists/arrays or an object array representing
        polygon geometry.

    Returns
    -------
    An nxn array adjacency for polygons and a id-keys to convert row-column
    indices to their original ID values.

    The diagonal of the output is assigned the shape ID. Adjacent to a cell
    is denoted by assigning the shape ID.  Non-adjacent shapes are assigned -1

    Example::

        ad =adjacency_matrix(a)             # -- 5 polygons shapes
        array([[ 0, -1, -1,  3,  4],
               [-1,  1, -1, -1, -1],
               [-1, -1,  2, -1, -1],
               [ 0, -1, -1,  3,  4],
               [ 0, -1, -1,  3,  4]])
        ad >= 0                             # -- where there are links
        array([[1, 0, 0, 1, 1],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [1, 0, 0, 1, 1],
               [1, 0, 0, 1, 1]])
        row_sums = np.sum(ad >= 0, axis=1)  # sum the rows
        w = np.where(row_sums > 1)[0]       # find out where sum is > 1
        np.unique(ad[w], axis=0)            # get the unique combinations
        array([[ 0, -1, -1,  3,  4]])       # of the adjacency test if needed

    Polygons 0, 3 and 4 are connected. Polygons 1 and 2 are not connected to
    other geometry objects.
    """
    def combos(ad):
        """Return the collapsed adjacency matrix.

        Parameters
        ----------
        ad : 2D array
            The adjacency matrix
        """
        n = ad.shape[0]
        m = [list(ad[i][ad[i] > 0]) for i in range(n)]
        out = []
        p_c = m[0]
        for i, cur in enumerate(m[1:]):
            # prev = m[i]
            tst = np.isin(cur, p_c).any()
            if tst:
                p_c += cur
            else:
                val = sorted(list(set(p_c)))
                out.append(val)
                p_c = cur
        if len(p_c) > 0:
            out.append(sorted(list(set(p_c))))
        return out

    def recl(arr, ids):
        """Reclass non-sequential id values in an array."""
        u = np.unique(ids)
        d = np.arange(len(u))
        ud = np.concatenate((u[:, None], d[:, None]), axis=1)
        du = ud[::-1]
        for i in du:
            if i[0] - i[1] != 0:
                arr[arr == i[1]] = i[0]  # reclass back to original values
            # else:
            #     arr[arr == i[0]] = i[0]
        return arr
    #
    rings = _bit_check_(a, just_outer=True)  # get the outer rings
    ids = a.IDs[a.CL == 1]                   # and their associated ID values
    n = len(rings)
    N = np.arange(n)
    ad = np.full((n, n), -1)
    np.fill_diagonal(ad, N)  # better with ids but will reclass later
    # -- could have used np.triu_indices but it is slower
    for i in N:
        for j in N:
            if j > i:  # changed from j != i
                ij = _adjacent_(rings[i], rings[j])
                if ij:
                    ad[i, j] = j
                    ad[j, i] = i  # added the flop
    if np.sum(ids - N) > 0:
        ad = recl(ad, ids)  # reclass the initial values using the actual ids
    if prn:
        print("Adjacency matrix\n  n : poly ids\n" + "-" * 14)
        row_frmt = "{:>3.0f} : {!s:<15}"
        out = "\n".join([row_frmt.format(i, row[row != -1])
                         for i, row in enumerate(ad)])
        print(out)
        return None
    if collapse:
        return combos(ad)
    return ad


# ----------------------------------------------------------------------------
# ---- (6) merge geometry
#
def merge_(this, to_this):
    """
    Merge `this` geometry and `to_this` geometry.  The direction is important.

    Parameters
    ----------
    this : array(s) or a Geo array
        The geometry to merge to the existing geometry (`to_this`).
    to_this : Geo array
        The Geo array to receive the new geometry.

    Notes
    -----
    The `this` array can be a single array, a list of arrays or a Geo array.
    If you want to append object array(s) (dtype= 'O'), then convert to a
    list of arrays or a list of lists first.

    During the merge operation, overlapping geometries are not intersected.

    Returns
    -------
    A new Geo array.

    this = np.array([[0, 8.], [5., 13.], [5., 8.], [0., 8]])
    b = this + [5, 2]
    this = [a, b]
    to_this = s0
    """
    a = this      # --- rename to simplify the input names
    b = to_this   # merge a to b, or this to_this
    if not hasattr(b, 'IFT'):
        b = npGeo.arrays_to_Geo(b)
    b_XY = b.XY
    b_IFT = b.IFT
    if hasattr(this, 'IFT'):
        if a.K != b.K:
            print("\nGeo array `kind` is not the same.\n")
            return None
        a_XY = a.XY
        a_IFT = a.IFT
    else:
        a = np.asarray(a)
        if a.ndim == 2:
            a = [a]
        a_XY, a_IFT, extent = npGeo.array_IFT(a)
        a_XY = a_XY + extent[0]
    last = b.IFT[-1, :]
    add_ = []
    for i, row in enumerate(a_IFT, 1):
        add_.append([last[0] + i, last[2] + row[1],
                     last[2] + row[2]] + list(row[3:]))
    add_ = np.atleast_2d(add_)
    new_ift = np.vstack((b_IFT, add_))
    xys = np.vstack((b_XY, a_XY))
    kind = b.K
    sr = b.SR
    out = npGeo.Geo(xys, IFT=new_ift, Kind=kind, Extent=None, Info="", SR=sr)
    return out


# ----------------------------------------------------------------------------
# -- (7) union geometry
#
# ---- Old
# def dissolve(a, asGeo=True):
#     """Dissolve polygons sharing edges.

#     Parameters
#     ----------
#     a : Geo array
#         A Geo array is required. Use ``arrays_to_Geo`` to convert a list of
#         lists/arrays or an object array representing geometry.
#     asGeo : boolean
#         True, returns a Geo array. False returns a list of arrays.

#     Notes
#     -----
#     >>> from npgeom.npg_plots import plot_polygons  # to plot the geometry

#     `_isin_2d_`, `find`, `_adjacent_` equivalent::

#         (b0[:, None] == b1).all(-1).any(-1)
#     """

#     def _adjacent_(a, b):
#         """Check adjacency between 2 polygon shapes."""
#         s = np.sum((a[:, None] == b).all(-1).any(-1))
#         if s > 0:
#             return True
#         return False

#     def _cycle_(b0, b1):
#         """Cycle through the bits."""

#         def _find_(a, b):
#             """Find.  Abbreviated form of ``_adjacent_``,
#             to use for slicing."""
#             return (a[:, None] == b).all(-1).any(-1)

#         idx01 = _find_(b0, b1)
#         idx10 = _find_(b1, b0)
#         if idx01.sum() == 0:
#             return None
#         if idx01[0] == 1:  # you can't split between the first and last pnt.
#             b0, b1 = b1, b0
#             idx01 = _find_(b0, b1)
#         dump = b0[idx01]
#         # dump1 = b1[idx10]
#         sp0 = np.nonzero(idx01)[0]
#         sp1 = np.any(np.isin(b1, dump, invert=True), axis=1)
#         z0 = np.array_split(b0, sp0[1:])
#         # direction check
#         # if not (dump[0] == dump1[0]).all(-1):
#         #     sp1 = np.nonzero(~idx10)[0][::-1]
#         sp1 = np.nonzero(sp1)[0]
#         if sp1[0] + 1 == sp1[1]:  # added the chunk section 2023-03-12
#             # print("split equal {}".format(sp1))
#             chunk = b1[sp1]
#         else:
#             sp2 = np.nonzero(idx10)[0]
#             # print("split not equal {} using sp2 {}".format(sp1, sp2))
#             chunks = np.array_split(b1, sp2[1:])
#             chunk = np.concatenate(chunks[::-1])
#         return np.concatenate((z0[0], chunk, z0[-1]), axis=0)

#     def _combine_(r, shps):
#         """Combine the shapes."""
#         missed = []
#         processed = False
#         for i, shp in enumerate(shps):
#             adj = _adjacent_(r, shp[1:-1])  # shp[1:-1])
#             if adj:
#                 new = _cycle_(r, shp[:-1])  # or shp)
#                 r = new
#                 processed = True
#             else:
#                 missed.append(shp)
#         if len(shps) == 2 and not processed:
#             missed.append(r)
#         return r, missed  # done

#     # --- check for appropriate Geo array.
#     if not hasattr(a, "IFT") or a.is_multipart():
#         msg = """function : dissolve
#         A `Singlepart` Geo array is required. Use ``arrays_to_Geo``
#         to convert
#         arrays to a Geo array and use ``multipart_to_singlepart`` if needed.
#         """
#         print(msg)
#         return None
#     # --- get the outer rings, roll the coordinates and run ``_combine_``.
#     a = a.outer_rings(True)
#     a = a.roll_shapes()  # not needed
#     a.IFT[:, 0] = np.arange(len(a.IFT))
#     out = []
#     # -- try sorting by  y coordinate rather than id, doesn't work
#     # cent = a.centers()
#     # s_ort = np.argsort(cent[:, 1])  # sort by y
#     # shps = a.get_shapes(s_ort, False)
#     # -- original, uncomment below
#     ids = a.IDs
#     shps = a.get_shapes(ids, False)
#     r = shps[0]
#     missed = shps
#     N = len(shps)
#     cnt = 0
#     while cnt <= N:
#         r1, missed1 = _combine_(r, missed[1:])
#         if r1 is not None and N >= 0:
#             out.append(r1)
#         if len(missed1) == 0:
#             N = 0
#         else:
#             N = len(missed1)
#             r = missed1[0]
#             missed = missed1
#         cnt += 1
#     # final kick at the can
#     if len(out) > 1:
#         r, missed = _combine_(out[0], out[1:])
#         if missed is not None:
#             out = [r] + missed
#         else:
#             out = r
#     if asGeo:
#         out = npGeo.arrays_to_Geo(out, 2, "dissolved", False)
#         out = npGeo.roll_coords(out)
#     return out  # , missed


# -- Extras

    # def on_pairs(col):
    #     """Return sequential ids from the intersections not in or out."""
    #     segs = []
    #     for cn, v in enumerate(col[1:], 0):
    #         prev = col[cn]
    #         dff = v - prev
    #         if dff == 1:
    #             segs.append([prev, v])
    #     return segs

    # def _chk_in_lst(_p, _c, _case):
    #     """Boolean check of poly or clip points.

    #     Parameters
    #     ----------
    #     _p, _c : integer
    #         Previous or current point id values.
    #     _case : list of lists
    #         Inside or outside point lists.

    #     Notes
    #     -----
    #     This function is used to see if the previous (`_p`) or current (`_c`)
    #     poly or clip points are inside or outside their counterpart.
    #       The same function can be used for either case.
    #     """
    #     for lst in _case:
    #         if _p in lst and _c in lst:
    #             return True, lst
    #     return False, []


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")

    #  all work as of 2024-01-1`
    # final, idx, clipped, cin_, hole_, symm_, erase_, erase_r, hull_ =
    # overlay_ops(edgy1, eclip)
    # overlay_ops(E, d0_)
    # overlay_ops(pl_, cl_)
    # overlay_ops(p00, c00)
