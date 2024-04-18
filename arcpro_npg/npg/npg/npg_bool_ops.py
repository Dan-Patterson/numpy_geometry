# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
------------
npg_bool_ops
------------

Modified :
    2024-03-28

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
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as uts  # noqa

import networkx as nx

import npg  # noqa
from npg import npGeo
from npg.npGeo import roll_arrays  # noqa
# from npg.npg_pip import np_wn
from npg.npg_bool_hlp import add_intersections, _del_seq_pnts_  # p_ints_p
from npg.npg_geom_hlp import _bit_check_, _bit_area_  # radial_sort
from npg.npg_prn import prn_  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]

__all__ = [
    'clp_', 'erase_', 'symm_diff_',
    'append_',
    'adjacency_matrix',
    'merge_',
    'union_adj',
    '_renumber_pnts_',
    'connections_dict',
    'polygon_overlay'
]   # 'dissolve'

__helpers__ = ['_adjacent_', '_cut_pairs_', 'orient_clockwise']
__imports__ = [
    'add_intersections', '_del_seq_pnts_',
    '_bit_check_', '_bit_area_', 'remove_geom', 'prn_'
]


# ---- helpers
#
def trichk(c_p, c_c, p_p, p_c, c_cut):
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


def _cut_across_(arr):
    """Return cut lines from `onConP` or `id_plcl` with only crossed lines."""
    c_segs = []
    p_segs = []
    idxs = arr[:, 0]
    prev = idxs[0]
    for cn, v in enumerate(idxs[1:], 0):
        dff = v - prev
        if dff == 1:
            vals = [arr[cn, 1], arr[cn + 1, 1]]
            if abs(vals[1] - vals[0]) > 1:
                c_segs.append([prev, v])
                p_segs.append(vals)
        prev = v
    return c_segs, p_segs


def _cut_pairs_(arr):
    """Return cut lines from `onConP` or `id_plcl`.

    A `cut` is where a line crosses two different segments.  The id value of
    those segments is returned.  For the clipper ids, `c_segs`, these will have
    a difference of 1.  For the poly ids, `p_segs` the id values indicate the
    point connections, the order indicting the direction of the cut::

      _cut_pairs_(onConP[:, :2])
      ([[0, 1], [3, 4], [4, 5], [ 9, 10], [10, 11], [13, 14]],  # clp ids
       [[0, 8], [6, 2], [2, 5], [16, 19], [19, 15], [13, 21]])  # poly ids
      ...
      _cut_pairs_(id_plcl)
      ([[5, 6], [15, 16]],  # poly ids
       [[5, 3], [11, 9]])   # clp ids

    - `onConP`  - the columns are arranged as clipper, poly
    - `id_plcl` - poly is first then clipper
    """
    c_segs = []
    p_segs = []
    idxs = arr[:, 0]
    prev = idxs[0]
    for cn, v in enumerate(idxs[1:], 0):
        dff = v - prev
        if dff == 1 or v == 0:
            vals = [arr[cn, 1], arr[cn + 1, 1]]
            c_segs.append([prev, v])
            p_segs.append(vals)
        prev = v
    return c_segs, p_segs


def orient_clockwise(geom):
    """Orient a polygon ring so it is clockwise.

    Parameters
    ----------
    geom : list
        a list of polygons geometry arrays

    Requires
    --------
    Ensure `_delete_seq_dups_` is run on the geometry first.
    """
    cw_ordered = []
    for i in geom:
        if _bit_area_(i) > 0.0:
            cw_ordered.append(i)
        else:
            cw_ordered.append(i[::-1])
    return cw_ordered


def _c_p_(arr):
    """Return cut lines from `onConP` or `id_plcl`."""
    _segs = [[0]]
    idxs = arr[:, 0]
    prev = idxs[0]
    for cn, v in enumerate(idxs[1:], 0):
        dff = v - prev
        if dff == 1:
            _segs[-1].append(v)
        else:
            # vals = [arr[cn, 1], arr[cn + 1, 1]]
            _segs.append([v])
        prev = v
    return _segs


def _renumber_pnts_(cl_n, pl_n):
    """Return renumbered points from two polygons that intersect.

    Parameters
    ----------
    cl_n, pl_n : arrays
        In normal useage, `pl_n` is the polygon being clipped by `cl_n`,

    Returns
    -------
    The an array of id values which include columns for the original ids,
    the re-sequenced ids and the new values the include id values from
    `cl_n` ids, where the points are equal.  See `Example` below.

     The combined `cl_n`, `pl_n` is returned as well as CP_.

    Example
    -------
    Since the start and end of each poly are rotated to the first intersection
    point, the first and last of each sequence are renumbered, then the polygon
    ids are sequenced from there.  Unique points in `pl_n` are assigned a new
    number.  Intersection points with `cl_n` are assigned their value, as
    follows::

      # Initial state
      c_ids  [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
      p_ids  [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
      # Renumber end points
      c_ids  [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 0]
      p_ids  [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 0]
      # pl_n intersects with points 1, 3, 5, 7, 8, 4 in `cl_n` sequentially
      # new sequence
      z = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,
            0, 13,  1,  3,  5,  7, 18, 19, 20, 21,  8,  4,  0]

    The old and new id values can be derived using the following::

      old_new = np.concatenate((p_ids_old[:, None],
                                p_ids[:, None],
                                z[N_c + 1:][:, None]),
                               axis=1)
      # where `z` is the new sequence and N_c is number of points in `cl_n`.
      N_c = cl_n.shape[0] - 1
      old_new
      array([[ 0, 12,  0],
             [ 1, 13, 13],
             [ 2, 14,  1],
             [ 3, 15,  3],
             [ 4, 16,  5],
             [ 5, 17,  7],
             [ 6, 18, 18],
             [ 7, 19, 19],
             [ 8, 20, 20],
             [ 9, 21, 21],
             [10, 22,  8],
             [11, 23,  4],
             [12, 24,  0]])

    The resultant `old_new` columns are the original sequence, the new
    sequence and the renumbered ids which reference identical points in `cl_n`.
    """
    #
    N_c = cl_n.shape[0] - 1
    N_p = pl_n.shape[0] - 1
    CP_ = np.concatenate((cl_n, pl_n), axis=0)  # noqa
    # --
    # -- Initially number the sequential points.
    c_ids = np.arange(0, N_c + 1)  # cl_n ids
    p_ids_old = np.arange(0, N_p + 1)
    #
    strt = c_ids[-1] + 1  # get new start and end points for `pl_n` sequence.
    end_ = N_c + N_p + 2
    p_ids = np.arange(strt, end_)  # new values for `pl_n` ids
    cp_ids = np.concatenate((c_ids, p_ids), axis=0)  # combine both
    #
    # -- Find where the points on `cl_n` (wc0) are equal to the points
    #    on `pl_n` (wp0).
    wc0, wp0 = np.nonzero((pl_n == cl_n[:, None]).all(-1))  # onConP by part
    wc0_ids = np.nonzero((wc0 == N_c))[0]  # reclass N_c to 0
    wc0[wc0_ids] = 0                       # first is already 0
    wp0_ids = np.nonzero((wp0 == N_p))[0]  # reclass N_p to 0 and set first
    wp0[wp0_ids] = 0                       # to 0 as well
    #
    zz = wp0[1:-1] + N_c + 1         # increment the `pl_n` ids
    zz0 = list(zip(zz, wc0[1:-1]))   # concatenate pl and cl ids to 2d array
    zz0 = np.array(zz0)              #
    new_ids = np.copy(cp_ids)            # use a copy of `cp_ids`
    # find where they are equal, r0 is basically clp ids, r1 is the ids
    # of where clp equals poly and poly ids outside and inside of clp
    r0, r1 = (new_ids == zz0[:, 0][:, None]).nonzero()
    # -- zz1 are the relabelled points in pl_n, cl_n is first, pl_n is 2nd
    new_ids[r1] = zz0[:, 1]
    new_ids[(new_ids == N_c).nonzero()] = 0   # renumber N_c and N_c+N_p+1 to 0
    new_ids[(new_ids == N_c + N_p + 1).nonzero()] = 0
    #
    # -- tricky, renumber the points that aren't duplicate of clp from N_c+1
    # tmp = np.nonzero(new_ids > N_c + 1)[0]
    # tmp0 = [N_c + 1 + i for i in range(len(tmp))]
    # new_ids[tmp] = tmp0
    #
    old_new = np.concatenate((p_ids_old[:, None],
                              p_ids[:, None],
                              new_ids[N_c + 1:][:, None]),
                             axis=1)
    return new_ids, old_new, CP_


def connections_dict(from_to, bidirectional=False):
    """Return a dictionary of point connections given an array of connections.

    Parameters
    ----------
    from_to : array-like
        A 2d array or list of lists showing from-to connections.
    bidirectional : boolean
        True, returns the dictionary showing from-to and to-from connections.

    Example
    -------
    Given point from-to connections::

        ex = [[0, 1], [0, 16], [0, 19], [1, 0], [1, 2], [2, 16], [16, 19]]

        connections_dict(ex, bidirectional=False)
        {0: [1, 16, 19], 1: [0, 2], 2: [16], 16: [19]}

        connections_dict(ex, bidirectional=True)
        {0: [1, 16, 19], 1: [0, 2], 2: [1, 16], 16: [0, 2, 19], 19: [0, 16]}
    """
    frto = from_to
    if bidirectional:
        _src_ = np.copy(frto)
        _ft_ = np.concatenate((_src_, _src_[:, [1, 0]]), axis=0)
    else:
        _ft_ = np.copy(frto)
    #
    uniq_frto = np.unique(_ft_, axis=0)
    if (uniq_frto[0] == [0, 0]).all():
        uniq_frto = uniq_frto[1:]
    frto2 = np.copy(uniq_frto)
    #
    d = {}
    for elem in frto2:
        if elem[0] in d:
            if elem[0] != elem[1]:
                d[elem[0]].append(elem[1])
        else:
            d[elem[0]] = [elem[1]]
    for k, v in d.items():
        if len(v) > 1:
            d[k] = sorted(list(set(v)))
    return d


# ---- ---------------------------
# ---- (1) polygon overlay operations.
#
def clp_(ply_a, ply_b, as_geo=True):
    """Return the symmetrical difference.  See `overlay_ops`."""
    result = polygon_overlay(ply_a, ply_b)
    if as_geo:
        return npg.arrays_to_Geo(result, kind=2, info=None, to_origin=False)
    return result


def erase_(ply_a, ply_b, as_geo=True):
    """Return the symmetrical difference.  See `overlay_ops`."""
    result = polygon_overlay(ply_a, ply_b)
    if as_geo:
        return npg.arrays_to_Geo(erase_, kind=2, info=None, to_origin=False)
    return result


def symm_diff_(ply_a, ply_b, as_geo=True):
    """Return the symmetrical difference.  See `overlay_ops`."""
    result = polygon_overlay(ply_a, ply_b)
    if as_geo:
        return npg.arrays_to_Geo(result, kind=2, info=None, to_origin=False)
    return result


# ---- ---------------------------
# ---- (2) polygon overlay functions.
#
def _in_out_(geom, from_to_val, cuts):
    """Return in, out polys for single cuts.  They aren't closed."""
    out_ = []
    in_ = []
    not_zero = from_to_val[from_to_val[:, -1] != 0]
    not_zero[:, 1] = not_zero[:, 1] + 1  # update the `to` value
    for cnt, nz in enumerate(not_zero):  # originally from_to_val:
        nz = not_zero[cnt]
        f, t, v = nz
        # if v != 0:  removed this check by using `not_zero`
        ar = geom[f:t]
        col0 = 0 if v == -1 else 1
        col1 = 1 if v == -1 else 0
        whr0 = f == cuts[:, col0]
        whr1 = t - 1 == cuts[:, col1]
        whr2 = ([t - 1, f] == cuts).all(-1).any(-1)
        if whr0.any() and whr1.any():
            pre = cuts[whr0][0]  # take the first
            post = cuts[whr1][0]
            btw = sorted([pre[1], post[0]])  # ensure btw is ascending
            if v == -1:
                pre = pre[::-1]
                post = post[::-1]
            if np.sum(whr1) > 1:  # check for more than one post crossing
                post_more = cuts[whr1]
                cross_cut = post_more[-1]
                chk = [cross_cut[1], geom.shape[0] - 1]  # check for close
                if chk in cuts:
                    post = cross_cut
            pre_g = geom[pre]
            post_g = geom[post]
            if (pre == post).all():  # btw_g will be ignored in this case
                arrs = [pre_g, ar]
                btw_g = None
            else:
                if (btw == cuts).all(-1).any(-1):
                    r = np.arange(min(btw), max(btw) + 1)
                    btw_g = geom[r]
                else:
                    btw_g = None
                if btw_g is None:
                    arrs = [pre_g, ar, post_g]
                else:
                    arrs = [pre_g, ar, post_g, btw_g]
            ar = np.concatenate(arrs, axis=0)
        elif whr2:
            ar = np.concatenate((geom[t - 1][None, :], ar), axis=0)
        if v == -1:
            out_.append(ar)
        else:
            in_.append(ar)
    return in_, out_


def _in_out_on_(combos, cl_n, pl_n, c_cut, p_cut):
    """Return the full results of in, out and on.

    Parameters
    ----------
    See `polygon_overlay` for outputs and descriptions.
    """
    r0 = []
    clp_ply = []
    keep_bits = []
    for cnt, _row_ in enumerate(combos):
        _row_ = combos[cnt]
        cfr, cto, cty, pty, pfr, pto = _row_
        #
        c_pnts = cl_n[cfr: cto + 1]
        p_pnts = pl_n[pfr: pto + 1]
        circ = None
        #
        # -- clip out
        if cty == -1:
            #
            # clp, ply out  (-1, -1)
            if pty == -1:  # both out, check pl_n cut line
                #  previous may have been (0, 0), so check previous cut line
                i, j = combos[cnt - 1, -2:]
                cut_ = pl_n[i:j + 1]
                chk = np.equal(c_pnts[[0, -1]], cut_[[0, -1]]).all()
                # add first out
                if chk:
                    circ = np.concatenate((c_pnts, cut_[::-1]), axis=0)
                    if not np.equal(clp_ply[-1], cut_).all():
                        clp_ply.append(cut_)
                    r0.append(circ)  # add circ
                    # add second out
                    circ = np.concatenate((p_pnts, p_pnts[0][None, :]), axis=0)
                    clp_ply.append(p_pnts[[0, -1]])
                else:
                    keep_bits.append(c_pnts)
                    clp_ply.append(c_pnts)  # have to append something
            #
            # ply on  (-1, 0)
            elif pty == 0:
                if (cl_n[cto] == pl_n[pto]).all(-1):
                    circ = np.concatenate((c_pnts, p_pnts[::-1]), axis=0)
                    clp_ply.append(p_pnts)
                elif (np.array([cfr, cto]) == c_cut).all(-1).any(-1):
                    circ = np.concatenate((c_pnts, cl_n[[cto, cfr]]), axis=0)
                    clp_ply.append(cl_n[cfr: cto + 1])
                else:  # -- close up double outside potentially, e.g. A,C
                    bits = np.concatenate((pl_n[[pto, pfr]], c_pnts))
                    if len(keep_bits) == 0:
                        keep_bits.append(bits)
                    else:
                        keep_bits.append(cl_n[cfr:cto + 1])
                        circ = np.concatenate(keep_bits, axis=0)
                        keep_bits = []
            #
            # clp out, ply in
            elif pty == 1:
                circ = np.concatenate((c_pnts, p_pnts[::-1]), axis=0)
                clp_ply.append(p_pnts)
        #
        # -- clip on
        elif cty == 0:  #
            # ply out  (0, -1)
            if pty == -1:
                circ = np.concatenate((c_pnts[::-1], p_pnts), axis=0)
                chk = (p_pnts[-1] == c_pnts[-1]).all()
                if not chk:
                    keep_bits.append(p_pnts)
                    circ = None
                clp_ply.append(c_pnts)
            # ply on  (0, 0)
            elif pty == 0:
                # -- possible issue, both are valid and may be a triangle
                if cnt == 0:
                    clp_ply.append(p_pnts)
                elif combos[cnt - 1][2] != 0:  # if == 0, append to clp_ply
                    # determine whether clp crosses 2 lines or is on one
                    chk0 = np.equal(cl_n[cto], pl_n[pfr + 1]).all()
                    chk1 = np.equal(cl_n[cto + 1], pl_n[pto]).all()
                    if chk0:  # clp forms a triangle, cuts 2 lines
                        circ = np.concatenate((c_pnts, p_pnts[::-1]),
                                              axis=0)
                        clp_ply.append(c_pnts)
                    elif chk1:  # did not work for p02,c02 previous was (-1,-1)
                        bits = cl_n[cfr:cto + 2]
                        clp_ply.append(bits)  # 2/3 triangle
                        circ = np.concatenate((p_pnts, bits[::-1]), axis=0)
                        # clp_ply = []  # empty clp_ply
                    else:
                        clp_ply.append(p_pnts)
                else:  # 2 sequential 0,0 s
                    clp_ply.append(p_pnts)
            # ply in  (0, 1)
            elif pty == 1:
                circ = np.concatenate((p_pnts, c_pnts[::-1]), axis=0)
                clp_ply.append(p_pnts)
        #
        # -- clip in
        elif cty == 1:  # (1, 0) and (1, -1)
            circ = np.concatenate((p_pnts, c_pnts[::-1]), axis=0)
            clp_ply.append(c_pnts)
        #
        # add circ
        if circ is not None:
            if len(circ) > 3:
                r0.append(circ)
    #
    # -- assemble and clean the geometry
    geom = [_del_seq_pnts_(i) for i in r0]  # -- remove duplicates
    geom = orient_clockwise(geom)           # -- orient clockwise
    clp_ply = np.concatenate(clp_ply, axis=0)
    clp_ply = _del_seq_pnts_(clp_ply)
    return geom, clp_ply, keep_bits


def polygon_overlay(ply_a, ply_b):
    """Return the overlay of two polygons and assemble the areas by type.

    Parameters
    ----------
    ply_a, ply_b : array_like
        `ply_a` is the polygon being differenced/overlain by polygon `ply_b`.

    synonyms::

        `ply_a` = `ply` = `under`
        `ply_b` = `clp` = `over`

    Extras
    ------
    pl_n, cl_n : array-like
        ply_a, ply_b point arrays with new intersections points added.
    p0new_ioo, p1_ioo : array-like
        polygon in-out-on, p1_ioo for ply_b

    Notes
    -----
    Description for `combos`::

        uniq_c_p, idx_ = np.unique(combos[:, 2:4], True, axis=0)
        # possible classes
        uni
        #                   over under
        #     ply_b ply_a   clp  ply
        #      ----------   ---------
        array([[-1,  0],     out, on
               [-1,  1],     out, in
               [-1, -1],     out, out
               [ 0, -1],     on,  out
               [ 0,  1],     on,  in
               [ 1, -1],     in,  out
               [ 1,  0]])    in,  on

    Example for `onConP` for `E`, `d0_`::

      np.nonzero((x_pnts == cl_n[:, None]).all(-1))
      array([ 0,  1,  3,  4,  5,  9, 10, 11, 13, 14])  # x_pnts equals cl_n
      array([ 0,  2,  4,  6,  8,  7,  5,  3,  1,  0])  # cl_n equals x_pnts

      np.nonzero((x_pnts == pl_n[:, None]).all(-1))
      array([ 0,  2,  5,  6,  8, 13, 15, 16, 19, 21])  # x_pnts equals pl_n
      array( [0,  6,  8,  4,  2,  1,  3,  7,  5,  0])  # pl_n equals x_pnts

      np.nonzero((x_pnts == pl_n[:, None]).all(-1))
      # cl_n equals pl_n
      array([ 0,  0,  1,  3,  4,  5,  9, 10, 11, 13, 14, 14])  # 0, 14 equal
      # pl_n equals cl_n
      array([ 0, 21,  8,  6,  2,  5, 16, 19, 15, 13,  0, 21])  # 0, 21 equal

      onConP[:, :2]     id_plcl
      array([[ 0,  0],  array([[ 0,  0],
             [ 1,  8],         [ 2,  4],
             [ 3,  6],         [ 5,  5],
             [ 4,  2],         [ 6,  3],
             [ 5,  5],         [ 8,  1],
             [ 9, 16],         [13, 13],
             [10, 19],         [15, 11],
             [11, 15],         [16,  9],
             [13, 13],         [19, 10],
             [14, 21]])        [21, 14]])

      c_out, c_on, c_in
      array([ 2,  6,  7,  8, 12]),
      array([ 0,  1,  3,  4,  5,  9, 10, 11, 13, 14]),
      array([])

      p_out, p_on, p_in
      array([ 1,  3,  4,  7, 14, 17, 18, 20]),
      array([ 0,  2,  5,  6,  8, 13, 15, 16, 19, 21]),
      array([ 9, 10, 11, 12])

    """
    #
    result = add_intersections(
                ply_a, ply_b,
                roll_to_minX=True,
                p0_pgon=True,
                p1_pgon=True,
                class_ids=False
                )
    # pl_ioo polygon in-out-on, cl_ioo for clip
    pl_n, cl_n, id_plcl, onConP, x_pnts, ps_info, cs_info = result
    p_out, p_on, p_in, pl_ioo = ps_info
    c_out, c_on, c_in, cl_ioo = cs_info
    #
    # CP_ = np.concatenate((cl_n, pl_n), axis=0)  # noqa
    N_c = cl_n.shape[0] - 1  # noqa
    N_p = pl_n.shape[0] - 1  # noqa
    #
    # -- get the renumbered point values for `pl_n`
    ids_new, old_new_ids, CP_ = _renumber_pnts_(cl_n, pl_n)
    _old, _, _new = old_new_ids.T
    frto = np.concatenate((ids_new[:-1][:, None], ids_new[1:][:, None]),
                          axis=1)
    frto = frto[frto[:, 0] != frto[:, 1]]  # -- remove 0,0 middle point
    #
    c_ft = list(zip(c_on[:-1], c_on[1:]))  # removed +1 to `to`
    c_subs = [cl_ioo[i[0]:i[1] + 1] for i in c_ft]  # added +1 to `to`
    c_vals = [sub[1][1] for sub in c_subs]
    c_ft_v = np.concatenate((c_ft, np.array(c_vals)[:, None]), axis=1)
    #
    # c_slices = [cl_n[i:j + 1] for i,j in c_ft_v[:, :2]]
    # c_cut_slices = [cl_n[[i, j]] for i,j in c_cut]
    #
    p_ft = list(zip(p_on[:-1], p_on[1:]))  # removed +1 to `to`
    p_subs = [pl_ioo[i[0]:i[1] + 1] for i in p_ft]  # added +1 to `to`
    p_vals = [sub[1][1] for sub in p_subs]
    p_ft_v = np.concatenate((p_ft, np.array(p_vals)[:, None]), axis=1)
    #
    combos = np.zeros((c_ft_v.shape[0], 6), dtype='int')
    combos[:, :3] = c_ft_v
    combos[:, 3:] = p_ft_v[:, [-1, 0, 1]]
    #
    """

    w0 = np.nonzero(old_new[:, 0] == p_cut[:, 0][:, None])[1]
    w1 = np.nonzero(old_new[:, 0] == p_cut[:, 1][:, None])[1]
    col1 = old_new[w1, -1]
    col0 = old_new[w0, -1]
    p_cut_n = np.concatenate((col0[:, None], col1[:, None]), axis=1)
    all_cuts = np.concatenate((c_cut, p_cut_n), axis=0)
    uni_cuts = np.unique(all_cuts, axis=0)

    """
    # -- If the equality below doesn't match, p_on and its comparison may be
    #    out of order.
    #
    if np.equal(p_on, onConP[:, 1]).all():  # c_on == onConP[:, 0] always
        c_cut0, p_cut0 = _cut_pairs_(onConP[:, :2])  # use onConP, it is sorted
        p_cut1, c_cut1 = _cut_pairs_(id_plcl)  # use id_plcl col 0, save a sort
        c_cut = np.array(sorted(c_cut0 + c_cut1, key=lambda l:l[0]))  # noqa
        p_cut = np.array(sorted(p_cut0 + p_cut1, key=lambda l:l[0]))  # noqa
        results = _in_out_on_(combos, cl_n, pl_n, c_cut, p_cut)
        # geom has dupls removed and oriented clockwise
        geom, clp_ply, keep_bits = results
        return geom, clp_ply, combos, keep_bits
    else:
        # -- `_cut_pairs_` to get cuts and crosses, then sort
        c_cut0, p_cut0 = _cut_pairs_(onConP[:, :2])  # use onConP, it is sorted
        p_cut1, c_cut1 = _cut_pairs_(id_plcl)  # use id_plcl col 0, save a sort
        c_cut = np.array(sorted(c_cut0 + c_cut1, key=lambda l:l[0]))  # noqa
        p_cut = np.array(sorted(p_cut0 + p_cut1, key=lambda l:l[0]))  # noqa
        #
        # -- `_in_out_` to get geometry
        ply_a_in, ply_a_out = _in_out_(pl_n, p_ft_v, p_cut)
        ply_b_in, ply_b_out = _in_out_(cl_n, c_ft_v, c_cut)
        #
        geom = ply_a_in + ply_a_out + ply_b_in + ply_b_out
        geom = [_del_seq_pnts_(i) for i in geom]
        geom = orient_clockwise(geom)
        #
        clip_polys = [cl_n[c_on], pl_n[p_on]]
        keep_bits = None
        # -- clip result is either pl_n[w0], or cl_n[w1] when the cut line
        #     is 2 points, otherwise you have to assemble the inside and cut
        #     segments
        return geom, clip_polys, combos, keep_bits

    # works for
    # (p00, c00), (p00, c01), (p00, c02), (p00, c03)
    # (p01, c00), (p01, c01), (p01, c02), (p01, c03)
    # (p02, c00), (p02, c01), (p02, c02)*, (p02, c03)*  * used _in_out_on_
    # (p03, c00), (p03, c01), (p03, c02), (p03, c03)
    # A, C  nope
    # B, K
    # C, K  nope
    # E, d0_


def overlay_nx(poly, clp, extras=True):
    """Return the results of overlay using networkx."""
    result = add_intersections(
                poly, clp,
                roll_to_minX=True,
                p0_pgon=True, p1_pgon=True,
                class_ids=False
                )
    # pl_ioo polygon in-out-on, cl_ioo for clip
    pl_n, cl_n, id_plcl, onConP, x_pnts, ps_info, cs_info = result
    p_out, p_on, p_in, pl_ioo = ps_info
    c_out, c_on, c_in, cl_ioo = cs_info
    # --
    # -- Initially number the sequential points, slicing off the duplicate 0
    #    point between clp and poly
    N_c = cl_n.shape[0] - 1
    N_p = pl_n.shape[0] - 1  # noqa
    #
    # --
    cp_ids, old_new_ids, CP_ = _renumber_pnts_(cl_n, pl_n)
    #
    # -- point designations and renumbering p_sing_o and p_sing_i
    c_sing_out = c_out
    c_sing_in = c_in  # -- c_on defined above
    p_sing_out = np.asarray([], dtype='int')
    p_sing_in = np.asarray([], dtype='int')
    if len(p_out > 0):
        p_sing_out = old_new_ids[p_out, :][:, -1]  # p_out + N_c + 1  # -- out
        cp_ids[p_sing_out] = p_sing_out
    if len(p_in) > 0:
        p_sing_in = old_new_ids[p_in, :][:, -1]  # p_in + N_c + 1  # -- in
        cp_ids[p_sing_in] = p_sing_in
    if len(p_on) > 0:
        p_sing_on = old_new_ids[p_on[1:-1], :][:, -1]  # first and last are on
        cp_ids[p_sing_on] = p_sing_on
    #
    # -- Create frto and get unique values
    frto = np.concatenate((cp_ids[:-1][:, None], cp_ids[1:][:, None]), axis=1)
    frto = frto[frto[:, 0] != frto[:, 1]]  # -- remove 0,0 middle point
    # ioo = np.concatenate((cl_ioo[:-1, 1], pl_ioo[:-1, 1]))  # ditto
    # frto_ioo = np.concatenate((frto, ioo[:, None]), axis=1)
    # -- construct the cycles
    #  import networkx as nx
    #
    # d = connections_dict(frto, True)
    # H = nx.from_dict_of_lists(d)  # yields the same edges etc as below
    #
    G = nx.Graph()
    G.add_edges_from(frto)
    #
    # -- weighted graph examples
    # G.add_weighted_edges_from(frto_ioo, weights='ioo')
    # --
    # nx.set_edge_attributes(G, values=ioo, name = 'ioo')
    # nx.set_edge_attributes(G, values=1, name = 'ioo')
    # to view edge data:  G.edges(data=True)
    # H = G.to_directed()  # to get a directed graph from `frto`
    # --
    # -- code from _minimum_cycle_basis
    # list(nx.minimum_spanning_tree(G)) # just returns the nodes of G
    # tree_edges = list(nx.minimum_spanning_edges(G, weight=None, data=False))
    # chords = sorted(list(G.edges - tree_edges -
    #                 {(v, u) for u, v in tree_edges}))
    # tree_edges = np.asarray(tree_edges)
    # chords = np.asarray(chords)
    # -- minimum_cycle_basis
    out = []
    # cycles = sorted(list(nx.cycle_basis(G)))  # needs to be parsed a bit
    cycles = sorted(nx.cycles.minimum_cycle_basis(G))  # not for directed
    for cycle in cycles:
        out.append(nx.find_cycle(G.subgraph(cycle), orientation='ignore'))
    #
    out_ = []
    for i in out:
        sub = []
        if len(i) > 1:
            for j in i[:-1]:
                sub.append(j[0])
            sub.extend(i[-1][:2])
            out_.append(sub)
    out_ = sorted(out_, key=len)
    #
    # -- `cycle_basis`, doesn't solve or return all the polys
    #    cycles2 = sorted(nx.cycles.cycle_basis(G))
    #
    ps = [CP_[i] for i in out_]                  # -- get the polygons
    #
    # -- get the area and centroid and reorient the polys to clockwise
    #
    area_ = [_bit_area_(i) for i in ps]  # -- calculate area, centroid
    _area_ = []
    cw_order = []
    cw_ps = []
    for cnt, ar in enumerate(area_):
        _p, _o = ps[cnt], out_[cnt]
        if ar < 0:  # cw area is positive, ccw area is negative
            _p = _p[::-1]
            _o = _o[::-1]
            ar = abs(ar)
        _area_.append(ar)
        cw_ps.append(_p)
        cw_order.append(_o)
    #
    # cw_ps = [CP_[i] for i in cw_order]  # these are the clockwise polygons
    #
    # -- Check the resultant geometries
    _class_ = np.zeros(len(cw_order), dtype='int')
    _class_.fill(9)
    for i, cw in enumerate(cw_order):
        cw = np.array(cw)
        chk1 = np.isin(cw[:-1], c_on).any()  # on check
        chk2 = np.isin(cw[:-1], c_sing_in).any()  # in check
        chk3 = np.isin(cw[:-1], c_sing_out).any()  # out check
        #
        if max(cw) <= N_c - 1:  # check clp all points are on clp edges
            # -- or :
            chk0 = (_bit_area_(cl_n) - _bit_area_(ps[i])) == 0.
            if chk0:  # could be the clp itself
                _class_[i] = -7
            elif chk1:  # all clp on, poly in
                if len(cw[:-1]) == 3:  # first line crosses 2 poly lines
                    _class_[i] = 1 if chk2 else 0  # poly chk
                elif chk2:
                    _class_[i] = 1     # poly is in
                elif chk3:
                    _class_[i] = -1    # poly is out
                else:  # clp on with > 3 pnts, hence poly in
                    _class_[i] = 1     # poly in, all pnts are on clp segments
            elif chk2:  # clp in
                _class_[i] = 1
            elif chk3:  # may be redundant
                _class_[i] = -1
        elif chk1 and chk2 and chk3:  # no clp on, out or in
            _class_[i] = -8
        else:  # some of the points are ply pnts
            whr = np.nonzero(cw < N_c - 1)[0]  # use n-1 probably a cut line
            vals = cw[whr]
            _c_o_ = np.isin(vals, c_sing_out).any()
            _c_i_ = np.isin(vals, c_sing_in).any()
            chk4 = max(cw) in p_sing_in  # ply in clp
            chk5 = max(cw) in p_sing_out
            if len(cw[:-1]) == 3:  # first line crosses 2 poly lines
                _class_[i] = 0 if chk4 else 1
            elif chk4 and _c_o_:  # some clp outside of poly
                _class_[i] = -1
            elif chk4 and _c_i_:  # some clp inside
                _class_[i] = 1
            elif chk5 and _c_i_:  # some clip in, some poly out
                _class_[i] = 0
            elif chk5 and _c_o_:
                _class_[i] = 2    # hole between poly, clp
            elif chk4:
                _class_[i] = 1    # ply in
            elif chk5:
                _class_[i] = 0    # ply out
            else:
                print("i {} cw {}".format(i, cw))
    #
    """
     -1 clip outside poly:
      0 poly outside clip :  erase
      1 poly inside clip :   clip
      2 area between clip and poly : hole
      3 identity : features or parts that overlap
      : symmetrical diff -1 and 0
    """
    final = []
    kind = [-1, 0, 1, 2]
    for i in kind:
        whr = np.nonzero(_class_ == i)[0]
        sub = []
        if whr.size > 0:
            for j in whr:
                sub.append(ps[j])
        final.append(sub)
    if extras:
        return final, kind, cw_order
    return final, None


# ---- ---------------------------
# ---- (3) append geometry
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


# ---- ---------------------------
# ---- (4) dissolve shared boundaries
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


def union_over(ply_a, ply_b):
    """Return union of geometry."""
    result = polygon_overlay(ply_a, ply_b)
    final, idx, over_, cin_, hole_, symm_, erase_, erase_r = result
    out = merge_(over_, ply_b)
    return out


# ---- ---------------------------
# ---- (5) adjacency
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


# ---- ---------------------------
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


def find_cycles(graph):
    """Find cycles in a graph represented by a dictionary.

    Reference
    ---------
    https://stackoverflow.com/questions/76948570/
    finding-cycles-in-an-undirected-graph-in-node-node-format-returns
    -wrong-resul
    """
    cycles = []
    checked = set()

    def dfs(node, visited, path):
        """Depth first search implementation."""
        visited.add(node)
        path.append(node)

        neighbors = graph.get(node, [])

        for neighbor in neighbors:
            if neighbor in visited:
                # Cycle detected
                start_index = path.index(neighbor)
                cycle = path[start_index:]
                m = tuple(sorted(cycle))
                if len(cycle) > 2 and m not in checked:
                    checked.add(m)
                    cycles.append(cycle)
            else:
                dfs(neighbor, visited, path)

        visited.remove(node)
        path.pop()

    for node in graph.keys():
        dfs(node, set(), [])

    return cycles


"""
visited = [] # List for visited nodes.
queue = []     #Initialize a queue

def bfs(visited, graph, node): #function for BFS
  visited.append(node)
  queue.append(node)

  while queue:          # Creating loop to visit each node
    m = queue.pop(0)
    print (m, end = " ")

    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

# Driver Code
print("Following is the Breadth-First Search")
bfs(visited, graph, '5')    # function calling

"""


"""
produces a dictionary for the above


graph = {
    # 'node': ['adjacent'],
}

n, m = map(int, data.pop(0).split())

for _ in range(m):
    a, b = map(int, data.pop(0).split())
    if b not in graph:
        graph[b] = []
    if a not in graph:
        graph[a] = []

    if a not in graph[b]:
        graph[b].append(a)
    if b not in graph[a]:
        graph[a].append(b)

ans = find_cycles(graph)
print(ans)
print(len(ans))
"""

"""
d = None
def get_cycles(n, o = None, c = []):
   if n == o: #check if current node being examined is the same as the start
      yield c
      # if the condition is met, then a cycle is found and the path is yielded
      #
   else:
      # get the children of the current node from d and iterate over only
      # those that have not been encountered before (except if the node is
      # the same as the start)
      for i in filter(lambda x:x not in c or x == o, d.get(n, [])):
         # recursively find the children of this new node `i`, saving the
         # running path (c + [n])
         yield from get_cycles(i, o = n if o is None else o, c = c + [n])
"""


# ---- old
def _inouton_(combos, cl_n, pl_n):
    """Return the full results of in, out and on.

    Parameters
    ----------
    See `polygon_overlay` for outputs and descriptions.
    """
    # -- 8 possible unique combinations of crosses and segment locations
    ch = np.array([[-1, -1], [-1, 0], [-1, 1],
                   [0, -1], [0, 0], [0, 1],
                   [1, -1], [1, 0], [1, 1]])

    uniq_c_p = np.unique(combos[:, 2:4], axis=0)
    z0 = combos[:, :2]
    # c_val = combos[:, 2]
    cs = [cl_n[i[0]:i[1]] for i in z0]  # overlay segments
    cs = np.array(cs, dtype='O')
    #
    z1 = combos[:, -2:]
    ps = [pl_n[i[0]:i[1]] for i in z1]  # poly segments
    ps = np.array(ps, dtype='O')
    #
    pl_in, pl_out, cl_in, cl_out, _tri_, odd = [], [], [], [], [], []
    clp_ply = []
    type_whr_lst = []
    #
    # -- Begin groupings
    for cnt, pair in enumerate(uniq_c_p):
        wh = np.nonzero((combos[:, 2:4] == pair).all(-1))[0]
        type_whr_lst.append([pair, wh])
        c_sub = cs[wh]
        p_sub = ps[wh]
        #
        _x_ = np.nonzero((pair == ch).all(-1))[0][0]
        #
        # ch[[1, 2]] : [-1, 0], [-1, 1]
        #   clp (b) out, ply (a) on/in
        if _x_ in [1, 2]:
            clp_ply.extend(p_sub)
            circs = [np.concatenate((i[0], i[1][::-1]), axis=0)
                     for i in list(zip(c_sub, p_sub))]
            cl_out.extend(circs)
        #
        # ch[[3] : [0, -1]
        #   clp on, ply out
        elif _x_ == 3:
            clp_ply.extend(c_sub)
            circs = [np.concatenate((i[0][::-1], i[1]), axis=0)
                     for i in list(zip(c_sub, p_sub))]
            pl_out.extend(circs)
        #
        # ch[[5]] : [0, 1]
        #   clp on,  ply in
        elif _x_ == 5:
            clp_ply.extend(p_sub)
            circs = [np.concatenate((i[1], i[0][::-1]), axis=0)
                     for i in list(zip(c_sub, p_sub))]
            pl_in.extend(circs)
        #
        # ch[[6, 7] : [1, -1], [1, 0]
        #   clp in, ply out or on
        elif _x_ in [6, 7]:
            clp_ply.extend(c_sub)
            circs = [np.concatenate((i[1], i[0][::-1]), axis=0)
                     for i in list(zip(c_sub, p_sub))]
            cl_in.extend(circs)
        #
        # ch[[4] : [0, 0]
        #   clp on ply , probable triangles
        elif _x_ == 4:
            clp_ply.extend(c_sub)
            circs = [np.concatenate((i[0], i[1]), axis=0)
                     for i in list(zip(c_sub, p_sub))]
            _tri_.append(circs)
        else:
            clp_ply.extend(c_sub)
            circs = [np.concatenate((i[0], i[1]), axis=0)
                     for i in list(zip(c_sub, p_sub))]
            odd.extend(circs)
    # --
    order = np.concatenate([i[1] for i in type_whr_lst])
    srt_order = np.argsort(order)
    clp_ply = np.array(clp_ply, dtype='O')
    clp_final = np.concatenate(clp_ply[srt_order], axis=0)
    clp_final = _del_seq_pnts_(clp_final)
    all_ = [pl_in, pl_out, cl_in, cl_out]
    tmp = []
    for lst in all_:
        sub = []
        if len(lst) > 0:
            for i in lst:
                sub.append(_del_seq_pnts_(i))
        tmp.append(sub)
    #
    pl_in, pl_out, cl_in, cl_out = tmp
    return clp_final, pl_in, pl_out, cl_in, cl_out, _tri_, odd, type_whr_lst


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
