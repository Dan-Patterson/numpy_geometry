# -*- coding: utf-8 -*-
# noqa: D205, D208, D400, F403
r"""
------------
npg_bool_ops
------------

Modified :
    2025-11-15

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
# from npg.npGeo import roll_arrays  # noqa
# from npg.npg_pip import np_wn
from npg.npg_bool_hlp import add_intersections, _del_seq_dupl_pnts_
from npg.npg_geom_hlp import _bit_check_, _bit_area_  # _is_turn, _bit_length_
from npg.npg_prn import prn_, prn_as_obj  # noqa


script = sys.argv[0]

__all__ = [
    'orient_clockwise',                # (1) general helpers
    'reorder_x_pnts',
    'renumber_pnts',
    'pnt_connections',
    'tri_array',
    'rolling_match',
    'turns',
    'prepare',                         # (2) polygon overlay functions
    'bail',
    'no_overlay_',
    'one_overlay_',
    'nx_solve',
    'wrap_',
    'polygon_overlay',
    'clp_',                            # (3) polygon overlay operations.
    'erase_',
    'symm_diff_',
    'append_',                         # (4) append geometry
    'split_at_intersections',          # (5) split at intersections
    'union_adj',                       # (6) dissolve shared boundaries
    'union_over',
    'adjacency_array',                 # (7) adjacency
    'adjacency_matrix',
    'merge_',                          # (8) merge geometry
]   # 'dissolve'

__helpers__ = [
    '_adjacent_',                      # general helpers, private
    '_cut_across_',
    '_cut_pairs_',

]

__imports__ = [
    'npGeo',                  # main import from npg
    'roll_arrays'             # npGeo
    'add_intersections',      # npg_bool_hlp
    '_del_seq_dupl_pnts_',
    '_bit_check_',            # npg_geom_hlp
    '_bit_area_',
    'prn_',                   # npg_prn
    'prn_as_obj'
]


# ---- ---------------------------
# ---- (1) private helpers
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


def _cut_pairs_(arr, _in_, _out_):
    """Return cut lines from `onConP` or `id_plcl`.

    A `cut` is where a line crosses two different segments.  The id value of
    those segments is returned.  For the clipper ids, `c_segs`, these will
    usually have a difference of 1, but this is not guaranteed.
    For the poly ids, `p_segs` the id values indicate the point connections,
    the order indicting the direction of the cut::

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
    bth = []
    for cn, v in enumerate(idxs[1:], 0):
        dff = v - prev
        if dff == 1 or v == 0:
            vals = [arr[cn, 1], arr[cn + 1, 1]]
            c_segs.append([prev, v])
            p_segs.append(vals)
            bth.append(vals + [prev, v])
        elif dff > 1:
            chk = [i for i in _in_ if prev < i < v]
            if len(chk) > 0:
                c_segs.append([prev, v])
            chk1 = [i for i in _out_ if prev < i < v]
            if len(chk1) > 0:
                c_segs.append([prev, v])
        prev = v
    # c_segs[-1][-1] = 0  # set last point id to 0 to close the polygons
    # p_segs[-1][-1] = 0
    return c_segs, p_segs, bth


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
    out = _del_seq_dupl_pnts_(out)  # -- npg.npg_bool_hlp
    return out


# ---- ---------------------------
# ---- (1) general helpers
#
def orient_clockwise(geom):
    """Orient polygons so they are clockwise.

    Parameters
    ----------
    geom : list
        A list of polygon geometry arrays.

    Requires
    --------
    Ensure `_delete_seq_dups_` is run on the geometry first.
    """
    cw_ordered = []
    for i in geom:
        if _bit_area_(i) > 0.0:  # -- 2025_10_27  changed not sure
            cw_ordered.append(i)
        else:
            cw_ordered.append(i[::-1])
    return cw_ordered


def reorder_x_pnts(cl_n, x_pnts, pl_n=None):
    """Reorder the intersection points so that they follow the `clp` order.

    Parameters
    ----------
    cl_n, pl_n, x_pnts : arrays
        cl_n, pl_n are the results from `add_intersections`.  The first point
        in each sequence is the first intersection point.
        x_pnts are the intersections between them.  They will be in x-value
        order (lexicographic sort) when created.

    Returns
    -------
    The intersection points, in the sequential order than they occur on `cl_n`.

    Notes
    -----
    - `pl_n` is currently optional in case it is desired for other returns.
    - The gaps in `c_srt` or `p_srt` sequences are points that are either
      inside or outside the other geometry.
    - `x_srt0` is the position that the intersection points were found on
      `cl_n`.  `x_srt1` is the same for `pl_n`.
    - `x_srt0` is used to sort them in the order that they appear on `cl_n`.

    """
    c_srt, x_srt0 = np.nonzero((x_pnts == cl_n[:-1, None]).all(-1))
    x_new = x_pnts[x_srt0]
    if pl_n is not None:  # -- option return for the future
        p_srt, x_srt1 = np.nonzero((x_pnts == pl_n[:, None]).all(-1))
    return x_new


def pnt_connections(from_to,
                    as_dict=True,
                    keep_size=1,
                    exclude=None,
                    bidirectional=True,
                    non_sequential=False):
    """Return a dictionary of point connections given an array of connections.

    Parameters
    ----------
    from_to : array-like
        A 2d array or list of lists showing from-to connections.
    as_dict : boolean
        True, returns a dictionary of connections.
        False returns a list of from-to pairs.
    keep_size : integer
       dictionary value size to keep keys, is set to 2, then ids that form
       multiple intersections are returned (eg combos[:, 0])
    exclude : list
       A list of keys that you want excluded from the dictionary creation.  The
       most common case would be ids that are only connecting points and not
       ids that form branches.

       exclude = _out_.tolist() + _in_.tolist()  # points out or in not on

    bidirectional : boolean
        True, returns the output format as from-to and to-from connections.
    non_sequential : boolean
        True, omits expected sequences which are a point`s preceeding and
        following number.

    Notes
    -----
    It is assumed that `renumber_pnts` has been used on the input data to
    account for points with two different numbers.

    Example
    -------
    Given point from-to connections::

        ex = [[0, 1], [0, 16], [0, 19], [1, 0], [1, 2], [2, 16], [16, 19]]

        pnt_connections(ex, bidirectional=False)
        {0: [1, 16, 19], 1: [0, 2], 2: [16], 16: [19]}

        d = pnt_connections(ex, bidirectional=True)
        {0: [1, 16, 19], 1: [0, 2], 2: [1, 16], 16: [0, 2, 19], 19: [0, 16]}

    Alternately, the dictionary can be converted to a list of lists::

        lol = []
        for key, vals in d.items():
            lol.append([key, vals])

        [[0, [1, 16, 19]], [1, [0, 2]], [2, [1, 16]], [16, [0, 2, 19]],
         [19, [0, 16]]]

    Alternately produce a square array with 0, 1 indicating the connections::

        d = pnt_connections(frto, True, 1, None, True)
        z =np.zeros((23, 23), dtype='int')
        for k, v in d.items():
            z[k, k] = k
            z[k, v] = 1
        prn_(z)  # npg function to print the array

    """
    frto = from_to
    if bidirectional:
        _src_ = np.copy(frto)
        _ft_ = np.concatenate((_src_, _src_[:, [1, 0]]), axis=0)
    else:
        _ft_ = np.copy(frto)
    #
    if exclude is None:
        exclude = []
    uniq_frto = np.unique(_ft_, axis=0)
    if (uniq_frto[0] == [0, 0]).all():
        uniq_frto = uniq_frto[1:]
    frto2 = np.copy(uniq_frto)
    #
    # -- return the result as a list
    if not as_dict:
        return frto2
    # -- continue and return a dictionary
    d = {}
    remove = []
    for elem in frto2:
        if elem[0] in exclude:
            continue
        if elem[0] in d:
            if elem[0] != elem[1]:
                d[elem[0]].append(elem[1])
        else:
            d[elem[0]] = [elem[1]]
    for k, v in d.items():
        if len(v) > keep_size:
            if non_sequential:
                v = sorted(list(set(v)))
                vals = [i for i in v if i not in [k-1, k+1]]
                if len(vals) > 0:
                    d[k] = vals
                else:
                    remove.append(k)
            else:
                d[k] = list(v)
        else:
            remove.append(k)
    if len(remove) > 0:
        for i in remove:
            d.pop(i, None)
    return d  # , remove, unused


def seg_connections(c_on, p_on, _CP_, c_seq, p_seq, as_lists=True):
    """Return the segment connections which represent overlapping polygons.

    Parameters
    ----------
    c_on : array_like
        The point ids of the segment start/end points.
    c_seq, p_seq : array_like
        Sequences of point ids for an overlay polygon and the intersecting
        polygon.

    Notes
    -----
    `c_on` provides the keys to the initial dictionary and it represents the
    start/end point id values for c_seq.  p_seq must meet at one of the c_seq
    segments since the polygons overlap.

    """
    if as_lists:
        c_seq = [i.tolist() for i in c_seq]
        p_seq = [i.tolist() for i in p_seq]
    # de = {k: [v] for k, v in zip(c_on, c_seq)}  # -- initial key/vals orig
    # de = {k: {'e': [v[-1]], 'seg': [v]} for k, v in zip(c_on, c_seq)}
    # de = {k: {'s': [v[0]], 'e': [v[-1]], 'seg': [v]}
    #       for k, v in zip(c_on, c_seq)}
    # de = {k: {'se': [v[0], v[-1]], 'seg': [v]} for k, v in zip(c_on, c_seq)}
    #
    de = {k: {'ids': [v[0], v[-1]], 'ft': [v], 'tf': []}
          for k, v in zip(c_on, c_seq)}
    # -- fill in segments not seen during the initialization
    # -- original
    """
    for k, v in de.items():
        frst = v[0]
        if len(frst) > 2:
            de[frst[-1]].append(frst[::-1])
        # else:  # only add reverse when cycling through p_seq below
        #     de[frst[-1]].append(frst)
    """
    # -- append from the second sequence
    # try sorting p_on first
    p_on_srt = np.argsort(p_on)
    p_seq_srt = [p_seq[i] for i in p_on_srt]
    for cnt, seq in enumerate(p_seq_srt):
        seq = p_seq_srt[cnt]
        st, en = seq[0], seq[-1]  # for arrays, seq[[0, -1]]
        if st < en:
            de[st]['ft'].append(seq)
            if en not in de[st]['ids']:
                de[st]['ids'].append(en)
        else:
            if st not in de[en]['ids']:
                de[en]['ids'].append(st)
            if en not in de[st]['ids']:
                de[st]['ids'].append(en)
            de[en]['tf'].append(seq)
            de[st]['ft'].append(seq)
    #
    # get the sorted dictionary
    xys = _CP_[c_on]
    xys_lex = np.lexsort((-xys[:, 1], xys[:, 0]))  # sorted intersection points
    c_on_srted = c_on[xys_lex]
    srted = [de[i] for i in c_on_srted]  # get the sorted dictionary
    # de_srted = {k: de[k] for k in c_on_srted}  # as a dictionary
    #
    # -- try putting some together
    out = []
    dups = []
    closed = []
    for cnt, s in enumerate(srted):
        # -- produce all the pairs, 2 methods, use the second
        # pairs = [[s[i], s[j]] for i in range(len(s))
        #          for j in range(i + 1, len(s))]
        s = srted[cnt]['ft']
        pairs = [[s[0], s[j]] for j in range(1, len(s))]
        sub = []
        for cnt1, p in enumerate(pairs):
            frst, secn = pairs[cnt1]
            chk = len(set(frst + secn))  # -- eg [0, 1], [1, 0]
            if chk == 2:
                dups.append(frst)
            elif frst[0] == secn[0]:
                vals = frst[::-1] + secn
                if vals[0] == vals[-1]:
                    closed.append(vals)
                else:
                    sub.append(frst[::-1] + secn)
            elif frst[-1] == secn[0]:
                vals = frst + secn
                if vals[0] == vals[-1]:
                    closed.append(vals)
                else:
                    sub.append(vals)
            elif frst[0] == secn[-1]:
                vals = secn + frst
                if vals[0] == vals[-1]:
                    closed.append(vals)
                elif abs(vals[0] - vals[-1]) == 1:
                    closed.append(vals)
                else:
                    sub.append(vals)
        #
        if len(sub) > 0:
            out.extend(sub)
    # -- cleanup
    zz = np.asarray(closed, dtype='O')  # array shape not guaranteed
    closed_u = np.unique(zz).tolist()
    zz = np.asarray(dups)  # this will be an Nx2 array
    dups_u = np.unique(zz, axis=0).tolist()
    return de, closed_u, dups_u


def tri_array(frto):
    """Return line segments forming triangles when geometry is overlain.

    Parameters
    ----------
    frto : array_like
        Segment id values for the start and end of a 2 point line segment
        represented by `from`-`to` pairs.

    Returns
    -------
    If a triangle is found, a list of integer ids, for the points forming the
    segments, is returned.
    t1 = ([c_nxt, c_p] == c_cut[:, None]).all(-1).any()
    """
    pre_cur = np.concatenate((frto[:-1], frto[1:]), axis=1)
    # c_c_c_nxt = pre_cur[:, :2]
    c_nxt_c_p = pre_cur[:, [3, 0]]
    c_p_c_nxt = pre_cur[:, [0, 3]]
    # t0 = np.nonzero((c_c_c_nxt == frto[:, None]).all(-1))
    t1 = np.nonzero((c_nxt_c_p == frto[:, None]).all(-1))
    t2 = np.nonzero((c_p_c_nxt == frto[:, None]).all(-1))
    tri_s = []
    if sum([i.size for i in t1]) > 0:  # use size to determine there are any
        t1_0 = frto[t1[0]]
        t1_1 = frto[t1[1]][:, [1, 0]]
        tmp = np.concatenate((t1_1[:, [1, 0]], t1_0), axis=1)
        tri_s.append(tmp)
    if sum([i.size for i in t2]) > 0:  # ditto
        t2_0 = frto[t2[0]]
        t2_1 = frto[t2[1]][:, [1, 0]]
        tmp = np.concatenate((t2_0, t2_1), axis=1)
        tri_s.append(tmp)
    # -- tri_s will either be an empty list or geometries
    return tri_s


def rolling_match(seq_, _out_, _in_, extras=False):
    """Pair segments whose st-en values match to form closed-loops.

    Parameters
    ----------
    seq_ : array-like
        A list of arrays representing segment from-to id values.  There are
        various sort values, but `seq_srted` from polygon_overlay can be used
        or `sw_srt` from sweep.

    Notes
    -----
    To plot the geometry::

        ge = [_CP_[i] for i in out]
    """
    out = []  # the ids that form a closed polygon
    whr = []  # the ids where the pairs match in seq_u
    jn = []
    u_l_ = [seq_[0]]
    for cnt, r1 in enumerate(seq_[1:], 1):  # note, beginning at 1
        r0 = seq_[cnt - 1]
        r1 = seq_[cnt]
        f0, f1 = r0[[0, -1]]    # r0 pair st-en values
        f2, f3 = r1[[0, -1]]    # r1 pair st-en values
        chk = list(dict.fromkeys([f0, f1, f2, f3]))  # preserve order
        # chk = list(set([f0, f1, f2, f3]))
        if len(chk) == 2:
            if f0 == f2:
                v = r0.tolist() + r1[::-1].tolist()
            else:
                v = r0.tolist() + r1.tolist()
            if len(set(v)) > 2:
                whr.append([cnt - 1, cnt])
                out.append(v)
        if extras:
            #
            # new_u = [seq_u[i] for i in sum(whr, []) if len(seq_u[i]) == 2]
            #
            if len(chk) == 3:
                if f0 == f2:
                    jn.append(np.concatenate((r1[::-1],  r0)))
                elif f0 == f3:
                    jn.append(np.concatenate((r0[::-1], r1[::-1])))
                else:
                    jn.extend([r0, r1])
            else:
                u_l_.append(r1)
    if extras:
        return out, whr, jn
    return out, whr


def turns(frto, _CP_):
    """Classify overlay points by the angle formed by moving triplets.

    Parameters
    ----------
    frto : array_like
        The from-to pairs of the segment ids from the overlapping polygons.
    _CP_ : array
        The x-y coordinates of the resultant polygon.  The point ids have been
        renumbered and redundant points removed.

    Notes
    -----
    The approach uses the ideas from `_is_turn` from `np_geom_hlp`.

       _is_turn(p0, p1=None, p2=None, tol=1e-6)  imported above
       -1 : right
        0 : straight
        1 : left

    turns_arr::

        st - start point id
        c  - center point id
        en - end point id
        v  - turn value as above
                st   c  en  val
        array([[ 1,  0, 10,  1],
               [ 1,  0, 19,  0],
               [10,  0, 19,  1],
               [10,  0,  1, -1],
               [19,  0,  1,  0],
               [19,  0, 10, -1],
               [ 2,  3,  4,  0],
               [ 2,  3,  8, -1],
               [ 4,  3,  8,  1],
               [ 4,  3,  2,  0] .... snip

    """
    # for testing use (p02, c02)  2025-01-26
    #
    # use a dictionary to create the associated segments
    d = pnt_connections(frto, as_dict=True, keep_size=2, exclude=None,
                        bidirectional=True)
    #
    out = []
    for a, b in d.items():
        if len(b) == 3:
            b0, b1, b2 = b
            sl = [[b0, a, b1], [b0, a, b2],
                  [b1, a, b2], [b1, a, b0],
                  [b2, a, b0], [b2, a, b1]]
            out.extend(sl)
        elif len(b) == 4:
            b0, b1, b2, b3 = b
            sl = [[b0, a, b1], [b0, a, b2], [b0, a, b3],
                  [b1, a, b2], [b1, a, b3], [b1, a, b0],
                  [b2, a, b3], [b2, a, b0], [b2, a, b1],
                  [b3, a, b0], [b3, a, b1], [b3, a, b2]]
            out.extend(sl)
    zz = np.array(out)
    tol = 1e-4
    x0y0x1y1x2y2 = _CP_[zz].reshape((-1, 6))
    x0, y0, x1, y1, x2, y2 = x0y0x1y1x2y2.T
    # (x1_x0 * y3_y2) - (y1_y0 * x3_x2) + 0.0
    v = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    tmp = np.where(np.abs(v) < tol, 0., v)
    final = np.int32(np.sign(tmp))
    rgt_arr = zz[final == -1]
    lft_arr = zz[final == 1]
    strg_arr = zz[final == 0]
    turns_arr = np.concatenate((zz, final[:, None]), axis=1)
    #
    return rgt_arr, lft_arr, strg_arr, turns_arr


# ---- ---------------------------
# ---- (2) polygon overlay functions.
#
def renumber_pnts(cl_n, pl_n):
    """Return renumbered points from two intersected polygons.

    Parameters
    ----------
    cl_n, pl_n : arrays
        In normal useage, `pl_n` is the polygon being clipped by `cl_n`,

    Notes
    -----
    Used by `prepare` which is called by `polygon_overlay`.

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
    # -- remove the duplicate last/first point in the middle
    CP_ = np.concatenate((cl_n, pl_n), axis=0)  #
    # CP_ = np.concatenate((cl_n[:-1], pl_n), axis=0)  #
    # --
    # -- Initially number the sequential points.
    c_ids = np.arange(0, N_c + 1)  # cl_n ids - 1
    p_ids_old = np.arange(0, N_p + 1)  # --- try negative
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
    # -- the relabelled points.  cl_n is first, pl_n is 2nd
    new_ids[r1] = zz0[:, 1]
    new_ids[(new_ids == N_c).nonzero()] = 0   # renumber N_c and N_c+N_p+1 to 0
    new_ids[(new_ids == N_c + N_p + 1).nonzero()] = 0
    #
    # -- tricky, renumber the points that aren't duplicate of clp from N_c + 1
    tmp = np.nonzero(new_ids > N_c)[0]
    tmp0 = [N_c + 1 + i for i in range(len(tmp))]  # 2024-12-12 to fix stuff
    new_ids[tmp] = tmp0
    #
    old_new = np.concatenate((p_ids_old[:, None],
                              p_ids[:, None],
                              new_ids[N_c + 1:][:, None]),
                             axis=1)
    return new_ids, old_new, CP_


def prepare(ply_a, ply_b):
    """Prepare for overlay analysis.

    Parameters
    ----------
    ply_a, ply_b : arrays
        Arrays of point coordinates representing the boundary of two polygons.

    Requires
    --------
    `add_intersections` and `renumber_pnts`

    Returns
    -------
      - result0 : pl_n, cl_n, id_plcl, onConP, x_pnts, ps_info, cs_info
      - result1 : new_ids, old_new, CP_
      - result2 : _CP_, N_c, N_p, frto

    Notes
    -----
    See `polygon_overlay`. Used by `polygon_overlay` and `overlay_nx`.

    """
    result0 = add_intersections(
                ply_a, ply_b,
                roll_to_minX=True,
                p0_pgon=True,
                p1_pgon=True,
                class_ids=False
                )
    pl_n, cl_n = result0[:2]  # >2 id_plcl, onConP, x_pnts, ps_info, cs_info
    #
    result1 = renumber_pnts(cl_n, pl_n)
    cp_ids = result1[0]  # aka new_ids
    #
    z = pl_n[(cl_n != pl_n[:, None]).any(-1).all(-1)]
    _CP_ = np.concatenate((cl_n, z), axis=0)
    N_c = cl_n.shape[0] - 1  # noqa
    N_p = pl_n.shape[0] - 1  # noqa
    frto = np.concatenate((cp_ids[:-1][:, None], cp_ids[1:][:, None]), axis=1)
    frto = frto[frto[:, 0] != frto[:, 1]]  # -- remove 0,0 middle point
    #
    result2 = [_CP_, N_c, N_p, frto]
    args = [result0, result1, result2]
    return args


def bail(new, u_st_en):
    """Check to see if the segment is closed."""
    n_chk = np.array([[new[0], new[-1]], [new[-1], new[0]]])
    tst = np.logical_or(n_chk[0] == u_st_en,
                        n_chk[1] == u_st_en).all(-1)
    w = np.nonzero(tst)[0]
    chk_ = tst.any(-1)
    return chk_, w


def no_overlay_(p_a, c_a, N_p, N_c, ply_a, ply_b):
    """Return geometry."""
    print("no intersection points")
    if p_a == N_p + 1:
        if c_a == N_c + 1:
            print("both outside each other")
            return [ply_a, ply_b]
        else:
            print("clp completely inside poly")
            return [ply_a, ply_b[::-1]]
    elif (c_a == N_c + 1):
        if p_a == N_p + 1:
            print("both outside each other")
            return [ply_a, ply_b]
        else:
            print("ply completely inside clp")
            return [ply_a[::-1], ply_b]


def one_overlay_(chk0, chk1, c_on, p_on, pl_n, cl_n, ply_a, ply_b):
    """Return geometry."""
    if (pl_n[p_on[:2]] == cl_n[c_on[:2]]).all():
        if chk1:  # clp out
            geom = [ply_b]  # poly all in
            if chk0 == 2:  # -- drop the duplicate last
                pre = cl_n[::-1][:-1]
            else:
                pre = cl_n[::-1]  # -- keep it when there is one int. pnt
            # -- single point check
            post = pl_n[1:]
        else:
            geom = [ply_a]
            pre = pl_n[::-1][:-1]
            post = cl_n[1:]
        bth = np.concatenate((pre, post), axis=0)
        geom.append(bth)
        return geom


def nx_solve(frto):
    """Solve complex geometry using networkx.

    Parameters
    ----------
    frto : array
        the array representing the from-to id pairs with renumbering complete.

    Requires
    --------
    Called by `polygon_overlay`.

    Notes
    -----
    Combinations::

        : -1 clip outside poly
        :  0 poly outside clip :  erase
        :  1 poly inside clip :   clip
        :  2 area between clip and poly : hole
        :  3 identity : features or parts that overlap
        :  symmetrical diff -1 and 0

    """
    # -- networkx section
    # ---- Generate the edges
    #
    d0 = pnt_connections(from_to=frto, as_dict=False,
                         keep_size=1, exclude=None,
                         bidirectional=True,
                         non_sequential=False)
    #
    G = nx.Graph()  # ---- Create the graph and return the indices
    G.add_edges_from(d0.tolist())  # frto used to create d0
    cycles = nx.minimum_cycle_basis(G)  # not for directed graphs
    tmp_ = [[i[-1]] + i for i in cycles]
    frst = [i[0] for i in tmp_]
    whr = np.argsort(frst)
    result = [tmp_[i] for i in whr]
    return result


def segment_classify(result, c_in, c_out, p_in, p_out):
    """Classify the segments as to their type
    """
    in_c = set(c_in)
    out_c = set(c_out)
    in_p = set(p_in)
    out_p = set(p_out)
    c_in_segs = []
    c_out_segs = []
    p_in_segs = []
    p_out_segs = []
    oth_segs = []
    for i in result:
        if len(in_c) > 0:
            if len(in_c.intersection(i)) > 0:
                c_in_segs.append(i)
                continue
        if len(out_c) > 0:
            if len(out_c.intersection(i)) > 0:
                c_out_segs.append(i)
                continue
        if len(in_p) > 0:
            if len(in_p.intersection(i)) > 0:
                p_in_segs.append(i)
                continue
        if len(out_p) > 0:
            if len(out_p.intersection(i)) > 0:
                p_out_segs.append(i)
                continue
        oth_segs.append(i)
        # fix the above to class c_out
    return c_in_segs, c_out_segs, p_in_segs, p_out_segs, oth_segs


def wrap_(seq, c_subs, p_subs, _in_, _out_, _on_):  # rgt_arr):
    """Return segments.

    Parameters
    ----------
    seq : Sorted segments
        Use the sweep_srt on seq_srted from polygon_overlay to produce
        `sw_srt`.
    """

    def _chk_(row):
        """Check to see if start end indices are equal."""
        frst, secn = row
        eq_chk = (frst[[0, -1]] == secn[[0, -1]]).all(-1)
        return eq_chk

    def _order_(row):
        """Check the sort order, by length, of a 2d array or object array."""
        frst, secn = row
        if frst.ndim == secn.ndim:
            if len(frst) > len(secn):
                return row[::-1]
        return row

    # ---- start here
    sp = [i[0] for i in seq]   # -- get the first id value of each seq
    sp_spl = sp[0::2]  # every second pair of ids  !! really cool
    sp_arg_srt = np.argsort(sp_spl)  # -- used to resort the sequences at end
    #
    w = np.where(np.diff(sp) != 0)[0] + 1
    tmp = np.array_split(np.array(seq, dtype='O'),  w)
    # or for pairs that match
    # tmp = np.array_split(np.array(sw_srt, dtype='O'), len(sw_srt)//2)
    #
    # pairs = [_order_(i) for i in tmp]  # error found cor cr1 and flipped
    # -- original
    pairs = []
    for i in tmp:
        if len(i) > 1:
            pairs.append(_order_(i))
    # -- new for testing
    """
    pairs = []
    for cnt_, i in enumerate(tmp):
        if len(i) == 2:
            pairs.append(_order_(i))
        elif len(i) > 2:
            sub = list(zip(i[:-1], i[1:]))
            for j in sub:
                pairs.append(_order_(j))
    """
    segs_out = []
    clps = []
    _un_ = []
    mrk = 0
    mrkers = [mrk]
    clipper = []
    #
    for cnt, p in enumerate(pairs):
        p = pairs[cnt]
        frst, secn = pairs[cnt]
        chk = (frst[[0, -1]] == secn[[0, -1]]).all()
        chk_in = (secn == _in_[:, None]).any()        # check the longest
        chk_out = (secn == _out_[:, None]).any()      # seq which is second
        #
        if chk:
            if chk_out:
                clps.append(frst.tolist())  # append vs extend
            elif chk_in:
                clps.append(secn.tolist())  # append vs extend
            elif (frst == secn[:, None]).any(-1).all(-1):  # duplicate seg
                clps.append(frst.tolist())
                continue
            # now add the seg
            seg = secn.tolist() + frst[::-1].tolist()
            segs_out.append(seg)
        else:
            if (secn[-1] - frst[0]) == 2:
                _f = frst.tolist()
                _s = secn.tolist()
                seg = _f + _s[::-1]
                segs_out.append(seg)
                clps.extend(_f)  # was extend
                clps.extend([_f[1], _s[1]])  # was extend
            else:
                if (mrk in frst) and (mrkers[0] in secn):  # close on 0 check
                    seg = mrkers + secn.tolist()
                    clps.append(mrk)
                    clipper.append(clps)
                    segs_out.append(seg)
                elif (mrk in frst):  # splitting up clippers
                    new_ = clps[:-2]  # slice off the last pair added
                    new_.extend(frst.tolist())  # was extend
                    clipper.append(new_)
                    #
                    clps = clps[-2:]  # add the last two back in to a new list
                    mrk = frst[0]
                    mrkers.append(mrk)
                    _un_.extend(p)
                else:  # -- all left over
                    _un_.extend(p)
    # ---- assemble
    # piece it together,  sp_arg_srt is from above, original traversal order
    #   is returned rather than the lex order
    segs_in = []
    clip_geom = []
    if len(clipper) > 0:
        for i in clipper:
            t = []
            for j in i:
                if isinstance(j, list):
                    t.extend(j)
                else:
                    t.append(j)
            clip_geom.append(t)
    else:
        clp_segs = [clps[i] for i in sp_arg_srt]
        segs_in = np.concatenate(clp_segs)
    #
    if len(segs_out) == len(sp_arg_srt):
        segs_out = [segs_out[i] for i in sp_arg_srt]
    #
    result = [segs_out, clip_geom, segs_in]
    return result


# ---- polygon overlay algorithm ----
#
def polygon_overlay(ply_a, ply_b, asGeo=False):
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
    p0_ioo, p1_ioo : array-like
        polygon in-out-on p0_ioo for ply_a and p1_ioo for ply_b

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
               [ 0,  0],     on,  on
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

    # ---- turns calculation
    rgt_arr, lft_arr, strg_arr, turns_arr = turns(frto, _CP_)

    """
    #
    # ---- start here ----
    args = prepare(ply_a, ply_b)
    result0, result1, result2 = args
    #
    # -- result0
    # x_pnts are lex sorted
    pl_n, cl_n, id_plcl, onConP, x_pnts, ps_info, cs_info = result0
    p_out, p_on, p_in, pl_ioo = ps_info
    c_out, c_on, c_in, cl_ioo = cs_info
    #
    # -- result1
    cp_ids, old_new_ids, CP_ = result1
    #
    # -- result2
    _CP_, N_c, N_p, frto = result2
    #
    # ---- CP_ and _CP_
    # -- CP_ from result1, _CP_ from result2
    #  cl_n are the clipper polygons pnts
    #  z are the poly pnts that are not duplicates/intersections of clipper
    #  _CP_ are the coordinates of cl_n and those of pl_n that aren't in cl_n
    #
    # ---- plotting
    """
    # label cl_n
    l0 = frto[:N_c, 0]
    z0 = _CP_[l0]
    # label renumbered pl_n
    l1 = frto[N_c:, 0]
    z1 = _CP_[l1]
    # this works
    zz = np.concatenate((z0, z1), axis=0)
    zzl = np.concatenate((l0, l1))
    plot_polygons(zz, labels=zzl)

    # individual plots
    plot_polygons(z0, True, True, labels=l0)
    plot_polygons(z1, True, True, labels=l1)

    """
    #
    # ---- (1) get p_subs with renumbered id values and produce p_ft
    p_ft = np.concatenate((p_on[:-1][:, None], p_on[1:][:, None]), axis=1)
    pl_ioo2 = np.copy(pl_ioo)
    pl_ioo2[:, 0] = old_new_ids[:, -1]  # -- use the renumbered ids
    z_ = np.concatenate((pl_ioo, pl_ioo2), axis=1)
    p_subs = [z_[i[0]:i[1] + 1][:, -2:] for i in p_ft]
    #
    # ---- (2) get c_ft, c_subs and p_ft2
    cl_ioo[-1, 0] = 0
    c_ft = np.concatenate((c_on[:-1][:, None], c_on[1:][:, None]), axis=1)
    c_subs = [cl_ioo[i[0]:i[1] + 1] for i in c_ft]
    c_ft[-1, 1] = 0  # set last `to` to 0
    c_subs[-1][-1][0] = 0  # set the last point to return to 0
    #
    # -- renumber p`s using new values
    _out_ = []
    for i in [p_on, p_in, p_out]:
        _whr_ = np.nonzero(i == old_new_ids[:, 0][:, None])[0]
        _out_.append(old_new_ids[_whr_, -1])  # get the new values
    p_on, p_in, p_out = _out_
    p_ft2 = list(zip(p_on[:-1], p_on[1:]))  # removed +1 to `to`
    p_ft2 = np.array(p_ft2)
    #
    # ---- (3) get c_ft_v, c_subs and  split_at_intersections
    c_vals = [sub[1][1] for sub in c_subs]
    c_ft_v = np.concatenate((c_ft, np.array(c_vals)[:, None]), axis=1)
    c_ft_v[-1, 1] = 0  # set last `to` to 0
    #
    # ---- (4) get p_ft_v, p_subs_full, then p_subs
    p_subs_full = [pl_ioo[i[0]:i[1] + 1] for i in p_ft]  # added +1 to `to`
    p_vals = [sub[1][1] for sub in p_subs_full]
    p_ft_v = np.concatenate((p_ft, np.array(p_vals)[:, None]), axis=1)
    p_ft_v[-1, 1] = 0  # set last `to` to 0
    #
    # ---- (5) generate combos and reclass p_on, p_in, p_out
    # -- "c_fr c_to v ... v  p_fr2 p_to2
    combos = np.zeros((c_ft_v.shape[0], 6), dtype='int')
    combos[:, :3] = c_ft_v
    combos[:, 3] = p_ft_v[:, -1]
    combos[:, 4:] = p_ft2  # set final 2 columns of combos
    #

    # ---- (6) sorting section,  -- all sequences, and seq_srted creation
    # the ids of the intersection pnts
    # x_ids, idx = np.unique(combos[:, :2], return_index=True)
    # ls = np.lexsort((-x_pnts[:, 1], x_pnts[:, 0]))
    #
    c_seq = [i[:, 0] for i in c_subs if i.size > 0]   # get the stuff needed
    p_seq = [i[:, 0] for i in p_subs if i.size > 0]
    all_seq = c_seq + p_seq
    #
    # -- added the following to ensure that reversed segments are captured
    """ uncomment if wanting to use swapped values
    p_swapped = [i[::-1] for i in p_seq if i[0] > i[1]]
    all_seq = c_seq + p_seq + p_swapped  # -- see _all_ below
    """
    all_st_en = [i[[0, -1]] for i in all_seq]
    all_st_en = np.array(all_st_en)
    #
    # -- lexsort on the minimum `x` of start-end point and max y, and an
    #    argsort on the first sequence id value,
    xs_2 = np.array([_CP_[i][:, 0][[0, -1]] for i in all_seq])  # st-en x`s
    xs_2_lex = np.lexsort((-xs_2[:, 1], xs_2[:, 0]))  # sorted
    #
    seq_srted_tmp = [all_seq[i] for i in xs_2_lex]  # all seqs sorted by min x
    #
    s_ = np.argsort([i[0] for i in seq_srted_tmp])  # sort by id
    #
    # ---- alternative use
    # _seq_srted_ = sequence_srt(seq, _CP_)
    seq_srted = [seq_srted_tmp[i] for i in s_]  # final paired segments
    #
    # -- get in/out/on corrections and split the sequence to identifys
    #    singletons. This replicates `sequences.py`
    #    Fix c_on and p_on since first and last are equal as are their ids.
    c_on = c_on[:-1]  # c_on[-1] = 0
    p_on = p_on[:-1]  # p_on[-1] = 0
    _out_ = np.concatenate((c_out, p_out))
    _in_ = np.concatenate((c_in, p_in))
    _on_ = np.unique(np.concatenate((c_on, p_on)))
    #
    # frto_2 = np.concatenate((c_ft, p_ft2), axis=0)  # !!! keep
    #
    c_a, c_b, c_c = [len(i) for i in [c_out, c_on, c_in]]  # clp
    p_a, p_b, p_c = [len(i) for i in [p_out, p_on, p_in]]  # ply
    chk0 = len(x_pnts)
    chk1 = c_a == 0
    chk2 = p_a == 0
    #
    # ---- (10) process geometry ----
    # ---- -- no intersections
    if chk0 == 0:
        geom = no_overlay_(p_a, c_a, N_p, N_c, ply_a, ply_b)
        return geom
    # ---- -- single intersection
    elif chk0 <= 2 and (chk1 or chk2):
        print("\n-- _one_overlay --")
        # -- returns the actual geometry
        geom = one_overlay_(chk0, chk1, c_on, p_on, pl_n, cl_n, ply_a, ply_b)
        return geom
    #
    # ---- -- simple, but multiple intersections
    if (c_ft == p_ft2).all():  # -- 2025-06-06 use seq_srted ... E, C case
        result = wrap_(seq_srted, c_subs, p_subs, _in_, _out_, _on_)
        segs_out, clip_geom, segs_in = result
        geom = []
        _add0 = [_CP_[i] for i in segs_out]
        if isinstance(segs_in, np.ndarray):
            segs_in = [segs_in]
        _add1 = [_CP_[i] for i in segs_in]
        _add2 = [_CP_[i] for i in clip_geom]
        geom = _add0 + _add1 + _add2
        geom = [_del_seq_dupl_pnts_(i) for i in geom]
    # ---- -- complex, use networkx
    else:  # -- geom sorts lexicographically
        #
        result = nx_solve(frto)  # or wrap_
        #
        # -- classify the result
        ret_ = segment_classify(result, c_in, c_out, p_in, p_out)
        c_in_segs, c_out_segs, p_in_segs, p_out_segs, oth_segs = ret_
        #
        tmp = [_CP_[i] for i in result]
        t_mins = [np.min(i[:, 0]) for i in tmp]
        s_1 = np.argsort(t_mins)
        geom = [tmp[i] for i in s_1]
        geom = orient_clockwise(geom)
        # geom = npg.roll_arrays(geom)  # `roll_arrays` optional
        # optionally get the ID values back using
        #  [np.nonzero((g == _CP_[:, None]).all(-1).any(-1))[0].tolist()
        #   for g in geom]
    if asGeo:
        geom = npg.arrays_to_Geo(geom, kind=2, info=None, to_origin=False)
    return geom

    #
    # works for
    #
    # (p00, c00), (c00, p00) : wrap_      : one_overlay_ :2 x_pnts on line
    # (p00, c01), (c01, p00) : wrap_      : 2 x_pnts, cuts go out and in
    # (p00, c02), (c02, p00) : wrap_      : 4 x_pnts, double in/out cuts
    # (p00, c03), (c03, p00) : wrap_      : 4 x_pnts, extra point on line
    # (p01, c00), (c00, p01) : wrap_      : 4 x_pnts, v-intersection
    # (p01, c01), (c01, p01) : wrap_      :
    # (p01, c02), (c02, p01) : wrap_      :
    # (p01, c03), (c03, p01) : wrap_      :
    # (p02, c00), (c00, p02) : wrap_      :
    # (p02, c01), (c01, p02) : wrap_      :
    # (p02, c02), (c02, p02) : wrap_      : handles c_ft != p_ft2
    # (p02, c03), (c03, p02) : wrap_      : 7 x_pnts
    # (p03, c00), (c00, p03) : wrap_      : one_overlay_ : 1 xsection pnt
    # (p03, c01), (c01, p03) : wrap_      : one_overlay_
    # (p03, c02), (c02, p03) : wrap_      : one_overlay_
    # (p03, c03), (c03, p03) : wrap_      : one_overlay_
    # (aoi, aoi0)            : wrap_      : 4 x_pnts aoi0 is rotated inside aoi
    # (pl_, cl_)             : wrap_      :
    # (edgy1, eclip)         : wrap_      :
    # (B, K), (K, B)         : wrap_      :
    # (d0_, d1_)             : wrap_      : or nx_overlay
    # E, C                   : wrap_      : or nx_overlay  2025-06-06
    # E, F                   : wrap_      : or nx_overlay  2025-06-08
    # C, Io                  : wrap_
    # E, Io                  : wrap_
    # (E, d0_), (d0_, E)     : nx_overlay
    # M, W                   : nx_overlay  2025-06-09
    # poly0, poly1           : nx_overlay  2025-06-22
    # E, d1_                 : nx_overlay  2025-09-01  tried others to no avail

    # last worked above
    # (A, C), (C, A) : nx_overlay
    # (C, K)  (K, C) : nx_overlay similar to A,C


# ---------------------------
# ---- (3) polygon overlay operations.
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
# ---- (4) append geometry
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
# ---- (5) split at intersections
#
def split_at_intersections(ply_a, ply_b, sort_by_x=False, as_array=False):
    """Return polygon features split at their intersections points.

    Parameters
    ----------
    ply_a, ply_b : arrays
        Polygon points as Nx2 arrays representing their boundaries.  It is
        assumed that they overlap.
    as_array : boolean
        True, returns object arrays of the segmented inputs.  False returns
        the outputs as a list of arrays.

    Returns
    -------
    Poly features split at their intersection points.
    Use `plot_polylines(*seg*)` to see the segments
    """

    def _srt_fwd_(seq, ft_vals):
        """Return the sequence sorted so that x increases."""
        tm = []
        # ft2 = []
        for cnt, i in enumerate(seq):
            chk = i[0][0] <= i[-1][0]
            if chk:
                tm.append(i)
                # ft2.append(ft_vals[cnt])
            else:
                tm.append(i[::-1])
                # ft2.append(ft_vals[cnt][::-1])
        return tm  # ft2

    r = add_intersections(
            ply_a, ply_b,
            roll_to_minX=True,
            p0_pgon=True,
            p1_pgon=True,
            class_ids=False
            )
    pl_n, cl_n, id_plcl, onConP, x_pnts, ps_info, cs_info = r
    p_out, p_on, p_in, p0_ioo = ps_info
    c_out, c_on, c_in, p1_ioo = cs_info
    c_ft = np.concatenate((c_on[:-1][:, None], c_on[1:][:, None]), axis=1)
    p_ft = np.concatenate((p_on[:-1][:, None], p_on[1:][:, None]), axis=1)
    c_segs = [cl_n[f:t + 1] for f, t in c_ft]
    p_segs = [pl_n[f:t + 1] for f, t in p_ft]
    #
    # --
    if sort_by_x:
        c_segs = _srt_fwd_(c_segs, c_ft)
        p_segs = _srt_fwd_(p_segs, p_ft)

    z = pl_n[(cl_n != pl_n[:, None]).any(-1).all(-1)]
    _CP_ = np.concatenate((cl_n, z), axis=0)
    xys = _CP_[c_on]
    xys_lex = np.lexsort((-xys[:, 1], xys[:, 0]))  # sorted intersection points
    c_on_srted = c_on[xys_lex]
    #
    if as_array:
        c_segs = np.asarray(c_segs, dtype='O')
        p_segs = np.asarray(p_segs, dtype='O')
    return p_segs, c_segs, c_on_srted


# ---- ---------------------------
# ---- (6) dissolve shared boundaries
#
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
            tmp = _del_seq_dupl_pnts_(tmp)  # -- npg.npg_bool_hlp
            out.append(tmp)
            a = b[:]  # -- assign the last poly as the first and repeat
        elif is_last:
            tmp = npg.roll_arrays(ret)
            tmp = _del_seq_dupl_pnts_(tmp)
            out.append(tmp)
        else:
            a = ret[:]
    if asGeo:
        out = npGeo.arrays_to_Geo(out, 2, "dissolved", False)
        out = npGeo.roll_coords(out)
    return out


def union_over(ply_a, ply_b):
    """Return union of geometry.  needs to be fixed."""
    result = polygon_overlay(ply_a, ply_b)
    final, idx, over_, cin_, hole_, symm_, erase_, erase_r = result
    out = merge_(over_, ply_b)
    return out


# ---- ---------------------------
# ---- (7) adjacency
#
def adjacency_array(d, to_array=True):
    """Return an array of connections from `pnt_connections` results.

    Parameters
    ----------
    d : dictionary
        Connection ids of the points forming the bounds of adjacent polygons.
    to_array : boolean
        True, returns a nxn numpy array.  False, returns an nxn list-of-lists.
    """
    kys = sorted(d.keys())
    sze = len(kys)
    arr = [[0]*sze for i in range(sze)]
    for a, b in [(kys.index(a), kys.index(b))
                 for a, row in d.items()
                 for b in row]:
        arr[a][b] = 2 if (a == b) else 1
    if to_array:
        return np.asarray(arr)
    return arr


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
# ---- (8) merge geometry
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


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")
