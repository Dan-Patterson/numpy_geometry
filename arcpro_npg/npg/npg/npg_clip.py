# -*- coding: utf-8 -*-
# noqa: D205, D400, F401, F403
r"""
---------
npg_clip
---------

----

Script :
    npg_clip.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2021-06-19

Purpose
-------
Functions for clipping polygons.

Call sequence::

    # c_in, _cut_, inouton and _wn_clip_ can all be used without poly_clip
    # inouton and _wn_clip have no other dependencies.
    poly_clip
    ...   |__ c_in
    ...        |__ _cut_   __|
    ...        |             |  _wn_clip_
    ...        |__ _split_ __|

Notes
-----
Examples::

    p0 = [[[0., 8.], [1., 6.]], [[0., 8.], [2., 8]],
          [[0., 8.], [4., 9]], [[0., 8.], [2., 10.]]]
    poly = [np.array(p) for p in p0]
    pnts = np.array(poly).reshape(-1, 2)
    poly = np.array([[0.0, 5.0], [2., 10.], [4., 10.], [2., 5.0], [0., 5.]])
    plot_polygons([pnts, poly])

"""

import sys
# from textwrap import dedent
import numpy as np

# -- optional numpy imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

# if 'npg' not in list(locals().keys()):
#     import npg
# from npg_helpers import _to_lists_
from npg import npg_plots
from npg.npg_plots import plot_polygons

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['common_extent', 'uniq_1d', 'uniq_2d', 'inouton', '_in_out_',
           'poly_clip']
__helpers__ = ['_onseg_', '_is_on_', '_side_', 'inouton', '_cut_',
               '_wn_clip_']


# ----------------------------------------------------------------------------
# ---- (1) helpers
#
def common_extent(a, b):
    """Return the extent overlap for two polygons as L, B, R, T or None."""
    ext0 = np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))
    ext1 = np.concatenate((np.min(b, axis=0), np.max(b, axis=0)))
    es = np.concatenate((ext0[None, :], ext1[None, :]), axis=0)
    maxs = np.max(es, axis=0)
    mins = np.min(es, axis=0)
    L, B = maxs[:2]
    R, T = mins[2:]
    if (L <= R) and (B <= T):
        return (True, np.array([L, B, R, T]))  # (x1, y1, x2, y2)
    return (False, None)


def uniq_1d(arr):  # *** remove when publishing clip
    """Return mini 1D unique."""
    mask = np.empty(arr.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = arr[1:] != arr[:-1]
    return arr[mask]


def uniq_2d(arr, return_sorted=False):
    """Return mini `unique` for 2D coordinates.  Derived from np.unique."""

    def _reshape_uniq_(uniq, dt, shp):
        n = len(uniq)
        uniq = uniq.view(dt)
        uniq = uniq.reshape(n, *shp[1:])
        uniq = np.moveaxis(uniq, 0, 0)
        return uniq

    shp = arr.shape
    dt = arr.dtype
    st_arr = arr.view(dt.descr * shp[1])
    ar = st_arr.flatten()
    if return_sorted:
        perm = ar.argsort(kind='mergesort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    ret = aux[mask]
    uniq = _reshape_uniq_(ret, dt, shp)
    if return_sorted:  # return_index in unique
        return uniq, perm[mask]
    return uniq


# ---- (2) standalone functions
#
def inouton(pnts, poly, do_crossings=False):
    r"""Return polygon overlay results.  These are `in`, `out` or `on` points.

    Parameters
    ----------
    pnts : array_like
        The polygon points that are being queried (Nx2 shape).
        These can also be any points that would yield points inside, outside
        the `poly`gon feature or intersecting with its boundary.
    poly : array_like
        The polygon feature with shape Nx2 with the first and last points
        being the same.

    Returns
    -------
    The following ndarrays::

    - x_type  : crossing type, inside-outside or outside-inside
    - whr     : segment pairs where intersections occur
    - in_ids  : point id number for inside points
    - inside  : points inside the polygon
    - out_ids : point id number for outside points
    - outside : points outside the polygon
    - x_pnts  : intersection/crossing points
    - eq      : points in common to both shapes
    - wn_vals : winding number value for each point, 0 for outside -1 for in

    Counterclockwise check for 3 points::

        def ccw(A, B, C):
            '''Tests whether the turn formed by A, B, and C is ccw'''
            return (B.x - A.x) * (C.y - A.y) > (B.y - A.y) * (C.x - A.x)

    Notes
    -----
    The commented out `crossings` section can be used to return the crossing
    type.  Add `x_ings` to the function output if this is the case.

    Point in/out of polygon uses `winding number` approach.
    If `denom` is 0, then the lines are coincedent or parallel.

    *intersection section*::

        denom = x1_x0 * y3_y2 - y1_y0 * x3_x2
        s_num = x1_x0 * y0_y2 - y1_y0 * x0_x2
        t_num = x3_x2 * y0_y2 - y3_y2 * x0_y2
        (s_num < 0.) == (denom > 0.0)
        (t_num < 0.) == (denom > 0.0)
        t4 = np.logical_or((s_num > denom) == (denom > 0.0),
                           (t_num > denom) == (denom > 0.0))
    ----

    """
    pnts, poly = [i.XY if hasattr(i, "IFT") else i for i in [pnts, poly]]
    # -- point equality check, not_eq = ~eq_
    eq_ = (pnts[:, None] == poly).all(-1).any(-1)  # pnts equal those on poly
    # -- pnts
    x0, y0 = pnts[:-1].T
    x2, y2 = poly[:-1].T
    x3, y3 = poly[1:].T  # x3 not needed
    y3 = poly[1:, 1]
    #
    x1_x0, y1_y0 = (pnts[1:] - pnts[:-1]).T  # deltas for pnts
    x3_x2, y3_y2 = (poly[1:] - poly[:-1]).T  # deltas for poly
    x3_x2 = x3_x2[:, None]      # reshape poly deltas
    y3_y2 = y3_y2[:, None]
    x0_x2 = (x0 - x2[:, None])  # deltas between pnts/poly x and y
    y0_y2 = (y0 - y2[:, None])
    # x0_x2, y0_y2 = (pnts[:-1, None] - poly[:-1]).T  # is slower
    #
    a_0 = y0_y2 * x3_x2
    a_1 = x0_x2 * y3_y2
    b_0 = x1_x0 * y0_y2
    b_1 = y1_y0 * x0_x2
    #
    denom = (x1_x0 * y3_y2) - (y1_y0 * x3_x2)
    #
    # winding number section
    # --
    a_num = a_0 - a_1  # t_num
    b_num = b_0 - b_1  # s_num
    chk1 = (y0_y2 >= 0.0)
    chk2 = np.less(y0, y3[:, None])
    chk3 = np.sign(a_num).astype(np.int32)  # t_num replaced with a_num
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=0, dtype=np.int32)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=0, dtype=np.int32)
    wn_vals = pos - neg
    # fix wn_vals to include duplicate start-end
    wn_vals = np.concatenate((wn_vals, np.array([wn_vals[0]])))
    in_ids = np.nonzero(wn_vals)[0]
    out_ids = np.nonzero(wn_vals + 1)[0]
    inside = pnts[in_ids]    # pnts in poly
    outside = pnts[out_ids]  # pnts outside poly
    x_type = np.vstack((wn_vals[:-1], wn_vals[1:])).T
    #
    # crossings
    # --
    if do_crossings:
        x1 = pnts[1:, 0]  # x1, y1 = pnts[1:].T
        y1 = pnts[1:, 1]  # x1, y1 used by crossing
        y1_y2 = (y1 - y2[:, None])
        x1_x2 = (x1 - x2[:, None])
        c_0 = y1_y2 * x3_x2  # used for x_ings
        c_1 = x1_x2 * y3_y2
        a = a_0 <= a_1  # b = b_0 <= b_1
        c = c_0 <= c_1
        w0 = np.logical_and(a, c) * 2     # both on right  (T, T)
        w1 = np.logical_and(a, ~c) * 1      # start on right (T, F)
        w2 = np.logical_and(~a, c) * -1     # start on left  (F, T)
        w3 = np.logical_and(~a, ~c) * -2    # both on left   (F, F)
        x_ings = (w0 + w1 + w2 + w3)  # whr = np.argwhere(abs(x_ings) == 1)
    #
    # intersections
    # --
    with np.errstate(all="ignore"):  # ignore all errors
        u_a = a_num / denom
        u_b = b_num / denom
        z0 = np.logical_and(u_a >= 0., u_a <= 1.)  # equal to `id_vals`
        z1 = np.logical_and(u_b >= 0., u_b <= 1.)
        both = z0 & z1
        # (u_a * x1_x0[None, :] + x0[None, :])[both]
        # (u_a * y1_y0[None, :] + y0[None, :])[both]
        xs = (u_a * x1_x0 + x0)[both]
        ys = (u_a * y1_y0 + y0)[both]
    x_pnts = []
    if xs.size > 0:
        x_pnts = np.concatenate((xs[:, None], ys[:, None]), axis=1)
    eq = pnts[eq_]
    whr = np.array(np.nonzero(both)).T  # or np.argwhere(abs(x_ings) == 1)
    args = [x_type, whr, in_ids, out_ids, inside, outside, x_pnts, eq, wn_vals]
    if do_crossings:
        args.append(x_ings)
    return args


def _wn_clip_(pnts, poly, all_info=True):
    """Return points in polygon using `winding number`.  See `inouton`.

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
    """
    def _w_(a, b, all_info):
        """Return winding number and other values."""
        x0, y0 = a[:-1].T  # point `from` coordinates
        x1, y1 = a[1:].T   # point `to` coordinates
        x1_x0, y1_y0 = (a[1:] - a[:-1]).T
        #
        x2, y2 = b[:-1].T  # polygon `from` coordinates
        x3, y3 = b[1:].T   # polygon `to` coordinates
        x3_x2, y3_y2 = (b[1:] - b[:-1]).T
        # reshape poly deltas
        x3_x2 = x3_x2[:, None]
        y3_y2 = y3_y2[:, None]
        # deltas between pnts/poly x and y
        x0_x2 = (x0 - x2[:, None])
        y0_y2 = y0 - y2[:, None]
        #
        a_0 = y0_y2 * x3_x2
        a_1 = x0_x2 * y3_y2
        b_0 = x1_x0 * y0_y2
        b_1 = y1_y0 * x0_x2
        a_num = a_0 - a_1
        b_num = b_0 - b_1
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
            denom = (x1_x0 * y3_y2) - (y1_y0 * x3_x2)
            return wn_, denom, x0, y0, x1_x0, y1_y0, a_num, b_num
        return wn_

    def _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0):
        """Return the intersection."""
        with np.errstate(all="ignore"):  # ignore all errors
            u_a = a_num / denom
            u_b = b_num / denom
            z0 = np.logical_and(u_a >= 0., u_a <= 1.)  # equal to `id_vals`
            z1 = np.logical_and(u_b >= 0., u_b <= 1.)
            both = z0 & z1
            xs = (u_a * x1_x0 + x0)[both]
            ys = (u_a * y1_y0 + y0)[both]
        x_pnts = []
        if xs.size > 0:
            x_pnts = np.concatenate((xs[:, None], ys[:, None]), axis=1)
        # eq = pnts[eq_]
        whr = np.array(np.nonzero(both)).T
        return whr, x_pnts
    #
    # Use `_w_` and `_xsect_` to determine pnts in poly
    wn_, denom, x0, y0, x1_x0, y1_y0, a_num, b_num = _w_(pnts, poly, True)
    whr, x_pnts = _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0)
    #
    p_in_c = np.nonzero(wn_)[0]
    p_out_c = np.nonzero(wn_ + 1)[0]
    p_in = pnts[p_in_c]    # pnts in poly
    p_out = pnts[p_out_c]  # pnts outside poly
    x_type = np.concatenate((wn_[:-1, None], wn_[1:, None]), axis=1)
    #
    wn2_ = _w_(poly, pnts, False)  # get poly points in other geometry
    c_in_p = np.nonzero(wn2_)[0]
    c_in = poly[c_in_p]
    vals = [p_in, p_out, c_in, x_pnts, p_in_c, p_out_c, c_in_p, x_type,
            wn_, whr]
    return vals


# ---- (3) poly_split
#
def _split_(poly, splitter):
    """Return.

    References
    ----------
    `<https://community.esri.com/t5/python-blog/patterns-sequences-occurrence
    -and-position/ba-p/902504>`_.

    Requires
    --------
    - `uniq_2d
    - `_wn_clip_` does the work of getting the points in polygons and
    intersection points.

    Example
    -------
    The following is an attempt to find points that may appear before an
    inclusion point. It uses sequences as given in the reference.::

        >>> p_in_c = np.array([0, 1, 2, 5], dtype=int64)  # pnts in polygon ids
        >>> stepsize = 1
        >>> w = np.where(np.diff(p_in_c) != stepsize)[0] + 1
        >>> sequences = np.split(p_in_c, w)
        ... [array([0, 1, 2], dtype=int64), array([5], dtype=int64)]
        >>> in_seq = np.nonzero([pl_ in i for i in sequences])[0]
        ... array([0], dtype=int64)
        >>> idx = sequences[in_seq[0]]
        >>> to_add = inside[idx]
        >>> to_add  # -- points in the clipping polygon in sequence
        ... array([[  1.00,  11.50],
        ...        [  2.50,  13.00],
        ...        [  4.00,  12.50]])
    """
    # out_ = []
    in_ = []
    vals = _wn_clip_(poly, splitter)
    p_in, p_out, c_in, x_pnts, p_in_c, p_out_c, c_in_p, x_type, wn_, whr = vals
    # 120 µs ± 574 ns
    if p_in_c.size == 0:  # -- no intersection -- bail.
        return None, None
    #
    stepsize = 1  # pnt ids incrementing by 1
    spl_whr = np.where(np.diff(p_in_c) != stepsize)[0] + 1
    all_sequences = np.array_split(p_in_c, spl_whr)
    sequences = [s for s in all_sequences if s.size > 1]
    visited = []
    # 13 µs ± 161 ns
    for i, w in enumerate(whr):
        cl_, pl_ = w
        visited.append(pl_)
        c_0, c_1 = splitter[cl_: cl_ + 2]
        p_0, p_1 = poly[pl_: pl_ + 2]
        bit1 = []
        chk0 = pl_ in p_in_c        # poly pnt id in splitter
        chk1 = pl_ + 1 in p_in_c    # next poly pnt in splitter
        chk2 = cl_ in c_in_p        # splitter pnt in poly
        chk3 = cl_ + 1 in c_in_p    # next splitter pnt in poly
        #
        if chk2:  # -- start split inside poly
            bit1 = c_0
        if chk3:  # -- end split inside poly
            bit1 = c_1
        #
        # sequence check
        in_seq = np.nonzero([pl_ in i for i in sequences])[0]  # trailing
        in_seq2 = np.nonzero([pl_ + 1 in i for i in sequences])[0]  # leading
        to_add = None
        if in_seq.size > 0:
            idx = sequences[in_seq[0]]
            to_add = poly[idx]  # sequence of poly points inside splitter
        elif in_seq2.size > 0:
            idx = sequences[in_seq2[0]]
            to_add = poly[idx]
        #
        if chk0 and ~chk1:  # -- inside, outside
            if chk2:
                in_seg = np.asarray([p_0, x_pnts[i], bit1])
            else:
                in_seg = np.asarray([p_0, x_pnts[i]])
        elif ~chk0 and chk1:  # -- outside, inside
            if chk2:
                in_seg = np.asarray([bit1, x_pnts[i], p_1])
            else:
                in_seg = np.asarray([x_pnts[i], p_1])
        elif ~chk0 and ~chk1:  # -- outside, outside
            # in_seg = None  # *** may 6th altered doesn't work for b0, c0
            in_seg = x_pnts[i][None, :]
        elif (c_0 in p_in) and (c_1 in p_in):  # -- clip inside, inside
            in_seg = np.asarray([c_0, c_1])
        else:
            # out_seg = np.concatenate((c_0[None, :], c_1[None, i]), axis=0)
            print("no condition met")
        # exit check
        if to_add is not None:
            if in_seq.size > 0:  # prepend sequential points
                in_seg = np.vstack((to_add, in_seg))
            else:                # append sequential points
                in_seg = np.vstack((in_seg, to_add))
        if in_seg is not None:
            if pl_ == visited[0]:  # check to see if you are back to the start
                in_.insert(visited.index(pl_), in_seg)
            else:
                in_.append(in_seg)
    idx = np.argsort(whr[:, 1])
    """
    whr2 = whr[idx]  # sorted indices
    x_x = x_pnts[idx]  # intersection points sorted
    """
    in_srted = [in_[i] for i in idx]
    z = np.vstack(in_srted)
    #
    u, idx = uniq_2d(z, True)  # unique for 2D coordinates
    # u, idx = np.unique(z, True, axis=0)
    idxs = np.sort(idx)
    final = z[idxs]  # ***** OR ****
    return final, in_  # , in_srted  # z, idxs


def _cut1_(pair_chunk, chunks, ps, sp, ft_p_shape, p0):  # ** not used
    """Return. # fix needs p0."""
    # construct inside    # works with c0, b0
    pairs = pair_chunk[0]
    # diff = ps.shape[0] - ft_p_shape[0]
    clp_ply = chunks[0].ravel()
    if clp_ply[1] == 0:  # rolled back to zero, needed for c0, b0
        p0 = 1
    else:
        p0 = clp_ply[1] + 1
    z0 = ps.copy()
    ps_new = np.atleast_2d(pairs[::-1].ravel())
    z0[0, :2] = pairs[0]
    z0[-1, -2:] = pairs[1]
    z0 = np.concatenate((z0, ps_new))
    pieces = [z0[0, :2], z0[:, -2:]]
    p2 = [np.atleast_2d(i) for i in pieces]
    z_0 = np.concatenate(p2)
    #
    # construct outside
    z1a = ps[:p0]
    z1a[-1, -2:] = pairs[0]
    z1b = np.atleast_2d(pairs.ravel())
    z1c = ps[-1]
    z1c[:2] = pairs[1]
    p3 = [np.atleast_2d(i) for i in [z1a, z1b, z1c]]
    z1 = np.concatenate(p3)
    z_1 = np.concatenate((z1[0, :2][None, :], z1[:, -2:]))
    return z_0, z_1, sp


# ---- (4) poly_cut  ** working

def _cut_1_(whr, ft_p, ft_c, x_pnts):
    """One line crossing."""
    clip_fr, clip_to = whr[:, 0]
    poly_to, poly_fr = whr[:, 1]
    in_c = ft_c[clip_fr: clip_to + 1]
    in_c[0, :2] = x_pnts[0]
    in_c[-1, -2:] = x_pnts[1]
    in_p = ft_p[poly_fr: poly_to + 1]
    in_p[0, :2] = x_pnts[1]
    in_p[-1, -2:] = x_pnts[0]
    in_ = np.vstack([in_c, in_p])
    in_ = np.concatenate((in_[0, :2][None, :], in_[:, -2:]))
    #
    out_ = []
    polys = [in_, out_]
    # test = [polys]
    return polys  # , test


def _cut_(poly, splitter):
    """Return the result of a polygon split.

    Parameters
    ----------
    poly, splitter : array_like
        The geometry with `splitter` being a line or lines which cross the
        segments of poly.

        `poly` is a polygon feature.

    Requires
    --------
    `_wn_clip_`, `_cut_1_`, `uniq_1d`

    Notes
    -----
    .. Note::
        note to self... to clip polylines, just don't close the geometry

    Split into chunks.

    >>> spl = np.array([[ 0.0, 7.5], [10.0, 2.5]])
    >>> sq = np. array([[ 0.0, 0.0], [ 0.0, 10.0], [10.0, 10.0],
    ...                 [10.0, 0.0], [ 0.0,  0.0]])
    >>> result = _c_(spl, sq)
    >>> result
    ... array([[ 0.0, 0.0], [ 0.0, 7.5], [10.0, 2.5],
    ...        [10.0, 0.0], [ 0.0, 0.0]])

    Reassemble an from-to array back to an N-2 array of coordinates. Also,
    split, intersection points into pairs.

    >>> coords = np.concatenate((r0[0, :2][None, :], r0[:, -2:]), axis=0)
    >>> from_to = [o_i[i: i+2] for i in range(0, len(o_i), 2)]
    """
    ft_p = np.concatenate((poly[:-1], poly[1:]), axis=1)
    ft_c = np.concatenate((splitter[:-1], splitter[1:]), axis=1)
    vals = _wn_clip_(poly, splitter)
    p_in, p_out, c_in, x_pnts, p_in_c, p_out_c, c_in_p, x_type, wn_, whr = vals
    if x_pnts is None:
        return None
    crossings = uniq_1d(whr[:, 0])
    # -- more than one splitter
    #
    polys = []
    ft_p_shape = ft_p.shape  # input shape for polygon from-to pairs
    ps = ft_p.copy()
    sp = ft_c.copy()
    test = []
    added_ = []
    # s_min, s_max = np.sort(whr[:, 1])[[0, -1]]  # last segment check
    # Nseg = len(crossings) - 1
    for i, seg in enumerate(crossings):  # clipper ids that cross poly
        w = (whr[:, 0] == seg)
        spl_X_ply = whr[w]           # splitter crosses polygon `spl_X_ply`
        all_pairs = x_pnts[w]        # the intersection points on that segment
        chunks = [spl_X_ply[i: i + 2] for i in range(0, len(spl_X_ply), 2)]
        pair_chunk = [all_pairs[i: i + 2] for i in range(0, len(spl_X_ply), 2)]
        sub = []
        #
        # last clipper only has 1 chunk
        if (len(chunks) == 1) and (len(chunks[0]) == 1):
            p0 = spl_X_ply[0]
            z_0, z_1 = _cut_1_(whr, ft_p, ft_c, x_pnts)  # z_0, z_1 = in, out
            ps = z_0.copy()
            sub.append([z_0, z_1])
            polys += [z_0, z_1]  # .extend([z_0, z_1])
# =============================================================================
#         elif (i == Nseg) and (len(chunks) == 1):
#             p0 = spl_X_ply[0]
#             whr2 = spl_X_ply
#             z_0, z_1 = _cut_1_(whr, ft_p, ft_c, x_pnts)
#             ps = z_0.copy()
#             sub.append([z_0, z_1])
#             polys += [z_0, z_1]  # polys.extend([z_0, z_1])
# =============================================================================
        else:
            # not last and/or has more than 1 chunk
            for j, clp_ply in enumerate(chunks):
                pairs = pair_chunk[j]   # intersection pnt or pnts
                diff = ps.shape[0] - ft_p_shape[0]  # ft_p.shape[0]
                added_.append(diff)
                p0p1 = clp_ply[:, 1] + diff  # only works for intersection
                if p0p1.size == 1:
                    p0 = p0p1[0]
                    p1 = p0 + 1
                else:
                    p0, p1 = p0p1
                if (p1 - p0) >= 2:         # remove extra rows
                    sp = ps[p0: p1 + 1]    # but update sp first
                    ps = np.concatenate((ps[:p0 + 1], ps[p1:]))  # needed by E
                else:
                    sp = ps[p0: p1 + 1]
                    ps = ps.copy()
                ps[p0, -2:] = pairs[0]
                ps_new = pairs[:2].ravel()
                if p0 + 1 == ps.shape[0]:
                    p_next = 0
                else:
                    p_next = p0 + 1
                ps[p_next, :2] = pairs[1]
                #
                # -- assemble outside z0
                pieces = [ps[:(p_next)], ps_new, ps[(p_next):]]
                z0 = [np.atleast_2d(i) for i in pieces]
                z0 = np.concatenate(z0, axis=0)
                z_0 = np.concatenate((z0[0, :2][None, :], z0[:, -2:]))
                #
                # -- assemble inside z1
                ps = z0.copy()  # copy, reuse for the next iteration
                sp[0, :2] = pairs[0]
                sp[-1, -2:] = pairs[1]
                sp_new = np.concatenate((pairs[1], pairs[0]))
                z1 = np.concatenate((sp, sp_new[None, :]), axis=0)
                z_1 = np.concatenate((z1[0, :2][None, :], z1[:, -2:]))
                sub.append([z_0, z_1])
                polys.extend([z_0, z_1])
        test.append(np.asarray(sub, dtype='O').squeeze())
    return polys, x_pnts, p_in, p_out, whr, test


"""
n = 0
m = 0
p = 0
z0 = []
zz = np.vstack((outside, inside))
ids = np.arange(0, len(out_ids) + len(x_pnts))
for i in ids:
    if i in out_ids:
        is_in = False
        z0.append(outside[n])
        n += 1
    elif i in in_ids:
        is_in = True
        z0.append(inside[m])
        m += 1
    else:
        z0.append(x_pnts[p])

"""


def _in_out_(poly, splitter, keep_in=True):  # --- c_in with _cut_
    """Return split parts.  Calls `_cut_` for all splitters.

    Requires
    --------
    uniq_2d
    Returns
    -------
    The split geometry is divided into two parts:  those inside and outside
    the clipping polygon.

    - `_cut_` does the actual work of splitting and sorting.
    - `_wn_clip_` does checking to ensure geometry are inside or outside.
    """

    def final_test(arr, x_in):
        """Return."""
        arr = uniq_2d(arr, False)  # use this for 2D coordinates
        # arr = np.unique(arr, axis=0)
        z0_s = np.sum((arr == x_in[:, None]).all(-1).any(-1))
        # z1_s = np.sum((arr == x_out[:, None]).all(-1).any(-1))
        is_in = False
        if z0_s == arr.shape[0]:
            is_in = True
        return is_in
    # polys, x_pnts, p_in, p_out, whr, test
    polys, x_pnts, inside, outside, whr, test = _cut_(poly, splitter)
    # if vals is None:
    #     return None
    # polys, x_pnts, whr, inside, outside, test = vals
    if len(polys) == 0:
        return None
    # x_out = np.vstack((x_pnts, outside))  # outside and intersection points
    x_in = uniq_2d(np.vstack((x_pnts, inside)), False)
    # x_in = np.unique(np.vstack((x_pnts, inside)), axis=0)
    in_ = []
    # out_ = []
    idx = np.nonzero([final_test(i, x_in) for i in polys])[0]
    in_ = [polys[i] for i in idx]
    # for i, inout in enumerate(polys):
    #     [final_test(i, x_in) for i in polys]
    #     if n == 2:
    #         is_in, is_out = final_test(inout, x_in)  # changed
    #         if is_in:
    #             in_.append(is_in)
    #         if is_out is not None:
    #             out_.append(is_out)
    #     elif n > 2:
    #         for j in range(0, n, 2):
    #             is_in, is_out = final_test(inout[j], x_in, outside)
    #             if is_in is not None:
    #                 in_.append(is_in)
    #             if is_out is not None:
    #                 out_.append(is_out)
    return in_  # , out_

    # vals = _cut_(splitter, geom)
    # if vals is None:
    #     return None
    # polys, x_pnts, whr, inside, outside = vals
    # # x_pnts = np.unique(x_pnts, axis=0)
    # if len(polys) == 0:
    #     return None
    # in_ = []
    # out_ = []
    # for i, arr in enumerate(polys):
    #     some_out = final_test(arr, outside)
    #     if some_out:
    #         out_.append(arr)
    #     else:
    #         in_.append(arr)
    # return in_, out_


# ---- (5) multi shape version  ** not final
#
def poly_clip(clippers, polys, inside=True):
    """Return the inside or outside clipped geometry.

    Parameters
    ----------
    clippers, polys : list or array_like
        The geometries representing the clipping features and the polygon
        geometry.  A list or array with ndim > 2 is expected for both
    inside : boolean
        True, keeps the inside geometry.  False for outside, the clip extents.

    Requires
    --------
    `common_extent`, `_in_out_`

    Returns
    -------
    The shared_extent between the clipping and poly features.  The geometry
    inside/outside depending on the `inside` boolean.

    Example
    -------
    >>> poly_clip([c0], [E])  # lists provided
    """
    shared_extent = []
    to_keep = []
    for i, clp in enumerate(clippers):
        for j, poly in enumerate(polys):
            shared, extent = common_extent(clp, poly)
            if shared:
                shared_extent.append([i, j, extent])
                inout = _in_out_(clp, poly, keep_in=inside)
                if inout is None:
                    pass
                in_, out_ = inout
                if inside:
                    if len(in_) > 0:
                        to_keep.extend(in_)
                else:
                    if len(out_) > 0:
                        to_keep.extend(out_)
    # remember that shared_extent and out2 should be sliced prior to plotting
    # ie. out2[0] is the result of clipping one polygon by one clipper
    return shared_extent, to_keep


# ---- (6) not used, but keep
#
def _ct_(poly, splitter):  # ** not used
    """Return the result of a polygon split.

    Parameters
    ----------
    splitter, geom : array_like
        The geometry with `splitter` being a line or lines which cross the
        segments of geom.

        `geom` is a polygon feature.

    Requires
    --------
    `_wn_clip_`
    Notes
    -----
    .. Note::
        note to self... to clip polylines, just don't close the geometry

    Split into chunks.

    >>> spl = np.array([[ 0.0, 7.5], [10.0, 2.5]])
    >>> sq = np. array([[ 0.0, 0.0], [ 0.0, 10.0], [10.0, 10.0],
    ...                 [10.0, 0.0], [ 0.0,  0.0]])
    >>> result = _c_(spl, sq)
    >>> result
    ... array([[ 0.0, 0.0], [ 0.0, 7.5], [10.0, 2.5],
    ...        [10.0, 0.0], [ 0.0, 0.0]])

    Reassemble a from-to array back to an N-2 array of coordinates. Also,
    split, intersection points into pairs.

    >>> coords = np.concatenate((r0[0, :2][None, :], r0[:, -2:]), axis=0)
    >>> from_to = [o_i[i: i+2] for i in range(0, len(o_i), 2)]
    """
    vals = _wn_clip_(poly, splitter)
    p_in, p_out, c_in, x_pnts, p_in_c, p_out_c, c_in_p, x_type, wn_, whr = vals
    #
    if x_pnts is None:
        return None
    in_polys = []
    # out_polys = []
    cnt = 0
    visited = []
    # stepsize = 1  # pnt ids incrementing by 1
    # spl_whr = np.where(np.diff(p_in_c) != stepsize)[0] + 1
    # all_sequences = np.array_split(p_in_c, spl_whr)
    # sequences = [s for s in all_sequences if s.size > 1]
    for i, ft in enumerate(x_type):  # clipper ids that cross poly
        sub = []
        f, t = ft
        p0, p1 = cnt, cnt + 1
        X_P = x_pnts[cnt]
        visited.append(p0)
        if (f == 0) and (t == 0):
            if (p0 == p_in_c[:, None]).any():
                sub.append(poly[p0])
            if (p1 == p_in_c[:, None]).any():
                sub.append(poly[p1])
            sub.append(X_P)
        elif (f == 0) and (t == -1):
            sub.append(X_P)
            sub.append(poly[p1])
        elif (f == -1) and (t == 0):
            sub.append(poly[p1])
            sub.append(X_P)
        elif (f == -1) and (t == -1):
            if (p0 == p_in_c[:, None]).any():
                sub.append(poly[p0])
            if (p1 == p_in_c[:, None]).any():
                sub.append(poly[p1])
            # visited.extend([p0, p1])  # maybe just p1
            # if np.isin(p0, visited, invert=True).any():
            #     sub.extend([poly[p0], poly[p1]])
        if np.sum(ft) != -2:
            cnt += 1
        if sub:
            in_polys.append(sub)
    return in_polys


def _onseg_(poly, pnts, tol=1.0e-12):  # ** not used
    """Determine whether points (pnts) lies on line segments (poly).

    Parameters
    ----------
    poly : array_like
        Polyline/polygon objects represented as Nx2 arrays of points.  Every
        segment of poly features are tested.
    pnts : array_like
        The x, y coordinates of point locations.
    tol : float
        The tolerance of accepting whether a point is on a segment.

    Requires
    --------
    `_collinear_` and `_within_`.  A default tolerance of 1.0e-12 to avoid
    floating point issues.

    Returns
    -------
    An array of the ids of the `pnts` versus `poly` segments.

    Example
    -------
    >>> ft_p = np.array([  5.000,  15.000,   7.000,  14.000])
    >>> pnt = np.array([  6.350,  14.325])
    >>> _onseg_(ft_p[:2], ft_p[-2:], pnt)
    ... True

    """
    def _collinear_(a, b, pnts):
        """Return True if all points lie on the same line.

        This is the same `_side_` check.
        >>> r = (x1 - x0) * (y[:, None] - y0) - (y1 - y0) * (x[:, None] - x0)
        """
        epsilon = tol
        x0, y0 = a.T
        x1, y1 = b.T
        x, y = pnts.T
        r0 = (x1 - x0) * (y[:, None] - y0)
        r1 = (x[:, None] - x0) * (y1 - y0)
        r01 = r0 - r1
        return abs(r01) < epsilon

    def _within_(a, b, pnts):
        """Return."""
        x = pnts[:, 0][:, None]
        y = pnts[:, 1][:, None]
        x0, y0 = a.T
        x1, y1 = b.T
        z0 = (x0 <= x) & (x <= x1)
        z1 = (y0 <= y) & (y <= y1)
        # or
        z2 = (x1 <= x) & (x <= x0)
        z3 = (y1 <= y) & (y <= y0)
        res = np.logical_or(z0 & z1, z2 & z3)
        return res
    # --
    a = poly[:-1]
    b = poly[1:]
    r0 = _collinear_(a, b, pnts)
    r1 = _within_(a, b, pnts)
    whr = r0 & r1
    return np.array(np.nonzero(whr)).T


def _is_on_(a, b, pnt):  # ** not used
    """Return whether a point, (pnt) lies on a line segment (a->b).

    Or the degenerate case where all 3 points coincide.

    See Also
    --------
    `_onseg_` is the multi poly-pnt version.
    """

    def collinear(a, b, pnt):
        """Return True if all points lie on the same line."""
        epsilon = 1.0e-12
        r = (b[0] - a[0]) * (pnt[1] - a[1]) - (pnt[0] - a[0]) * (b[1] - a[1])
        return abs(r) < epsilon

    def within(a, pnt, b):
        """Return true if `pnt` is between `a` and `b` (inclusive)."""
        return a <= pnt <= b or b <= pnt <= a
    # --
    chk0 = collinear(a, b, pnt)
    chk1 = within(a[0], pnt[0], b[0])
    chk2 = within(a[1], pnt[1], b[1])
    return ((chk0 and chk1) if a[0] != b[0] else chk2)


def _side_(pnts, poly):  # ** not used
    r"""Return points inside, outside or equal/crossing a convex poly feature.

    Notes
    -----
    See `_wn_clip_` as another option to return more information.

    >>> `r` == diff_ in _wn_ used in chk3
    >>> `r` == t_num = a_0 - a_1 ... in previous equations
    >>> r_lt0 = r < 0, r_gt0 = ~r_lt0, to yield =>  (r_lt0 * -1) - (r_gt0 + 0)
    >>> (r < 0).all(-1)  # just the boolean locations
    ... array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    >>> (r < 0).all(-1).nonzero()[0]  # the index numbers
    ... array([2, 3, 4, 6, 7], dtype=int64)
    """
    if pnts.ndim < 2:
        pnts = np.atleast_2d(pnts)
    x0, y0 = pnts.T
    x2, y2 = poly[:-1].T  # poly segment start points
    x3, y3 = poly[1:].T   # poly segment end points
    r = (x3 - x2) * (y0[:, None] - y2) - (y3 - y2) * (x0[:, None] - x2)
    # -- from _wn_, winding numbers for concave/convex poly
    chk1 = ((y0[:, None] - y2) >= 0.)
    chk2 = (y0[:, None] < y3)
    chk3 = np.sign(r).astype(int)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn_vals = pos - neg
    in_ = pnts[np.nonzero(wn_vals)]
    inside = pnts[(r < 0).all(axis=-1)]  # all must be True along row, convex
    outside = pnts[(r > 0).any(-1)]      # any must be True along row
    equal_ = pnts[(r == 0).any(-1)]      # ditto
    return r, in_, inside, outside, equal_


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
