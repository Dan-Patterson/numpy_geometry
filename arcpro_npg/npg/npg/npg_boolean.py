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
    2023-04-07

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
import npg
from npg.npGeo import roll_arrays
from npg.npg_helpers import a_eq_b
from npg.npg_plots import plot_polygons  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]

__all__ = ['add_intersections', 'clip_poly', 'find_segment_overlaps',
           'split_poly']
__helpers__ = ['_add_pnts_', 'del_seq_pnts', '_roll_', 'side',
               '_sort_on_line_', '_w_', '_wn_clip_', '_node_type_',
               '_prepare_']


# ---- (1) private helpers
#
def _add_pnts_(ply0, ply1, x_pnts, whr):
    """Return input arrays with intersections added to their lines.

    Parameters
    ----------
    ply0, ply1 : array_like
        N-2 arrays of clockwise ordered points representing poly* features.
    x_pnts : array_like
        The intersection points.
    whr : array_like
        The id locations where the line points intersect the polygon segments.

    Requires
    --------
    `_wn_clip_` is used to generate the intersection points and segments of the
    poly* features that they intersect on (this is the `whr`ere parameter).
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
    p_ = np.concatenate((ply0[:-1], ply0[1:]), axis=1).reshape((-1, 2, 2))
    p_ = list(p_)
    c_ = np.concatenate((ply1[:-1], ply1[1:]), axis=1).reshape((-1, 2, 2))
    c_ = list(c_)
    for cnt, cp in enumerate(whr):
        cl, pl = cp  # print(f"cnt {cnt}  cp {cp}") add this below to see order
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
    dt = [(f'f{i}', dt_in) for i in range(arr.shape[1])]
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
    arrs : list of Geo arrays or ndarrays.  Two arrays are expected.

    Notes
    -----
    Rolls the coordinates of the Geo array or ndarray to put the start/end
    points as close to the lower-left of the ring extent as possible.

    If a single array is passed, a single array is returned otherwise a list
    of arrays.
    """
    # --
    if not isinstance(arrs, (list, tuple)):
        arrs = [arrs]
    out = []
    for ar in arrs:
        chk = npg.is_Geo(ar)
        if chk:
            out.append(npg.roll_coords(ar))
        else:
            out.append(roll_arrays(ar))
    return out


# ---- (2) prepare for boolean operations
#
def _w_(a, b, all_info):
    """Return winding number and other values."""
    x0, y0 = a[:-1].T   # point `from` coordinates
    # x1, y1 = a[1:].T  # point `to` coordinates
    x1_x0, y1_y0 = (a[1:] - a[:-1]).T
    #
    x2, y2 = b[:-1].T  # clip polygon `from` coordinates
    x3, y3 = b[1:].T   # clip polygon `to` coordinates
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
    #
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
    if all_info:  # denom of determinant
        denom = (x1_x0 * y3_y2) - (y1_y0 * x3_x2) + 0.0
        return wn_, denom, x0, y0, x1_x0, y1_y0, a_num, b_num
    return wn_


def _wn_clip_(pnts, poly, all_info=True):
    """Return points in a polygon or on its perimeter, using `winding number`.

    Information required to determine intersection points is also provided.
    These data are used for clipping the polygon represented by `pnts` by the
    clipping polygon `poly`.

    Parameters
    ----------
    pnts, poly : array_like
        Geometries represent the points and polygons.  `pnts` is assumed to be
        the polygon being clipped and `poly` is the clipping polygon.
    all_info : boolean
        True, returns points in polygons, the in and out id values, the
        crossing type and winding number.  False, simply returns the winding
        number, with 0 being outside points and -1 being inside points for a
        clockwise-oriented polygon.

    Notes
    -----
    Negative and positive zero np.NZERO, np.PZERO == 0.0.
    `The denominator of this expression is the (squared) distance between
    P1 and P2.  The numerator is twice the area of the triangle with its
    vertices at the three points, (x0, y0), p1 and p2.` Wikipedia
    With p1, p2 defining a line and x0,y0 a point.

    Other
    -----
    z = np.asarray(np.nonzero(npg.eucl_dist(a, b) == 0.)).T
    a[z[:, 0]] and b[z[:, 1]] return the points from both arrays that have a
    distance of 0.0 and they intersect.
    """

    def _xsect_(a_num, b_num, denom, x1_x0, y1_y0, x0, y0):
        """Return the intersections and their id values."""
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
    # -- defaults
    px_in_c = []
    cx_in_p = []
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
        px_in_c = sorted(list(set(p_in_c + p_eq_c + p_eq_x)))
    if c_eq_p or c_eq_x:  # c_in_p + (p_eq_c, c_eq_x)
        cx_in_p = sorted(list(set(c_in_p + c_eq_p + c_eq_x)))
    return px_in_c, p_in_c, p_eq_c, p_eq_x, cx_in_p, c_in_p, c_eq_p, c_eq_x


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

    - `roll_arrays` (optional)
    - `_wn_clip_`
    - `_node_type_`
    - `_add_pnts_`
    - `_del_seq_pnts_`

    Returns
    -------
    The following are returned::

    - x_pnts : intersection points
    - a0, a1 : arrays rolled to first intersection,
    - a0_new, a1_new : rolled with intersections added on,
    - args : optional arguments
    -   px_in_c, cx_in_p : poly/intersection in c and clip/intersection in p
    -   p_in_c, c_in_p : poly in clip, clip in poly
    -   c_eq_p, c_eq_x : clip equals poly or intersection
    -   p_eq_c, p_eq_x : poly equals clip or intersection

    Notes
    -----
    The sequence is as follows::

      - Roll the arrays so that their first coordinate is the closest
        to the lower left of the geometry extent.
      - Determine points inside each other`s geometry.
      - Classify the points.
      - Add intersection points to both geometries.
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
    # px_in_c, cx_in_p, p_in_c, c_in_p, c_eq_p, c_eq_x, p_eq_c, p_eq_x = args
    a0_new, a1_new = _add_pnts_(a0, a1, x_pnts, whr)
    x_pnts = _del_seq_pnts_(x_pnts, poly=False)
    a0_new = _del_seq_pnts_(np.concatenate((a0_new), axis=0), poly=is_0)
    a1_new = _del_seq_pnts_(np.concatenate((a1_new), axis=0), poly=is_1)
    return x_pnts, a0, a1, a0_new, a1_new, args


def _in_out_(w0, as_array=False):
    """Return the array indices.

    Requires
    --------
    `add_intersections` is used to produce the following::

      # -- outside and inside ids
      w0 = p0_ioo[p0_ioo[:, 1] < 0, 0]  # slice the indices where p0_ioo < 0
      w1 = p0_ioo[p0_ioo[:, 1] > 0, 0]  # slice the indices where p0_ioo > 0
      w2 = p1_ioo[p1_ioo[:, 1] < 0, 0]  # slice the indices where p1_ioo < 0
      w3 = p1_ioo[p1_ioo[:, 1] > 0, 0]

    The following can be determined, but w* needs to be checked that they
    are not empty::

      Pout = subs(w0)  # poly outside clip
      Pin = subs(w1)  # poly inside clip
      Cout = subs(w2)  # clip outside poly
      Cin = subs(w3)  # clip inside poly

    """
    if w0.size == 0:
        print("Empty array.")
        return []
    out = []
    sub = [w0[0] - 1, w0[0]]
    for cnt, i in enumerate(w0[1:], 0):
        prev = w0[cnt]
        if i - prev == 1:
            sub.append(i)
        else:
            sub.append(prev + 1)
            out.append(sub)
            sub = [i - 1, i]
    if cnt == len(w0) - 2:
        if len(sub) >= 2:
            sub.append(i + 1)
        out.append(sub)
    if as_array:
        if len(out) >= 1:
            out = np.concatenate([np.array(i) for i in out])
    return out


# ---- (3) add intersection points
#
def add_intersections(p0, p1, polygons=[True, True]):
    """Return input polygons with intersections points added.

    Parameters
    ----------
    p0, p1 : array_like
       The overlapping poly features.
    polygons : list/tuple
        True, the input geometry is a polygon feature, False, for polyline.
        Some operations permit polygon and polyline inputs, so you can alter
        `polygons=[True, False]` if the first is a polygon and the second a
        polyline.

    Requires
    --------
    `_add_pnts_`, `_del_seq_pnts_`, `_w_`, `_wn_clip_`

    Returns
    -------
    The poly features:
      - rotated to the first intersection point (`p0_n, p1_n`),
      - their respective indices from the start (`id_01`),
      - the intersection points (`x_pnts`),
      - the classified indices for each polygon as to whether the points are
        outside, on or inside the other

    p0_n, p1_n : arrays
        The input arrays, rotated to their first intersection point and those
        points added to their perimeter.

    x_pnts : array
        The intersection points with sequential duplicates removed.

    id_01 : array
        Where the polygons intersect with p0, p1 representing the ids in their
        respective column.

    p0_ioo, p1_ioo : arrays
        Poly id values and whether the point is outside the other (-1), an
        intersection point on the boundary (0) or inside the other polygon (1).

    Example
    -------
    Using `E` and `d0_` as `p0_` and `p1_`::

        w0 =  np.nonzero(p0_ioo[:, 1] <= 0)[0]  # outside and on
        z0 = p0_n[w0]
        w1 =  np.nonzero(p1_ioo[:, 1] >= 0)[0]  # inside and on
        z1 = p1_n[w1]
        z0_id, z1_id = np.nonzero((z0 == z1[:, None]).all(-1))
        id_s10 = np.concatenate((z0_id[:, None], z1_id[:, None]), axis=1)  # or
        id_s01 = np.concatenate((z1_id[:, None], z0_id[:, None]), axis=1)
        id_s01srt = id_s01[np.argsort(id_s01[:, 0])]
        plot_polygons([z0, z1])
    """
    def _classify_(p0_, p1_, id_):
        """Return poly points classified as inside, on or outside."""
        p_ids = np.arange(0, p0_.shape[0])
        p_neq = sorted(list(set(p_ids).difference(set(id_))))
        p_neq = np.array(p_neq)  # convert to array
        z = p0_[p_neq]  # check the points not on, but may be in or out
        p_w = _w_(z, p1_, False)  # use _w_ from _wn_clip_
        p_i = np.nonzero(p_w)[0]
        p_o = np.nonzero(p_w + 1)[0]
        p_in = p_neq[p_i]   # in ids
        p_out = p_neq[p_o]  # out ids
        p_ioo = np.zeros(p0_.shape, dtype='int')  # create the output indices
        p_ioo[:, 0] = p_ids  # p0 ids (i)n (o)ut (o)n -> ``ioo``
        p_ioo[p_in, 1] = 1
        p_ioo[p_out, 1] = -1
        return p_ioo
    # --
    # id0, id1 : where p0_n, p1_n are equal
    # p0_neq, p1_neq : p0_n, p1_n not equal, may be in or out
    # p0_i, p1_i   : p0_n, p1_n in their counterpart
    is_0, is_1 = polygons
    p0, p1 = _roll_([p0, p1])  # roll the arrays so they are closest to LL
    # -- get intersection information
    vals = _wn_clip_(p0, p1, all_info=True)
    x_pnts, pInc, cInp, x_type, whr = vals
    p0_n, p1_n = _add_pnts_(p0, p1, x_pnts, whr)
    p0_n = _del_seq_pnts_(np.concatenate((p0_n), axis=0), poly=is_0)
    p1_n = _del_seq_pnts_(np.concatenate((p1_n), axis=0), poly=is_1)
    x_pnts = _del_seq_pnts_(x_pnts, False)  # True, if wanting a polygon
    # try rolling the x_pnts closest to its lower left
    x_pnts = _roll_(x_pnts)[0]
    # -- locate the roll coordinates
    r0 = np.nonzero((x_pnts[0] == p0_n[:, None]).all(-1).any(-1))[0]
    r1 = np.nonzero((x_pnts[0] == p1_n[:, None]).all(-1).any(-1))[0]
    v0, v1 = r0[0], r1[0]
    p0_n = np.concatenate((p0_n[v0:-1], p0_n[:v0], [p0_n[v0]]), axis=0)
    p1_n = np.concatenate((p1_n[v1:-1], p1_n[:v1], [p1_n[v1]]), axis=0)
    # -- fix the id pairing
    p0N = len(p0_n) - 1
    p1N = len(p1_n) - 1
    id0, id1 = np.nonzero((p1_n == p0_n[:, None]).all(-1))
    whr0 = np.nonzero(id0 == p0N)[0]
    whr1 = np.nonzero(id1 == p1N)[0]
    id0[whr0] = 0
    id1[whr1] = 0  # slice off the first and last
    id_01 = np.concatenate((id0[:, None], id1[:, None]), axis=1)[1:-1]
    id_01[-1] = [p0N, p1N]
    #
    p0_ioo = _classify_(p0_n, p1_n, id0)
    p1_ioo = _classify_(p1_n, p0_n, id1)
    return p0_n, p1_n, id_01, x_pnts, p0_ioo, p1_ioo


# ---- (4) clip polygons
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

    local helpers : `_roll_`, `_wn_clip_`, `_node_type_`, `_add_pnts_`,
    `_del_seq_pnts_`
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
    pl_n, cl_n, id_01, x_pnts, pl_ioo, cl_ioo = add_intersections(poly, clp)
    #
    srt = np.argsort(id_01[:, 1])
    inCinP = id_01[srt][:, [1, 0]]
    #
    w0 = pl_ioo[pl_ioo[:, 1] < 0, 0]  # slice the indices where pl_ioo < 0
    w1 = pl_ioo[pl_ioo[:, 1] > 0, 0]  # slice the indices where pl_ioo > 0
    w2 = cl_ioo[cl_ioo[:, 1] < 0, 0]  # slice the indices where cl_ioo < 0
    w3 = cl_ioo[cl_ioo[:, 1] > 0, 0]  # slice the indices where cl_ioo > 0
    #
    p_out_on = _in_out_(w0, as_array=True)  # poly outside clip
    p_in_on = _in_out_(w1, as_array=True)   # poly inside clip
    c_out_on = _in_out_(w2, as_array=True)  # clip outside poly
    c_in_on = _in_out_(w3, as_array=True)   # clip inside poly
    #
    pinside = sorted(list(set(p_in_on).union(inCinP[:, 1])))
    cinside = sorted(list(set(c_in_on).union(inCinP[:, 0])))
    # result = _prepare_([poly, clp], roll=False, polygons=[True, True])
    # x_pnts, pl, cl, pl_new, cl_new, args = result
    # px_in_c, p_in_c, p_eq_c, p_eq_x, cx_in_p, c_in_p, c_eq_p, c_eq_x = args
    #
    # -- locate first intersection and roll geometry to it.
    # r0 = np.nonzero((x_pnts[0] == pl_new[:, None]).all(-1).any(-1))[0]
    # r1 = np.nonzero((x_pnts[0] == cl_new[:, None]).all(-1).any(-1))[0]
    # fix_out = []
    # out2 = []
    # nums = [r0[0], r1[0]]
    # --
    # fixes = [[px_in_c, p_in_c, p_eq_c, p_eq_x],
    #          [cx_in_p, c_in_p, c_eq_p, c_eq_x]]
    # for cnt, ar in enumerate([pl_new, cl_new]):
    #     num = nums[cnt]
    #     fix = fixes[cnt]
    #     tmp = np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0)
    #     fix_out.append(tmp)
    #     new = []
    #     for i in fix:
    #         if i:
    #             v = [j + num for j in i]
    #             v = [i - 1 if i < 0 else i for i in v]
    #             new.append(v)
    #         else:
    #             new.append(i)
    #     out2.append(new)
    # #
    # # -- rolled output and point locations fixed
    # pl_r, cl_r = fix_out  # temporary, renamed down further
    # px_in_c_1, p_in_c_1, p_eq_c_1, p_eq_x_1 = out2[0]
    # cx_in_p_1, c_in_p_1, c_eq_p_1, c_eq_x_1 = out2[1]
    # # --
    # z0 = np.nonzero((x_pnts == cl_r[:, None]).all(-1).any(-1))[0]
    # z1 = np.nonzero((cl[c_in_p] == cl_r[:, None]).all(-1).any(-1))[0]
    # z1a = np.nonzero((cl[c_eq_x] == cl_r[:, None]).all(-1).any(-1))[0]
    # idx0 = sorted(list(set(np.concatenate((z0, z1, z1a)))))
    # # --
    # z2 = np.nonzero((x_pnts == pl_r[:, None]).all(-1).any(-1))[0]
    # z3 = np.nonzero((pl[p_in_c] == pl_r[:, None]).all(-1).any(-1))[0]
    # z3a = np.nonzero((pl[p_eq_x] == pl_r[:, None]).all(-1).any(-1))[0]
    # idx1 = sorted(list(set(np.concatenate((z2, z3, z3a)))))
    # #
    # # -- mask those that are in, out or on
    # cl_n = cl_r[idx0]  # cl, with just the intersection and `in` points
    # pl_n = pl_r[idx1]
    #
    cN = len(cl_n) - 1
    pN = len(pl_n) - 1
    # --
    # inside points for both
    # inC, inP = np.where((pl_n == cl_n[:, None]).all(-1))
    # inCinP = np.concatenate((inC[None, :], inP[None, :])).T
    # cinside = np.nonzero((pl_n != cl_n[:, None]).any(-1).all(-1))[0]
    # pinside = np.nonzero((cl_n != pl_n[:, None]).any(-1).all(-1))[0]
    #
    # -- make sure first intersection is added
    #
    # whr0 = np.nonzero(inCinP[:, 0] == cN)[0]
    # whr1 = np.nonzero(inCinP[:, 1] == pN)[0]
    # inCinP[whr0, 0] = 0
    # inCinP[whr1, 1] = 0
    # inCinP = inCinP[1:-1]  # strip off one of the duplicate start/end 0`s
    #
    # preP, preC = prePC(i0_, i1_, cN, j0_, j1_, pN, pinside, cinside)
    # prev = inCinP[0]      # -- set the first `previous` for enumerate
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
    prev = inCinP[0]
    ic = sorted(cinside)  # -- use the equivalent of p_in_c, c_in_p
    ip = sorted(pinside)
    close = False
    null_pnt = np.array([np.nan, np.nan])
    out, p_seen, c_seen = [], [], []
    out = [cl_n[0]]
    for cnt, p in enumerate(inCinP[1:], 1):  # enumerate from inCinP[1:]
        p = inCinP[cnt]
        i_c, j_c = p       # current ids, this is an intersection point
        i_p, j_p = prev    # previous ids
        d0, d1 = p - prev  # differences in ids
        sub, p_a, c_a = [], [], []
        # --
        # if i_p == 0:
        #     out.append(cl_n[i_p])  # already added, unless i_c is first, 0
        # --
        if d0 == 0:
            if j_c not in p_seen:
                out.append(cl_n[i_p])
                c_seen.append(i_p)
        elif d0 == 1:
            if d1 == 1:
                if j_c not in p_seen:  # same point
                    sub.append(cl_n[i_c])  # in original
                    c_seen.append(i_c)
            elif d1 < 0:  # negative so can't use `_bits_`
                if j_c not in p_seen:
                    sub.append(pl_n[j_c])
                    p_seen.append(j_c)
            elif d1 > 1:  # this may close a sub-polygon
                if i_c not in c_seen:
                    sub.append(cl_n[i_c])
                    c_seen.append(i_c)
                if j_c not in p_seen and cnt > 1:   # cludge
                    p_a = _bits_(j_p, j_c, pinside, p_seen)  # !!!!!
                    if d1 > 2 and cnt > 2 and len(p_a) == 0:  # check
                        sub.append(null_pnt)  # this works with E, d0_
                    if p_a:  # needed for edgy1, eclip
                        sub.extend(pl_n[p_a])
                        p_seen.extend(p_a)
                elif cnt == 1:
                    sub.append(cl_n[i_c])  # append cl_n[i_c] or pl_n[j_c]
                    c_seen.append(i_c)
                # if j_c + 1 in p_seen:  # if the next point was seen, close out
                #     sub.extend([pl_n[j_c + 1], null_pnt])
                #     p_seen.append(j_c)  # add the index before decreasing
                #     j_c -= 1
                #     close = True
                # elif j_c + 1 in ip:
                #     # same as
                #     # _bits_(j_c + 1, pN, ip, p_seen)
                #     p_a = []  # poly points to add
                #     st = j_c
                #     for i in ip:
                #         if i - st == 1:
                #             p_a.append(i)
                #             st = i
                #         else:
                #             break
                #     if p_a:
                #         sub.extend(pl_n[p_a])
                #         nxt = p_a[-1] + 1
                #         if nxt in inCinP[:, 1]:
                #             sub.append(pl_n[nxt])
                #             p_a.append(nxt)
                #         close = True
                # if i_c + 1 in ic:
                #     sub.append(cl_n[i_c + 1])
                #     c_a.append(i_c + 1)
                # # --
                # if close:
                #     sub.append(null_pnt)
                #     close = not close
        # --
        elif d0 > 1:
            if d1 == 1:
                c_a = _bits_(i_p, i_c, ic, c_seen)
                # sub.append(cl_n[i_c])  # only if d0 previous == 1 ????
                sub.extend(cl_n[c_a])
                sub.append(cl_n[i_c])  # edgy1, eclip
                c_seen.extend(c_a)
                c_seen.append(i_c)
                # sub.append(pl_n[j_c])
            elif d1 < 0:
                c_a = _bits_(i_p, i_c, ic, c_seen)
                sub.extend(cl_n[c_a])
                sub.append(cl_n[i_c])  # in clip_poly2
                c_seen.extend(c_a)
                c_seen.append(i_c)
            elif d1 > 1:
                c_a = _bits_(i_p, i_c, ic, c_seen)
                if c_a:
                    sub.extend(cl_n[c_a])
                    c_seen.extend(c_a)
                sub.append(cl_n[i_c])
                c_seen.append(i_c)
        # --
        elif d0 < 0:  # needs fixing with d1==1, d1<0, d1>0
            if i_c == 0 and i_p == cN - 1:  # second last connecting to first
                p_a = _bits_(j_p, pN, pinside, p_seen)
                sub.extend(pl_n[p_a])
                p_seen.extend(p_a)
                # sub.append(pl_n[pN])  # commented out 2023-03-19 for E, d0_
            elif i_c == 0 and cnt == len(inCinP) - 2:  # last slice
                c_a = _bits_(i_p, cN, cinside, c_seen)
                sub.extend(cl_n[c_a])
                sub.append(cl_n[i_c])
                c_seen.extend(c_a)
                c_seen.append(i_c)
        else:
            if i_c not in c_seen:
                sub.append(cl_n[i_c])
                c_seen.append(i_c)
            # if i_c not in c_seen:  # not in clip_poly2
            #     sub.append(cl_n[i_c])
        #
        # c_seen.extend([i_c, i_p])
        # p_seen.extend([j_c, j_p])
        # c_seen.extend(c_a)
        # p_seen.extend(p_a)
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
    whr = np.nonzero(np.isnan(out[:, 0]))[0]  # split at null_pnts
    if len(whr) > 0:
        ft = np.asarray([whr, whr + 1]).T.ravel()
        subs = np.array_split(out, ft)
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


# ---- (5) split polygon
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
    result = _prepare_([poly, line], roll=False, polygons=[True, False])
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
def find_segment_overlaps(arr, is_poly=True, return_all=True):
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


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")
