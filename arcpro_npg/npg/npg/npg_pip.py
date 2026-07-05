# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
-------
npg_pip
-------

Point in Polygon implementation using winding numbers.  This is for Geo arrays
and uses numpy enhancements.

----

Script :
    npg_pip.py
Author :
    `<https://github.com/Dan-Patterson>`_.

Modified :
    2026-06-19

Purpose
-------
Functions for point partitioning and winding number inclusion tests for points
in polygons.

Notes
-----
**np_wn notes**

The polygon is represented as from-to pairs (fr_, to_).  Their x, y values
are obtained by translation and splitting (x0, y0, x1, y1).
The input points are processing in a similar fashion (pnts --> px, py).
The `winding number` is determined for all points at once for the given
polygon.

**pnts_in_Geo notes**

Pre-processing to remove duplicates or partition the points hasn't proved
to be optimal in all situations.  They are included for experimental
purposes.  In such cases, the process is as follows::

- Determine polygon extents for the Geo array `geo`.
- Derive the unique points for the test points `pnts`.
- Assign points to the appropriate extent.
- Run `winding number` algorithm (or `crossing number` if so inclined)
- Deleting points as you go does not improve things.

How to remove points from an array, if found in an array.  In the example below
`sub` is a subarray of `pnts`. The indices where they are equal is `w`.

>>> w = np.where((pnts == sub[:, None]).all(-1))[1]
>>> pnts = np.delete(pnts, w, 0)

References
----------
`<https://en.wikipedia.org/wiki/Point_in_polygon>`_.  ** general information

`<https://web.archive.org/web/20131210180851/http://geomalgorithms.com/a03-
_inclusion.html>`_.  ** original site usurped

`<https://stackoverflow.com/questions/33051244/numpy-filter-points-within-
bounding-box/33051576#33051576>`_.

`<https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html>`_.  ** good

"""
# pycodestyle D205 gets rid of that one blank line thing
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0402,E0611,E1136,E1121,R0904,R0914,
# pylint: disable=W0201,W0212,W0221,W0612,W0621,W0105
# pylint: disable=R0902


import sys
import numpy as np

# ---- optional imports
# import npgeom as npg
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

# noqa: E501
fmt_ = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.3f}'.format}
np.set_printoptions(precision=3, threshold=100, edgeitems=10, linewidth=80,
                    suppress=True,
                    formatter=fmt_,
                    floatmode='maxprec_equal',
                    legacy='1.25')  # legacy=False or legacy='1.25'
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    'crossing_num',
    'winding_num',
    'partition',
    'np_wn',
    'pip',
    'pnts_in_Geo',
    'pnts_on_segments'
]

__helpers__ = [
    '_side_',
    '_is_right_side',
]


# ---- ---------------------------
# ---- (1) single use helpers
#
def _side_(pnts, poly, on_is_in=True):  # ** not used
    r"""Return points inside, outside or equal/crossing a convex poly feature.

    Parameters
    ----------
    pnts, poly : array-like
        Nx2 array of points.  `poly` is a polygon, `pnts` need not form a
        perimeter.

    Experimenting with the various options.  Not used for any specific
    functions.

    Returns
    -------
    r       the equation value array
    in_     the points based on the winding number
    inside  (r < 0)
    outside (r > 0)
    equal_  (r == 0)

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
    x, y = pnts.T
    x0, y0 = poly[:-1].T  # poly segment start points
    x1, y1 = poly[1:].T   # poly segment end points
    x_x0 = x[:, None] - x0
    y_y0 = y[:, None] - y0
    # y_y1 = y[:, None] - y1
    r = ((x1 - x0) * y_y0 - (y1 - y0) * x_x0) + 0.0
    # r = (x3 - x0) * (y[:, None] - y0) - (y3 - y2) * (x[:, None] - x2)
    msk = np.isclose(r, 0., rtol=1e-05, atol=1e-08)  # mask for close to zero
    r[msk == 1] = 0.
    # -- from _wn_, winding numbers for concave/convex poly
    #
    chk3 = np.sign(r).astype(int)
    #
    if on_is_in:
        chk1 = (y_y0 >= 0.0)  # -- >= 0, original
        chk2 =  np.less(y[:, None], y1)  # (y_y1 < 0.0)
        pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
        neg = (~chk1 & ~chk2 & (chk3 <= 0)).sum(axis=1, dtype=int)  # chk3 < 0
    else:  # -- gives the same as above -- testing
        chk1 = (y_y0 >= 0.0)  # -- top and bottom point inclusion!   try `>`
        chk2 = np.less(y[:, None], y1)
        pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
        neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int) 
    #
    wn_vals = pos - neg
    wn_ids = np.nonzero(wn_vals)[0]
    in_ = pnts[wn_ids]  # -- in and on
    # 
    inside = pnts[(r < 0).all(axis=-1)]  # all must be True along row, convex
    outside = pnts[(r > 0).any(-1)]      # any must be True along row
    equal_ = pnts[(r == 0).any(-1)]      # ditto
    # !!! equal needs a check, it is actually just collinear and needs to be
    #     compared to segment length
    return r, wn_vals, wn_ids, in_, inside, outside, equal_


def _is_right_side(p, strt, end):
    """Determine if point (p) is `inside` a line segment (strt-->end).

    See Also
    --------
    line_crosses, in_out_crosses in npg_geom_hlp.
    position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))

    Returns
    -------
    Negative for right of clockwise line, positive for left. So in essence,
    the reverse of _is_left_side with the outcomes reversed ;)
    """
    x, y, x0, y0, x1, y1 = *p, *strt, *end
    return (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)


def crossing_num(pnts, poly, line=True):
    """Crossing Number for point(s) in polygon.  See `pnts_in_poly`.

    Parameters
    ----------
    pnts : array of points
        Points are an N-2 array of point objects determined to be within the
        extent of the input polygons.
    poly : polygon array
        Polygon is an Nx2 array of point objects that form the clockwise
        boundary of the polygon.
    line : boolean
        True to include points that fall on a line as being inside.
    """
    def _in_ex_(pnts, ext):
        """Return the points within an extent or on the line of the extent."""
        LB, RT = ext
        comp = np.logical_and(LB <= pnts, pnts <= RT)  # using <= and <=
        idx = np.logical_and(comp[..., 0], comp[..., 1])
        return idx, pnts[idx]

    pnts = np.atleast_2d(pnts)
    xs = poly[:, 0]
    ys = poly[:, 1]
    N = len(poly)
    xy_diff = np.diff(poly, axis=0)
    dx = xy_diff[:, 0]  # np.diff(xs)
    dy = xy_diff[:, 1]  # np.diff(ys)
    ext = np.array([poly.min(axis=0), poly.max(axis=0)])
    idx, inside = _in_ex_(pnts, ext)
    is_in = []
    for pnt in inside:
        cn = 0   # the crossing number counter
        x, y = pnt
        for i in range(N - 1):
            if line is True:
                c0 = (ys[i] < y <= ys[i + 1])  # changed to < <=
                c1 = (ys[i] > y >= ys[i + 1])  # and > >=
            else:
                c0 = (ys[i] < y < ys[i + 1])
                c1 = (ys[i] > y > ys[i + 1])
            if (c0 or c1):  # or y in (ys[i], ys[i+1]):
                vt = (y - ys[i]) / dy[i]  # compute x-coordinate
                if line is True:
                    if (x == xs[i]) or (x < (xs[i] + vt * dx[i])):  # include
                        cn += 1
                else:
                    if x < (xs[i] + vt * dx[i]):  # exclude pnts on line
                        cn += 1
        is_in.append(cn % 2)  # either even or odd (0, 1)
    return inside[np.nonzero(is_in)]


def winding_num(pnts, poly, batch=True):
    """Point in polygon using winding numbers.

    Parameters
    ----------
    pnts : array
        This is simply an (x, y) point pair of the point in question.
    poly : array
        A clockwise oriented Nx2 array of points, with the first and last
        points being equal.

    Notes
    -----
    Until this can be implemented in a full array of points and full suite of
    polygons, you have to test for all the points in each polygon.

    >>> w = [winding_num(p, e1) for p in g_uni]
    >>> g_uni[np.nonzero(w)]
    array([[ 20.00,  1.00],
    ...    [ 21.00,  0.00]])

    References
    ----------
    `<https://web.archive.org/web/20131210180851/http://geomalgorithms.com/
    a03-_inclusion.html> `_.
    """
    def _is_right_side(p, strt, end):
        """Determine if a point (p) is `inside` a line segment (strt-->end).

        See Also
        --------
        `line_crosses`, `in_out_crosses` in npg_geom_hlp.
        position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
        negative for right of clockwise line, positive for left. So in essence,
        the reverse of _is_left_side with the outcomes reversed ;)
        """
        x, y, x0, y0, x1, y1 = *p, *strt, *end
        val = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        # print("pnt {} : val {}".format(p, val))
        return val

    def cal_w(p, poly):
        """Do the calculation."""
        w = 0
        y = p[1]
        ys = poly[:, 1]
        for i in range(poly.shape[0]):  # -- see notes
            if ys[i - 1] <= y:
                if ys[i] > y:
                    if _is_right_side(p, poly[i - 1], poly[i]) > 0:
                        w += 1
            elif ys[i] <= y:
                if _is_right_side(p, poly[i - 1], poly[i]) < 0:
                    w -= 1
        return w

    if batch:
        w = [cal_w(p, poly) for p in pnts]
        return pnts[np.nonzero(w)], w
    else:
        return cal_w(pnts, poly)


# ---- ---------------------------
# ---- (2) ... points in polygons
#
def partition(pnts, geo, as_structured=False):
    """Partition points into the first polygon they fall into.

    Parameters
    ----------
    pnts, geo : ndarrays
        `pnts` is an Nx2 array representing point objects (x, y).
        `geo` is a Geo array.
    return_remainder : boolean
        True, returns the inside and outside points

    Notes
    -----
    This code block can be added to pnts_in_Geo if you want to test partition::

    if partition:
        ps_in_exts = partition(pnts, geo)
        polys = geo.outer_rings(False)
        for i, pts in enumerate(ps_in_exts):
            if pts.size > 0:
                in_, w = np_wn(pts, polys[i])
                w_s.append(w)  # [w, pts])
                out.append(in_)  # [geo.shp_IFT[i]

    Extras
    ------
    plotting::
        data = [[geo.bits, 2, 'red', '.', True ],
                [pnts, 0, 'black', 'o', False]]
        plot_mixed(data, "Partition")

    """
    def _as_structured_(ids_, pnts):
        """Return output as structured array"""
        n_ = 0
        sub0 = []
        sub1 = []
        sub2 = []
        for i in ids_:
            n0 = len(i[1])
            sub0.append(np.repeat(i[0], n0))
            sub1.append(i[1])
            sub2.append(pnts[i[1]])
            n_ += n0
        f0 = np.concatenate(sub0)
        f1 = np.concatenate(sub1)
        f2 = np.concatenate(sub2, axis=0)
        dt = [('Poly_ID', '<i4'),  ('Pnt_ids', '<i4'),
              ('Xs', '<f8'), ('Ys', '<f8')]
        tmp = np.empty((n_,), dtype=dt)
        tmp['Poly_ID'] = f0
        tmp['Pnt_ids'] = f1
        tmp['Xs'] = f2[:, 0]
        tmp['Ys'] = f2[:, 1]        
        return tmp

    extents = geo.extents(splitter="shape")
    L_ = extents[:, 1]
    B_ = extents[:, 0]
    srt_idx = np.lexsort((B_, L_)).tolist()
    extents = extents[srt_idx]
    in_ = []
    ids_ = []
    for cnt, e in enumerate(extents):  # extents are in LB to RT order
        c0 = np.logical_and(e[0] <= pnts[:, 0], pnts[:, 0] <= e[2])
        c1 = np.logical_and(e[1] <= pnts[:, 1], pnts[:, 1] <= e[3])
        c2 = np.logical_and(c0, c1).nonzero()[0]
        N_in = pnts[c2]
        if len(N_in) > 0:
            in_.append(N_in)
            ids_.append([cnt, c2])
        # out_pnts = pnts[np.logical_not(c2)]
    if len(in_) == 0:  # -- bail, none in extent(s)
        return None
    #
    if as_structured:
        if len(in_) == 1:
            ids_ = [ids_[0]]
        in_pnts = _as_structured_(ids_, pnts)
    else:
        in_pnts = np.concatenate(in_, axis=0)
    return in_pnts


def np_wn(pnts, poly, on_is_in=False, return_winding=False):
    """Return points in polygon using a winding number algorithm in numpy.

    Parameters
    ----------
    pnts : Nx2 array
        Points represented as an x,y array.
    poly : Nx2 array
        Polygon consisting of at least 4 points oriented in a clockwise manner.
    on_is_in : boolean
        True makes points on the perimeter included with `in` points.  False
        includes them with `out` points (usually).
    return_winding : boolean
        True, returns the winding number pattern for testing purposes.  Keep as
        False to avoid downstream errors.

    Returns
    -------
    The points within or within and on the boundary of the polygon.

    Notes
    -----
    The polygon is represented as from-to pairs (`fr_`, `to_`).  Their x,y
    values are obtained by translation and splitting (x0, y0, x1, y1).
    The input points are processed in a similar fashion (pnts --> x, y).
    The `winding number` is determined for all points at once for the given
    polygon.

    Original form

    >>> c0 = (x1 - x0) * (y[:, None] - y0)
    >>> c1 = (y1 - y0) * (x[:, None] - x0)
    >>> diff_ = c0 - c1

    Useage
    ------
    >>> out_ = [np_wn(points, poly) for poly in polygons]
    >>> final = np.unique(np.vstack(out_), axis=0)  # points only

    Inclusion checks
    ----------------
    on the perimeter is deemed `out`
        chk1 (y_y0 > 0.0)  changed from >=
        chk2 np.less is ok
        chk3 leave
        pos  leave
        neg  chk3 <= 0  to keep all points inside poly on edge included

    from
    `<https://lazyjobseeker.github.io/en/posts/winding-number-algorithm/>`_.

    if p_x > np.max(x_coords): continue  # ... (2)
            if (y_coords[0]-p_y)*(y_coords[1]-p_y)<0: # ... (3)
                total = total + 1
            if y_coords[0] == p_y: total = total + 0.5 # ... (4)
            if y_coords[1] == p_y: total = total + 0.5 # ... (4)

    References
    ----------
    `<https://community.esri.com/t5/python-blog/point-in-polygon-geometry-
    mysteries/ba-p/893890>`_.  my blog
    `<https://github.com/congma/polygon-inclusion/blob/master/
    polygon_inclusion.py>`_.  inspiration for this numpy version
    """
    def extent_chk(pnts, poly):
        """Prune out extraneous points"""
        e = np.concatenate((np.min(poly, axis=0), np.max(poly, axis=0)))
        c0 = np.logical_and(e[0] <= pnts[:, 0], pnts[:, 0] <= e[2])
        c1 = np.logical_and(e[1] <= pnts[:, 1], pnts[:, 1] <= e[3])
        in_extent = np.logical_and(c0, c1).nonzero()[0]
        return in_extent

    def on_edge_chk(diff_, dot_, seg_len):
        """Edge check from `_wn_`"""
        is_collinear = np.abs(diff_) < 1e-10
        is_within_segment = (dot_ >= 0) & (dot_ <= seg_len)
        on_edge = np.any(is_collinear & is_within_segment, axis=1)
        # is_collinear_ids = np.nonzero(is_collinear.any(-1))[0]
        on_edge_ids = np.nonzero(on_edge)[0]
        on_vertex_ids = np.nonzero((poly[:-1] == pnts[:, None]).all(-1))[0]
        on_boundary = np.unique(
            np.concatenate((on_vertex_ids, on_edge_ids)), sorted=True)
        return on_boundary

    # -- single point check
    if pnts.ndim == 1:
        pnts = pnts[None, :]  # 2025-11-09 to check for a single point
    #
    # -- pnts in extent check
    # p = np.atleast_2d(pnts)
    # in_extent = extent_chk(p, poly)
    # pnts = pnts[in_extent]
    #
    x, y = pnts.T         # point coordinates
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    x1_x0, y1_y0 = (poly[1:] - poly[:-1]).T  # get dx, dy 
    x_x0 = x[:, None] - x0
    y_y0 = y[:, None] - y0
    y_y1 = y[:, None] - y1
    #
    # -- cross product, is right determination, original below
    #    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0
    diff_ = x1_x0 * y_y0 - y1_y0 * x_x0
    #
    # -- for boundary point check
    dot_ = x1_x0 * x_x0 + y1_y0 * y_y0  # -- dot product
    seg_len = x1_x0**2 + y1_y0**2  # segment length squared
    #
    chk1 = (y_y0 >= 0.0)  # -- top and bottom point inclusion!   try `>`
    chk2 = (y_y1 < 0.0)  # was  chk2 = np.less(y[:, None], y1)  try `<`
    chk3 = np.sign(diff_).astype(np.int32)  # chk3 == 0 collinear pnt check
    #
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn = pos - neg
    #
    # -- edge check to look for boundary points
    on_boundary = on_edge_chk(diff_, dot_, seg_len)
    #
    if on_is_in:  # -- include boundary points
        z0 = np.sum(diff_ > 0., axis=1)
        in_and_on = np.nonzero(z0 == 0)[0]
        in_ = pnts[in_and_on]
    else:  # -- use set difference
        whr_in = np.nonzero(wn)[0]  # -- initial estimate
        in_fully = np.array(sorted(list(set(whr_in).difference(on_boundary))))
        in_ = pnts[in_fully]
    if return_winding:  # correct wn for boundary points
        if on_is_in and len(on_boundary) > 0:
            wn[on_boundary] = -1
        if not on_is_in and len(on_boundary) > 0:
            wn[on_boundary] = 0
        return in_, wn
    return in_


def pip(pnts, poly, on_is_in=False, extras=False):
    """Return points in polygon with boundary check.

    Parameters
    ----------
    pnts : Nx2 array
        Points represented as an x,y array.
    poly : Nx2 array
        Polygon consisting of at least 4 points oriented in a clockwise manner.
    on_is_in : boolean
        True makes points on the perimeter included with `in` points.  False
        includes them with `out` points (usually).
    extras : boolean
        True, returns options information on point location.

    Returns
    -------
    The ids of the points that are `in_fully`. If extras is True, then a list,
    containing values for [`on_boundary`, `out_fully`, `in_and_on`,
    `out_and_on`] are also provided.

    Notes
    -----
    See `np_wn` if a winding number value is required.
    This variant returns a different set of values for the `extras` parameter.
    If True, the following are returned as a list.

    - point ids that are on the boundary of the polygon perimeter
      (`on_boundary`)
    - point ids fully outside and on the perimeter (`out_and_on`)

    References
    ----------
    `<https://community.esri.com/t5/python-documents/point-in-analysis/ta-p/
    916504>`_.  my blog
    """

    def extent_chk(pnts, poly):
        """Prune out extraneous points. See reference."""
        e = np.concatenate((np.min(poly, axis=0), np.max(poly, axis=0)))
        c0 = np.logical_and(e[0] <= pnts[:, 0], pnts[:, 0] <= e[2])
        c1 = np.logical_and(e[1] <= pnts[:, 1], pnts[:, 1] <= e[3])
        in_extent = np.logical_and(c0, c1).nonzero()[0]
        return in_extent

    if pnts.ndim == 1:
        pnts = pnts[None, :]  # 2025-11-09 to check for a single point
    #
    # -- pnts and poly info
    x, y = pnts.T         # point coordinates
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    #
    # -- calculations
    x_x0 = x[:, None] - x0  # pnt x minus segment start x
    y_y0 = y[:, None] - y0  # pnt y relative to segment start y
    x1_x0, y1_y0 = (poly[1:] - poly[:-1]).T  # get dx, dy 
    #
    seg_len = x1_x0**2 + y1_y0**2  # segment length squared
    cross_ = x1_x0 * y_y0 - y1_y0 * x_x0  # -- cross product, `diff_`
    dot_ = x1_x0 * x_x0 + y1_y0 * y_y0  # -- dot product
    #
    is_collinear = np.abs(cross_) < 1e-10
    is_within_segment = (dot_ >= 0) & (dot_ <= seg_len)
    on_edge = np.any(is_collinear & is_within_segment, axis=1)
    # is_collinear_ids = np.nonzero(is_collinear.any(-1))[0]
    on_edge_ids = np.nonzero(on_edge)[0]
    on_vertex_ids = np.nonzero((poly[:-1] == pnts[:, None]).all(-1))[0]
    on_boundary = np.unique(
        np.concatenate((on_vertex_ids, on_edge_ids)), sorted=True)
    #
    z0 = np.sum(cross_ > 0., axis=1)  # sum positive values from above
    in_and_on = np.nonzero(z0 == 0)[0]  # yields ids of points on and in 
    in_fully = np.array(sorted(list(set(in_and_on).difference(on_boundary))))
    out_fully = np.nonzero(z0 > 0)[0]  # point full outside, boundary
    out_and_on = np.unique(
        np.concatenate((on_boundary, out_fully)), sorted=True)
    #
    extra_info = [on_boundary, out_fully, in_and_on, out_and_on]
    if extras:
        return in_fully, extra_info
    return in_fully


def pnts_in_Geo(pnts, geo, stacked=True):
    """Geo array implementation of points in polygon using `winding number`.

    Parameters
    ----------
    pnts : array (N, 2)
       An ndarray of point objects.
    geo : Geo array
        The Geo array of polygons.  Only the outer rings are used.
    stacked : boolean
        True, stack the inclusion points as one set.  False, returns the points
        as separate entities.

    Returns
    -------
    Points completely inside or on the boundary of a polygon are returned.

    Requires
    --------
    The helper, `np_wn`, (winding number inclusion test).

    Notes
    -----
    See docstring notes.

    >>> # for my testing
    >>> final  = pnts_in_Geo(g_uni, g4, True)
    """
    #
    out = []
    polys = geo.outer_rings(False)
    for poly in polys:
        in_ = np_wn(pnts, poly, on_is_in=True)
        out.append(in_)
    pts = [i for i in out if len(i) > 0]
    if len(pts) > 1 and stacked:
        return np.unique(np.vstack(pts), axis=0)
    return pts


def pnts_on_segments(pnts, segs, ids_only=False):
    """Determine whether any of the points in `pnts` are on segments.

    Parameters
    ----------
    pnts : array (N, 2)
    segs : array (N, 2) or (N, 4)
        `segs` can represent a collection of line segments (N, 4) or a polygon
        perimeter (N, 2).
    ids_only : boolean
        True, returns the id values of the points that are on segments.  False,
        returns a structured array with full information

    Returns
    -------
    The `ids_only` controls the type of output.  Point ids only or full
    information regarding whether the points intersect a segment and on which
    ones.

    Notes
    -----
    The working part of the code also sorts the x, y values within the pairs to
    facilitate determining whether a points x or y value lies within the
    range of those values of the segment.

    To plot the data::

        # -- aoi = polygon, pnts = points
        data = [[segs, 1, 'red', '.', True ], [pnts, 0, 'black', 'o', False]]
        plot_mixed(data)

    """
    def _is_pnt_on_seg_(seg, pnt, tol=1e-6):
        """Mini pnt_on_seg function normally required by `npGeo.pnt_on_poly`.
        From `npg_geom_ops._closest_pnt_on_poly_`
        """
        x, y = pnt
        x0, y0, x1, y1, dx, dy = *seg[0], *seg[1], *(seg[1] - seg[0])
        dist_0 = (x - x0)**2 + (y - y0)**2
        dist_1 = (x - x1)**2 + (y - y1)**2
        dist_ = dx * dx + dy * dy  # squared length
        _a, _b, _c = np.sqrt([dist_, dist_0, dist_1])
        chk = _a - (_b + _c)
        if -tol <= chk and chk < tol:
            return True
        return False

    xs = pnts[:, 0]
    ys = pnts[:, 1]
    seg_shp = segs.shape[1]  # segment shape is either Nx2 or Nx4
    #
    if seg_shp== 2:  # -- polygon perimeter
        s_xs = segs[:, 0]
        s_ys = segs[:, 1]
        xs_pairs = np.concatenate(
            (s_xs[:-1][:, None], s_xs[1:][:, None]), axis=1)
        xs_pairs = np.sort(xs_pairs, axis=1)
        fr_x = xs_pairs[:, 0]
        to_x = xs_pairs[:, 1]
        ys_pairs = np.concatenate(
            (s_ys[:-1][:, None], s_ys[1:][:, None]), axis=1)
        ys_pairs = np.sort(ys_pairs, axis=1)
        fr_y = ys_pairs[:, 0]
        to_y = ys_pairs[:, 1]
    #
    elif seg_shp == 4:  # -- segments as from-to points
        # -- testing use fr_to = np.concatenate((aoi2[:-1], aoi2[1:]), axis=1)
        fr_x, fr_y = segs[:, :2].T
        to_x, to_y = segs[:, -2:].T
        xs_pairs = np.concatenate((fr_x[:, None], to_x[:, None]), axis=1)
        xs_pairs = np.sort(xs_pairs, axis=1)
        fr_x = xs_pairs[:, 0]
        to_x = xs_pairs[:, 1]
        ys_pairs = np.concatenate((fr_y[:, None], to_y[:, None]), axis=1)
        ys_pairs = np.sort(ys_pairs, axis=1)
        fr_y = ys_pairs[:, 0]
        to_y = ys_pairs[:, 1]
    #    
    x_idx = np.logical_and(xs >= fr_x[:, None], xs <= to_x[:, None])
    y_idx = np.logical_and(ys >= fr_y[:, None], ys <= to_y[:, None])
    whr_ids = np.nonzero(x_idx & y_idx)  # rows are segments
    tmp_ = np.concatenate((whr_ids[0][:, None], whr_ids[1][:, None]), axis=1)
    splt = np.nonzero(np.diff(tmp_, axis=0)[:, 0] >= 1)[0] + 1
    subs = np.array_split(tmp_, splt, axis=0)
    #
    # -- np.nonzero((a == b[:, None]).all(-1))  # -- compare for equality
    # -- those that do not meet the spatial criteria
    # not_ids = np.array(set(np.arange(pnts.shape[0])).difference(whr_ids[1]))
    #
    # -- process that meet the spatial criteria to see if they are on a segment
    out = []
    for cnt, s in enumerate(subs):
        s = subs[cnt]
        id0 = s[0][0]  # -- segment id, values in `sub` col 0 will be the same
        id1 = s[:, 1]  # -- pnt ids, values will differ and can occur elsewhere
        if seg_shp == 2:
            seg_xy = segs[[id0, id0 + 1]]
        else:
            seg_xy = segs[id0].reshape(2, 2)
        pnt_xys = pnts[s[:, 1]]  # -- Note, may need to change pnt_xys to pnts
        result = [(id0, id1[c], pnt[0], pnt[1], _is_pnt_on_seg_(seg_xy, pnt))
                  for c, pnt in enumerate(pnt_xys)]
        out.extend(result)
    #
    seg_dt = np.dtype([('seg_id', 'i4'), ('pnt_id', 'i4'),
                       ('X', 'f8'), ('Y', 'f8'), ('Is_on', 'bool')])
    final = np.array(out, dtype=seg_dt)
    whr_on = np.nonzero(final['Is_on'] == True)[0]  # -- only the `on` points
    if ids_only:
        vals = np.sort(final['pnt_id'][whr_on])
        return vals
    return final[whr_on]


def pip_test(pnts=None, poly=None):
    """Point in polygon test."""
    from npg.npg_plots import plot_mixed
    if poly is None:
        # -- duplicate of aoi0, plus triangles
        poly = np.array([[1., 1.], [1., 9.], [9., 9.], [9., 1.], [1., 1.]])
        # poly = np.array([[1., 1.], [1., 9.], [9., 1.], [1., 1.]])  # left tri
        # poly = np.array([[1., 1.], [9., 9.], [9., 1.], [1., 1.]])  # rgt tri
    if pnts is None:
        pnts = np.array([[0.0, 0.0], [0.0, 10.0],
                       [10.0, 10.0], [10.0, 0.0],
                       [0.5, 1.0], [1.0, 1.0],
                       [1.0, 0.5], [1.5, 1.5],
                       [0.5, 9.0], [1.0, 9.0],
                       [1.0, 9.5], [1.5, 8.5],
                       [9.5, 9.0], [9.0, 9.0],
                       [9.0, 9.5], [8.5, 8.5],
                       [9.5, 1.0], [9.0, 1.0],
                       [9.0, 0.5], [8.5, 1.5],
                       [1.001, 4.0], [1.0, 5.0], [0.999, 6.0],
                       [4., 8.999], [5.0, 9.0], [6.0, 9.001],
                       [8.999, 6.0], [9.0, 5.0], [9.001, 4.0],
                       [6.0, 1.001], [5.0, 1.0], [4.0, 0.999],
                       ])
    data = [[poly, 2, 'red', '.', True ], [pnts, 0, 'black', 'o', False]]
    plot_mixed(data, title="Points in Polygons",  invert_y=False, ax_lbls=None)
    #
    in_, wn = np_wn(pnts, poly, on_is_in=False, return_winding=True)
    return in_, wn

#
# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
