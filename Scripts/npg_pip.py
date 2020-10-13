# -*- coding: utf-8 -*-
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
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2020-02-13

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
- run `crossing number/ray crossing` algorithm
- deleting points as you go does not improve things.

How to remove points from an array, if found in an array.  In the example below
`sub` is a subarray of `pnts`. The indices where they are equal is `w`.

>>> w = np.where((pnts == sub[:, None]).all(-1))[1]
>>> pnts = np.delete(pnts, w, 0)

References
----------
`<http://geomalgorithms.com/a03-_inclusion.html>`_.

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
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    '_is_right_side', 'crossing_num', 'winding_num', '_partition_', 'np_wn',
    'pnts_in_Geo'
]


# ---- single use helpers
#
def _is_right_side(p, strt, end):
    """Determine if point (p) is `inside` a line segment (strt-->end).
    See : line_crosses, in_out_crosses in npg_helpers.
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
                c0 = (ys[i] < y <= ys[i + 1])  # changed to <= <=
                c1 = (ys[i] > y >= ys[i + 1])  # and >= >=
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


def winding_num(pnts, poly):
    """Point in polygon using winding numbers.

    Parameters
    ----------
    pnts : array
        The (x, y) point pairs to be tested for inclusion.
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
    """
    def cal_w(p, poly):
        """Do the calculation"""
        w = 0
        y = p[1]
        ys = poly[:, 1]
        for i in range(poly.shape[0]):
            if ys[i-1] <= y:
                if ys[i] > y:
                    if _is_right_side(p, poly[i-1], poly[i]) > 0:
                        w += 1
            elif ys[i] <= y:
                if _is_right_side(p, poly[i-1], poly[i]) < 0:
                    w -= 1
        return w
    w = [cal_w(p, poly) for p in pnts]
    return pnts[np.nonzero(w)]


# ----------------------------------------------------------------------------
# ---- (1) ... points in polygons
#
def _partition_(pnts, geo, return_remainder=False):
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
        ps_in_exts = _partition_(pnts, geo)
        polys = geo.outer_rings(False)
        for i, pts in enumerate(ps_in_exts):
            if pts.size > 0:
                in_, w = np_wn(pts, polys[i])
                w_s.append(w)  # [w, pts])
                out.append(in_)  # [geo.shp_IFT[i]

    """
    extents = geo.extents(splitter="shape")
    L_ = extents[:, 1]
    B_ = extents[:, 0]
    srt_idx = np.lexsort((B_, L_)).tolist()
    extents = extents[srt_idx]
    in_ = []
    for e in extents:
        c0 = np.logical_and(e[0] <= pnts[:, 0], pnts[:, 0] <= e[2])
        c1 = np.logical_and(e[1] <= pnts[:, 1], pnts[:, 1] <= e[3])
        c2 = np.logical_and(c0, c1)
        in_.append(pnts[c2])
        out_pnts = pnts[np.logical_not(c2)]
    in_pnts = np.asarray(in_)[sorted(srt_idx)]
    if return_remainder:
        return in_pnts, out_pnts
    return in_pnts


def np_wn(pnts, poly, return_winding=False):
    """Return points in polygon using a winding number algorithm in numpy.

    Parameters
    ----------
    pnts : Nx2 array
        Points represented as an x,y array.
    poly : Nx2 array
        Polygon consisting of at least 4 points oriented in a clockwise manner.
    return_winding : boolean
        True, returns the winding number pattern for testing purposes.  Keep as
        False to avoid downstream errors.

    Returns
    -------
    The points within or on the boundary of the geometry.

    Notes
    -----
    The polygon is represented as from-to pairs (`fr_`, `to_`).  Their x, y
    values are obtained by translation and splitting (x0, y0, x1, y1).
    The input points are processing in a similar fashion (pnts --> x, y).
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

    References
    ----------
    `<https://github.com/congma/polygon-inclusion/blob/master/
    polygon_inclusion.py>`_.  inspiration for this numpy version
    """
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    x, y = pnts.T         # point coordinates
    y_y0 = y[:, None] - y0
    x_x0 = x[:, None] - x0
    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
    chk1 = (y_y0 >= 0.0)
    chk2 = np.less(y[:, None], y1)  # pnts[:, 1][:, None], poly[1:, 1])
    chk3 = np.sign(diff_).astype(np.int)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn = pos - neg
    out_ = pnts[np.nonzero(wn)]
    if return_winding:
        return out_, wn
    return out_


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
        in_ = np_wn(pnts, poly, return_winding=False)  # run np_wn
        out.append(in_)
    pts = [i for i in out if len(i) > 0]
    if len(pts) > 1 and stacked:
        return np.unique(np.vstack(pts), axis=0)
    return pts


#
# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
