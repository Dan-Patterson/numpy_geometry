# -*- coding: utf-8 -*-
r"""
--------------------------------------
  npg_geom: Geometry focused methods
--------------------------------------

Geometry focused methods that work with Geo arrays or np.ndarrays.

----

Script :
    npg_geom.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-01-22

Purpose
-------
Geometry focused methods that work with Geo arrays or np.ndarrays.
In the case of the former, the methods may be being called from Geo methods
in such things as a list comprehension.

Notes
-----
(1) ``_npgeom_notes_.py`` contains other notes of interest.

(2) See references for the origin of this quote.

"For an Esri polygon to be simple, all intersections have to occur at
vertices. In general, it follows from 'a valid Esri polygon must have
such structure that the interior of the polygon can always be unambiguously
determined to be to the right of every segment', that"::

    - segments can touch other segments only at the end points,
    - segments have non-zero length,
    - outer rings are clockwise and holes are counterclockwise,
      each polygon ring has non-zero area.
    - order of the rings does not matter,
    - rings can be self-tangent,
    - rings can not overlap.

(3) How to flatten a searchcursor to points and/or None::

    in_fc = "C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
    SR = npg.getSR(in_fc)
    with arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as c:
        pnts = [[[[p for p in arr] for arr in r[1]]] for r in c]
    c.reset()  # don't forget to reset the cursor

Example
-------
Sample data::

    f_name = "C:/Git_Dan/npgeom/data/g_arr.npz"
    g, arrs, names = npg.load_geo(f_name, suppress_extras=False)
    arr_names = arrs.files  # returns the list of array names inside

g - the geo array
arrs - the sub arrays


References
----------
See comment by Serge Tolstov in:

`List of geometry topics
<https://en.wikipedia.org/wiki/List_of_geometry_topics>`_.

`Geometry checks
<https://community.esri.com/thread/244587-check-geometry-fails-in-shared
-origin-edge-case>`_.

**Clipping,intersection references**

`<https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping>`_.

`<http://geomalgorithms.com/a09-_intersect-3.html>`_.

`<https://codereview.stackexchange.com/questions/166702/cythonized-
sutherland-hogman-algorithm>`_.

`<https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm>`_.

`<https://en.wikipedia.org/wiki/Weiler%E2%80%93Atherton_clipping_
algorithm>`_.  polygon-polygon clipping

`<https://scicomp.stackexchange.com/questions/8895/vertical-and-horizontal
-segments-intersection-line-sweep>`_.
"""

# pylint: disable=C0103, C0302, C0326, C0415, E0611, E1136, E1121
# pylint: disable=R0904, R0914
# pylint: disable=W0201, W0212, W0221, W0612, W0621, W0105
# pylint: disable=R0902

import sys
import numpy as np

# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import repack_fields

from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay


import npgeom as npg
from npGeo import *  # is_Geo, _area_bit_
# from npGeo import Geo, arrays_to_Geo
from npgeom.npg_helpers import (
    compare_geom, crossing_num, radial_sort, line_crosses
)
# import npg_io
# from npGeo_io import fc_data

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=3, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    'extent_to_poly', 'eucl_dist',
    'scale_by_area', 'offset_buffer',
    '_area_centroid_', '_angles_',
    '_ch_scipy_', '_ch_simple_', '_ch_',
    '_dist_along_', '_percent_along_', '_pnts_on_line_',
    '_polys_to_unique_pnts_', '_simplify_lines_',
    'pnts_in_Geo_extents', 'pnts_in_Geo',
    '_pnt_on_poly_', '_pnt_on_segment_', 'p_o_p',
    '_rotate_', '_tri_pnts_',
    'segments_to_polys', 'in_hole_check',
    'pnts_in_pnts'
]


def extent_to_poly(extent, kind=2):
    """Create a polygon/polyline feature from an array of x,y values.

    The array returned is ordered clockwise with the first and last point
    repeated to form a closed-loop.

    Parameters
    ----------
    extent : array-like
        The extent is specified as four float values in the form of
        L(eft), B(ottom), R(ight), T(op) eg. np.array([5, 5, 10, 10])
    kind : integer
        A value of 1 for a polyline, or 2 for a polygon.
    """
    shp = extent.shape
    if shp not in [(2, 2), (4,)]:
        print("Check the docs...\n{}".format(extent_to_poly.__doc__))
        return None
    L, B, R, T = extent
    L, R = min(L, R), max(L, R)
    B, T = min(B, T), max(B, T)
    ext = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
    return arrays_to_Geo([ext], kind=kind, info="extent to poly")


# ==== ====================================================
# ---- distance related
def eucl_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum

    Parameters
    ----------
    a, b : array like
        Inputs, list, tuple, array in 1, 2 or 3D form
    metric : string
        euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),

    Notes
    -----
    mini e_dist for 2d points array and a single point

    >>> def e_2d(a, p):
            diff = a - p[np.newaxis, :]  # a and p are ndarrays
            return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    >>> a.shape  # (5, 2)
    >>> a[:, np.newaxis]  # (5, 1, 2)
    >>> (np.prod(a.shape[:-1]), 1, a.shape[-1])  # (5, 1, 2)

    See Also
    --------
    `arraytools` has more functions and documentation.
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    if a.ndim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a.ndim >= 2:
        a = a[:, np.newaxis]  # see old version above
    if b.ndim > 2:
        b = b[:, np.newaxis]  # ditto
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


def _dist_along_(a, dist=0):
    """Add a point along a poly feature at a distance from the start point.

    Parameters
    ----------
    dist : number
      `dist` is assumed to be a value between 0 and to total length of the
      poly feature.  If <= 0, the first point is returned.  If >= total
      length the last point is returned.

    Notes
    -----
    Determine the segment lengths and the cumulative length.  From the latter,
    locate the desired distance relative to it and the indices of the start
    and end points.

    The coordinates of those points and the remaining distance is used to
    derive the location of the point on the line.

    See Also
    --------
    _percent_along_ : function
        Similar to this function but measures distance as a percentage.
    """
    dxdy = a[1:, :] - a[:-1, :]                        # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    cumleng = np.concatenate(([0], np.cumsum(leng)))   # cumulative length
    if dist <= 0:              # check for faulty distance or start point
        return a[0]
    if dist >= cumleng[-1]:    # check for greater distance than cumulative
        return a[-1]
    _end_ = np.digitize(dist, cumleng)
    x1, y1 = a[_end_]
    _start_ = _end_ - 1
    x0, y0 = a[_start_]
    t = (dist - cumleng[_start_]) / leng[_start_]
    xt = x0 * (1. - t) + (x1 * t)
    yt = y0 * (1. - t) + (y1 * t)
    return np.array([xt, yt])


def _percent_along_(a, percent=0):
    """Add a point along a poly feature at a distance from the start point.

    The distance is specified as a percentage of the total poly feature length.

    See Also
    --------
    _dist_along_ : function
        Similar to this function but measures distance as a finite value from
        the start point.

    Requires
    --------
    Called by `pnt_on_poly`.
    """
    if percent > 1.:
        percent /= 100.
    dxdy = a[1:, :] - a[:-1, :]                        # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    cumleng = np.concatenate(([0], np.cumsum(leng)))
    perleng = cumleng / cumleng[-1]
    if percent <= 0:              # check for faulty distance or start point
        return a[0]
    if percent >= perleng[-1]:    # check for greater distance than cumulative
        return a[-1]
    _end_ = np.digitize(percent, perleng)
    x1, y1 = a[_end_]
    _start_ = _end_ - 1
    x0, y0 = a[_start_]
    t = (percent - perleng[_start_])
    xt = x0 * (1. - t) + (x1 * t)
    yt = y0 * (1. - t) + (y1 * t)
    return np.array([xt, yt])


def _pnts_on_line_(a, spacing=1, is_percent=False):  # densify by distance
    """Add points, at a fixed spacing, to an array representing a line.

    **See**  `densify_by_distance` for documentation.

    Parameters
    ----------
    a : array
        A sequence of `points`, x,y pairs, representing the bounds of a polygon
        or polyline object.
    spacing : number
        Spacing between the points to be added to the line.
    is_percent : boolean
        Express the densification as a percent of the total length.

    Notes
    -----
    Called by `pnt_on_poly`.
    """
    N = len(a) - 1                                    # segments
    dxdy = a[1:, :] - a[:-1, :]                       # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    if is_percent:                                    # as percentage
        spacing = abs(spacing)
        spacing = min(spacing / 100, 1.)
        steps = (sum(leng) * spacing) / leng          # step distance
    else:
        steps = leng / spacing                          # step distance
    deltas = dxdy / (steps.reshape(-1, 1))              # coordinate steps
    pnts = np.empty((N,), dtype='O')                  # construct an `O` array
    for i in range(N):              # cycle through the segments and make
        num = np.arange(steps[i])   # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1, -1)        # add the final point and concatenate
    return np.concatenate((*pnts, a0), axis=0)


# ---- buffer, scale
#
def scale_by_area(poly, factor=1, asGeo=False):
    """Scale a polygon geometry by its area.

    Parameters
    ----------
    a : ndarray
        A polygon represented by an ndarray.
    factor : number
        Positive scaling as an integer or decimal number.

    Requires
    --------
    `isGeo`, `_area_bit_` from npGeo

    Notes
    -----
    - Translate to the origin of the unique points in the polygon.
    - Determine the initial area.
    - Scale the coordinates.
    - Shift back to the original center.
    """
    def _area_scaler_(a, factor):
        """Do the work"""
        if factor <= 0.0:
            return None
        a = np.array(a)
        cent = np.mean(np.unique(a, axis=0), axis=0)
        shifted = a - cent
        area_ = _area_bit_(shifted)
        alpha = np.sqrt(factor * area_ / area_)
        scaled = shifted * [alpha, alpha]
        return scaled + cent
    # ----
    if is_Geo(poly):
        final = [_area_scaler_(a, factor) for a in poly.bits]
    else:
        final = _area_scaler_(poly, factor)
    if asGeo:
        a_stack, ift, extent = array_IFT(final, shift_to_origin=False)
        return npg.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
    return final


def offset_buffer(poly, buff_dist=1, keep_holes=False, asGeo=False):
    """Buffer singlepart polygons with squared ends.

    Parameters
    ----------
    poly : ndarray
        The poly feature to buffer in the form of an ndarray.
    buff_dist : number
        The offset/buffer distance.  Positive for expansion, negative for
        contraction.

    Returns
    -------
    A buffer without rounded corners.

    Notes
    -----
    If you want rounded corners, use something else.
    Singlepart shapes supported with or without holes.
    """
    def intersection(p0, p1, p2, p3):
        """Line intersections."""
        x1, y1, x2, y2, x3, y3, x4, y4 = *p0, *p1, *p2, *p3
        dx1, dy1, dx2, dy2 = x2 - x1, y2 - y1, x4 - x3, y4 - y3
        a = x1 * y2 - x2 * y1
        b = x3 * y4 - x4 * y3
        c = dy1 * dx2 - dy2 * dx1
        if 1e-12 < abs(c):
            n1 = (a * dx2 - b * dx1) / c
            n2 = (a * dy2 - b * dy1) / c
            return (n1, n2)
        return (x2, y2)

    def _buff_(bit, buff_dist=1):
        """Offset line"""
        ft_ = []
        segs = []
        bit = np.array(bit)
        for i in range(bit.shape[0] - 1):
            x1, y1, x2, y2 = *bit[i], *bit[i+1]
            r = buff_dist / np.hypot(x2 - x1, y2 - y1)
            vx, vy = (x2 - x1) * r, (y2 - y1) * r
            pnt0 = (x1 - vy, y1 + vx)
            pnt1 = (x2 - vy, y2 + vx)
            ft_.append([pnt0, pnt1])
        f_t = np.array(ft_)
        z = list(zip(f_t[:-1], f_t[1:]))
        z.append([z[-1][-1], z[0][0]])
        z = np.array(z)
        for i, j in z:
            x_tion = intersection(i[0], i[1], j[0], j[1])
            segs.append(x_tion)  # np.array([i[0], middle]))
        frst = np.atleast_2d(segs[-1])
        final = np.concatenate((frst, np.array(segs)), axis=0)
        return final

    def _buffer_array_(poly, buff_dist):
        """Perform the buffering."""
        p0 = poly[:-1]
        p1 = poly[1:]
        diff = p1 - p0
        r = buff_dist/np.sqrt(np.einsum('ij,ij->i', diff, diff))
        vy_vx = (diff * r[:, None] * [1, -1])[:, ::-1]
        pnts0 = p0 + vy_vx
        pnts1 = p1 + vy_vx
        fr_to = np.concatenate((pnts0, pnts1), axis=1).reshape(-1, 2, 2)
        z = list(zip(fr_to[:-1], fr_to[1:]))
        z.append([z[-1][-1], z[0][0]])
        z = np.array(z)
        segs = [intersection(i[0], i[1], j[0], j[1]) for i, j in z]
        frst = np.atleast_2d(segs[-1])
        return np.concatenate((frst, np.array(segs)), axis=0)

    def _buffer_Geo_(poly, buff_dist, keep_holes):
        """Move the Geo array buffering separately"""
        arr = poly.bits
        ift = poly.IFT
        cw = poly.CW
        final = []
        for i, a in enumerate(arr):
            if cw[i] == 0 and keep_holes:
                buff_dist = -buff_dist
                a = a[::-1]
                ext = [np.min(a, axis=0), np.max(a, axis=0)]
                b = _buff_(a, buff_dist)
                in_check = _in_extent_(b, ext)
                if in_check:   # print(buff_dist, a, b, in_check)
                    final.append(b)
            elif cw[i] == 1:   # print(buff_dist, a, b)
                b = _buff_(a, buff_dist)
                final.append(b)
        return final
    # ----
    # Buffer Geo arrays or ndarray
    if is_Geo(poly):
        final = _buffer_Geo_(poly, buff_dist, keep_holes)
    else:
        final = _buffer_array_(poly, buff_dist)
    if asGeo:
        a_stack, ift, extent = array_IFT(final, shift_to_origin=False)
        return npg.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
    return final  # fr_to, z, final


# ===== Workers with Geo and ndarrays. =======================================
# ---- area and centroid helpers
#
def _area_centroid_(a):
    """Calculate area and centroid for a singlepart polygon, `a`.

    This is also used to calculate area and centroid for a Geo array's parts.

    Notes
    -----
    For multipart shapes, just use this syntax:

    >>> # rectangle with hole
    >>> a0 = np.array([[[0., 0.], [0., 10.], [10., 10.], [10., 0.], [0., 0.]],
                      [[2., 2.], [8., 2.], [8., 8.], [2., 8.], [2., 2.]]])
    >>> [npg._area_centroid_(i) for i in a0]
    >>> [(100.0, array([ 5.00,  5.00])), (-36.0, array([ 5.00,  5.00]))]
    """
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    t = e1 - e0
    area = np.sum((e0 - e1) * 0.5)
    x_c = np.sum((x1 + x0) * t, axis=0) / (area * 6.0)
    y_c = np.sum((y1 + y0) * t, axis=0) / (area * 6.0)
    return area, np.asarray([-x_c, -y_c])


# ---- angle related
#
def _angles_(a, inside=True, in_deg=True):
    """Worker for Geo `polygon_angles` and `polyline_angles`.

    Sequential points, a, b, c for the first bit in a shape, so interior holes
    are removed in polygons and the first part of a multipart shape is used.
    Use multipart_to_singlepart if you want to  process that type.

    Parameters
    ----------
    inside : boolean
        True, for interior angles.
    in_deg : boolean
        True for degrees, False for radians.
    """
    # ----
    def _x_(a):
        """Cross product.  see npg_helpers as well."""
        ba = a - np.concatenate((a[-1, None], a[:-1]), axis=0)
        bc = a - np.concatenate((a[1:], a[0, None]), axis=0)
        return np.cross(ba, bc), ba, bc
    # ----
    if np.allclose(a[0], a[-1]):                 # closed loop, remove dupl.
        a = a[:-1]
    cr, ba, bc = _x_(a)
    dt = np.einsum('ij,ij->i', ba, bc)
    ang = np.arctan2(cr, dt)
    TwoPI = np.pi * 2.
    if inside:
        angles = np.where(ang < 0, ang + TwoPI, ang)
    else:
        angles = np.where(ang > 0, TwoPI - ang, ang)
    if in_deg:
        angles = np.degrees(angles)
    return angles


def _rotate_(geo_arr, R, as_group):
    """Rotation helper.

    Parameters
    ----------
    geo_arr : array
        The input geo array, which is split here.
    as_group : boolean
        False, rotated about the extent center.  True, rotated about each
        shapes' center.
    R : array
        The rotation matrix, passed on from Geo.rotate.
    clockwise : boolean
    """
    shapes = geo_arr.shapes
    out = []
    if as_group:
        uniqs = []
        for chunk in shapes:
            _, idx = np.unique(chunk, True, axis=0)
            uniqs.append(chunk[np.sort(idx)])
        cents = [np.mean(i, axis=0) for i in uniqs]
        for i, chunk in enumerate(shapes):
            ch = np.einsum('ij,jk->ik', chunk - cents[i], R) + cents[i]
            out.append(ch)
        return out
    cent = np.mean(geo_arr, axis=0)
    for chunk in shapes:
        ch = np.einsum('ij,jk->ik', chunk - cent, R) + cent
        out.append(ch)
    return out


# ---- convex hull helpers
#
def _ch_scipy_(points):
    """Convex hull using scipy.spatial.ConvexHull.

    Remove null_pnts, calculate
    the hull, derive the vertices and reorder clockwise.
    """
    out = CH(points)
    ch = out.points[out.vertices][::-1]
    return np.concatenate((ch, [ch[0]]), axis=0)


def _ch_simple_(points):
    """Calculate the convex hull for given points.

    Removes null_pnts, finds the unique points, then determines the hull from
    the remaining.
    """
    def _x_(o, a, b):
        """Cross product for vectors o-a and o-b... a<--o-->b."""
        xo, yo = o
        xa, ya = a
        xb, yb = b
        return (xa - xo) * (yb - yo) - (ya - yo) * (xb - xo)
    # ----
    _, idx = np.unique(points, return_index=True, axis=0)
    points = points[idx]
    if len(points) <= 3:
        return points
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and _x_(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _x_(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    ch = np.array(lower[:-1] + upper)[::-1]  # sort clockwise
    if np.all(ch[0] != ch[-1]):
        ch = np.concatenate((ch, [ch[0]]), axis=0)  # np.vstack((ch, ch[0]))
    return ch


def _ch_(points, threshold=50):
    """Perform a convex hull using either simple methods or scipy's."""
    if len(points) > threshold:
        return _ch_scipy_(points)
    return _ch_simple_(points)


# ---- poly conversion helpers
#
def _polys_to_unique_pnts_(a, as_structured=True):
    """Based on `polys_to_points`.

    Allows for recreation of original point order and unique points.
    """
    uni, idx, cnts = np.unique(a, True, return_counts=True, axis=0)
    if as_structured:
        N = uni.shape[0]
        dt = [('New_ID', '<i4'), ('Xs', '<f8'), ('Ys', '<f8'), ('Num', '<i4')]
        z = np.zeros((N,), dtype=dt)
        z['New_ID'] = idx
        z['Xs'] = uni[:, 0]
        z['Ys'] = uni[:, 1]
        z['Num'] = cnts
        return z[np.argsort(z, order='New_ID')]
    return np.asarray(uni)


def _simplify_lines_(a, deviation=10):
    """Simplify array."""
    ang = _angles_(a, inside=True, in_deg=True)
    idx = (np.abs(ang - 180.) >= deviation)
    sub = a[1: -1]
    p = sub[idx]
    return a, p, ang


def segments_to_polys(self):
    """Return segments from one of the above to their original form."""
    return np.vstack([i.reshape(2, 2) for i in self])


# ----------------------------------------------------------------------------
# ---- points in, or on, geometries
#
def pnts_in_Geo_extents(pnts, geo):
    """Return points in the extent of a Geo array.

    `g.extents_pnts` provides the bounding geometry for the `inside` check.

    Parameters
    ----------
    pnts : array-like
        The points to query.
    geo : Geo array
        The Geo array delineating the query space.

    Requires
    --------
    `crossing_num` from npgeom.npg_helpers.

    Example
    -------
    Three polygons and their extent:

    >>> # ---- extent Geo    and  IFT for each shape
    >>> ext = g.extents_pnts(False, True)  # by part not bit
    >>> Geo([[ 0.,  0.],  LB array([[0, 0, 2, 0, 1, 0],
    ...      [10., 10.],  RT
    ...      [10.,  0.],  LB        [1, 2, 4, 0, 1, 0],
    ...      [25., 14.],  RT
    ...      [10., 10.],  LB        [2, 4, 6, 0, 1, 0]], dtype=int32)
    ...      [15., 18.]]) RT
    >>> ext = np.array([[400, 400], [600, 600]])
    >>> pnts = np.array([[1., 1], [2.5, 2.5], [4.5, 4.5], [5., 5],
        ...              [6, 6], [10, 10], [12., 12], [12, 16]])
    """
    # extents = geo.extent_pnts(splitter="shape", asGeo=True)
    # LB = extents[::2]
    # RT = extents[1::2]
    extents = geo.extents("Shape")
    LB = extents[:, :2]
    RT = extents[:, 2:]
    comp = np.logical_and(LB[:, None] <= pnts, pnts <= RT[:, None])
    idx = np.logical_and(comp[..., 0], comp[..., 1])
    idx_t = idx.T
    x, y = np.meshgrid(np.arange(len(LB)), np.arange(len(pnts)), sparse=True)
    p_inside = [pnts[idx_t[:, i]] for i in range(idx_t.shape[-1])]
    inside = []
    for poly in geo.bits:  # outer_rings(False):
        c_n = crossing_num(pnts, poly)
        inside.append(c_n)
    return idx_t, p_inside, inside  # , x0, y0, pnts_by_extent, comb


def pnts_in_pnts(pnts, geo, just_common=True):
    """
    Check to see if pnts are coincident (common) with pnts in a Geo array.

    Parameters
    ----------
    pnts : ndarray
        Nx2 array of points.
    geo : ndarray or Geo array.
        Nx2 array of points.
    just_common : boolean
        If ``just_common`` is True, only the points in both data sets are
        returned.  If False, then the common and unique points are returned as
        two separate arrays.
        If one of the two arrays is empty ``None`` will be returned for that
        array.

    See Also
    --------
    `npGeo._pnts_in_geo` for Geo arrays explicitly.
    """
    w = np.where((pnts == geo[:, None]).all(-1))[1]
    if len(w) > 0:
        common = np.unique(pnts[w], axis=0)
        if just_common:
            return common, None
        w1 = np.where((pnts == common[:, None]).all(-1))[1]
        idx = [i for i in np.arange(len(pnts)) if i not in w1]
        if len(idx) > 0:
            uniq = pnts[idx]
            return uniq, common
        return None, common
    return pnts, None


def pnts_in_Geo(pnts, geo, uniq_pnts=True):
    """Geo array implementation of points in polygon. `pntply`.

     Crossing number is used to determine whether a point is completely inside
     or on the boundary of a polygon.

    Parameters
    ----------
    pnts : array (N, 2)
       An ndarray of point objects.
    g : Geo array
        The Geo array of singlepart polygons.
     (ms)  pnts
      8.8  1e02
      6.6  1e03
     45.2  1e04
    461    1e05
    4.17s  1e06  4 shapes, 1.47 for 1 shape
    data = [[p3.bits, 2, 'red', '.', True ], [psrt, 0, 'black', 'o', False]]
    plot_mixed(data, title="Points in Polygons", invert_y=False, ax_lbls=None)
    out, ift, ps, final = pnts_in_Geo(psrt, p3)
    """
    # ----
    def _cr_num_(pnts, poly, line=True):
        """Crossing Number for point(s) in polygon. See full implementation
        in npg.npg_helpers `crossing_num`.
        """
        pnts = np.atleast_2d(pnts)
        xs = poly[:, 0]
        ys = poly[:, 1]
        N = len(poly)
        xy_diff = np.diff(poly, axis=0)
        dx = xy_diff[:, 0]  # np.diff(xs)
        dy = xy_diff[:, 1]  # np.diff(ys)
        is_in = []
        for pnt in pnts:
            cn = 0    # the crossing number counter
            x, y = pnt
            for i in range(N - 1):
                c0 = (ys[i] <= y < ys[i + 1])
                c1 = (ys[i] >= y > ys[i + 1])
                if (c0 or c1) or y in (ys[i], ys[i+1]):
                    if dy[i] != 0:
                        vt = (y - ys[i]) / dy[i]  # compute x-coordinate
                        xcal = (xs[i] + vt * dx[i])
                        if (x == xs[i]) or (x < xcal):  # include
                            cn += 1
            is_in.append(cn % 2)  # either even or odd (0, 1)
        return pnts[np.nonzero(is_in)]

    def _pnts_in_ext_(pnts, geo):
        """Return the indices of points in Geo extents."""
        extents = geo.extents(splitter="shape")
        L = pnts[:, 0][:, None] >= extents[:, 0]
        R = pnts[:, 0][:, None] <= extents[:, 2]
        B = pnts[:, 1][:, None] >= extents[:, 1]
        T = pnts[:, 1][:, None] <= extents[:, 3]
        c_0 = np.logical_and(L, R)
        c_1 = np.logical_and(B, T)
        idx = np.logical_and(c_0, c_1)
        return idx
    #
    # ---- Determine points in the extents of each feature in g
    # main section
    geo = geo.outer_rings(True)  # remove holes, keep as Geo array
    extents = geo.extents(splitter="shape")
    # get the `points in extent` indices
    uni, cnts = np.unique(pnts, return_counts=True, axis=0)
    idx = _pnts_in_ext_(uni, geo)
    p_inside = np.asarray([uni[idx[:, i]] for i in range(idx.shape[-1])])
    out = []
    # cycle through the shapes
    polys = geo.outer_rings(False)
    for i, p in enumerate(p_inside):
        if p.size > 0:
            poly = polys[i]
            cn = _cr_num_(p, poly)  # _cr_num_(p, poly)
            if len(cn) > 0:  # cn.size > 0:
                out.append([geo.shp_IFT[i], cn])
    ift, ps = zip(*out)
    ps = [i for i in ps if len(ps) > 0]
    final = np.unique(np.vstack(ps), axis=0)
    return out, ift, ps, final


def _cr_np_(pnts, poly):
    """Crossing number, using numpy"""
    pnts = np.atleast_2d(pnts)
    yp = pnts[:, 1]
    xp = pnts[:, 0]
    xs = poly[:, 0]
    ys = poly[:, 1]
    xy_diff = np.diff(poly, axis=0)
    dx = xy_diff[:, 0]  # np.diff(xs)
    dy = xy_diff[:, 1]  # np.diff(ys)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = dy != 0
        vt = (yp[:, None] - ys[:-1])/dy  # [:, g] / dy[g]
        xcal = xs[:-1] + (vt * dx)
        xcal = xcal[:, g]   # slice out the good
        c5 = (xp[:, None] == xs[:-1])[:, g]
        c6 = xp[:, None] <= xcal
        c7 = np.logical_or(c5, c6)
        cn_1 = np.sum(c7, axis=1)
        cn_final = cn_1 % 2
        w = np.where(cn_final == 1)[0]
    return pnts[w]


def _pnt_on_poly_(pnt, poly):
    """Find closest point location on a polygon/polyline.

    See : `p_o_p` for batch running of multiple points to a polygon.

    Parameters
    ----------
    pnt : 1D ndarray array
        XY pair representing the point coordinates.
    poly : 2D ndarray array
        A sequence of XY pairs in clockwise order is expected.  The first and
        last points may or may not be duplicates, signifying sequence closure.

    Returns
    -------
    A list of [x, y, distance, angle] for the intersection point on the line.
    The angle is relative to north from the origin point to the point on the
    polygon.

    Notes
    -----
    `e_dist` is represented by _e_2d and pnt_on_seg by its equivalent below.

    `_line_dir_` is from it's equivalent line_dir included here.

    This may be as simple as finding the closest point on the edge, but if
    needed, an orthogonal projection onto a polygon/line edge will be done.
    This situation arises when the distance to two sequential points is the
    same.
    """
    def _e_2d_(a, p):
        """Array points to point distance... mini e_dist."""
        diff = a - p[None, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    def _pnt_on_seg_(seg, pnt):
        """Mini pnt_on_seg function normally required by pnt_on_poly."""
        x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
        dist_ = dx * dx + dy * dy  # squared length
        u = ((x0 - x1) * dx + (y0 - y1) * dy) / dist_
        u = max(min(u, 1), 0)  # u must be between 0 and 1
        xy = (np.array([dx, dy]) * u) + [x1, y1]  # noqa
        return xy

    def _line_dir_(orig, dest):
        """Mini line direction function."""
        orig = np.atleast_2d(orig)
        dest = np.atleast_2d(dest)
        dxy = dest - orig
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        return ang
    #
    pnt = np.asarray(pnt)
    poly = np.asarray(poly)
    if np.all(poly[0] == poly[-1]):  # strip off any duplicate points
        poly = poly[:-1]
    # ---- determine the distances
    d = _e_2d_(poly, pnt)   # abbreviated edist =>  d = e_dist(poly, pnt)
    key = np.argsort(d)[0]  # dist = d[key]
    if key == 0:  # np.vstack((poly[-1:], poly[:3]))
        seg = np.concatenate((poly[-1:], poly[:3]), axis=0)
    elif (key + 1) >= len(poly):  # np.vstack((poly[-2:], poly[:1]))
        seg = np.concatenate((poly[-2:], poly[:1]), axis=0)
    else:
        seg = poly[key - 1: key + 2]       # grab the before and after closest
    n1 = _pnt_on_seg_(seg[:-1], pnt)  # abbreviated pnt_on_seg
    d1 = np.linalg.norm(n1 - pnt)
    n2 = _pnt_on_seg_(seg[1:], pnt)   # abbreviated pnt_on_seg
    d2 = np.linalg.norm(n2 - pnt)
    if d1 <= d2:
        dest = [n1[0], n1[1]]
        ang = _line_dir_(pnt, dest)
        ang = np.mod((450.0 - ang), 360.)
        r = (pnt[0], pnt[1], n1[0], n1[1], d1.item(), ang.item())
        return r
    dest = [n2[0], n2[1]]
    ang = _line_dir_(pnt, dest)
    ang = np.mod((450.0 - ang), 360.)
    r = (pnt[0], pnt[1], n2[0], n2[1], d2.item(), ang.item())
    return r


def _pnt_on_segment_(pnt, seg):
    """Orthogonal projection of a point onto a 2 point line segment.

    Returns the intersection point, if the point is between the segment end
    points, otherwise, it returns the distance to the closest endpoint.

    Parameters
    ----------
    pnt : array-like
        `x,y` coordinate pair as list or ndarray
    seg : array-like
        `from-to points`, of x,y coordinates as an ndarray or equivalent

    Notes
    -----
    >>> seg = np.array([[0, 0], [10, 10]])  # p0, p1
    >>> p = [10, 0]
    >>> pnt_on_seg(seg, p)
    array([5., 5.])

    Generically, with cross products and norms.

    >>> d = np.linalg.norm(np.cross(p1-p0, p0-p))/np.linalg.norm(p1-p0)
    """
    x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
    dist_ = dx * dx + dy * dy  # squared length
    u = ((x0 - x1) * dx + (y0 - y1) * dy) / dist_
    u = max(min(u, 1), 0)
    xy = np.array([dx, dy]) * u + [x1, y1]
    d = xy - pnt
    return xy, np.hypot(d[0], d[1])


def p_o_p(pnts, poly):
    """Run multiple points to a polygon."""
    result = []
    for p in pnts:
        result.append(_pnt_on_poly_(p, poly))
    result = np.asarray(result)
    dt = [('X0', '<f8'), ('Y0', '<f8'), ('X1', '<f8'), ('Y1', '<f8'),
          ('Dist', '<f8'), ('Angle', '<f8')]
    z = np.zeros((len(result),), dtype=dt)
    names = z.dtype.names
    for i, n in enumerate(names):
        z[n] = result[:, i]
    return z



# ---- triangulation, Delaunay helper
#
def _tri_pnts_(pnts):
    """Triangulate the points and return the triangles.

    Parameters
    ----------
    pnts : array
        Points for a shape or a group of points in array format.
        Either geo.shapes or np.ndarray.
    out : array
        An array of triangle points.

    .. note::

       The simplices are ordered counterclockwise, this is reversed in this
       implementation.

    References
    ----------
    `<C:/Arc_projects/Polygon_lineTools/Scripts/triangulate.py>`_.
    """
    pnts = np.unique(pnts, axis=0)    # get the unique points only
    avg = np.mean(pnts, axis=0)
    p = pnts - avg
    tri = Delaunay(p)
    simps = tri.simplices
    # ---- indices holder, fill with indices, repeat first and roll CW
    # translate the points back
    z = np.zeros((len(simps), 4), dtype='int32')
    z[:, :3] = simps
    z[:, 3] = simps[:, 0]
    z = z[:, ::-1]                               # reorder clockwise
    new_pnts = p[z] + avg
    new_pnts = new_pnts.reshape(-1, 2)
    return new_pnts


# ----------------------------------------------------------------------------


# ---- Not included yet -----------------------------------------------------
#

def in_hole_check(pnts, geo):
    """Check points are in a hole."""
    w = np.where(geo.CW == 0)[0]
    holes = geo.bits[w]
    out = []
    for h in holes:
        inside = crossing_num(pnts, h, False)
        if inside.size > 0:
            out.append([h, inside])
    return out


# ===========================================================================
# ---- Extras used elsewhere
'''

def pnt_in_list(pnt, pnts_list):
    """Check to see if a point is in a list of points.

    sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0
    """
    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
    return is_in

'''
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # optional controls here

"""
Demo

r = np.array(['A', 'A', 'B', 'B', 'B', 'A', 'A', 'C', 'C', 'A'], dtype='<U1')
c = np.array(['b', 'a', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'a'], dtype='<U1')
rc = np.array(["{}_{}".format(*i) for i in zip(r, c)])
u, idx, cnts = np.unique(rc, return_index=True, return_counts=True)
dt = [('r_c', u.dtype.str), ('cnts', '<i4')]
ctab = np.array(list(zip(u, cnts)), dtype=dt)
"""
