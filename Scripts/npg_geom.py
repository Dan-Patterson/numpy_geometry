# -*- coding: utf-8 -*-
r"""
----------------------------------
npg_geom: Geometry focused methods
----------------------------------

**Geometry focused methods that work with Geo arrays or np.ndarrays.**

----

Script :
    npg_geom.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-10-07

Purpose
-------
Geometry focused methods that work with Geo arrays or np.ndarrays.
In the case of the former, the methods may be being called from Geo methods
in such things as a list comprehension.

Notes
-----
(1) `_npgeom_notes_.py` contains other notes of interest.

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

(3) How to flatten a searchcursor to points and/or None

>>> in_fc = "C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
>>> SR = npg.getSR(in_fc)
>>> with arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as c:
>>>     pnts = [[[[p for p in arr] for arr in r[1]]] for r in c]
>>> c.reset()  # don't forget to reset the cursor

Example
-------
Sample data

>>> f_name = "C:/Git_Dan/npgeom/data/g_arr.npz"
>>> g, arrs, names = npg.load_geo(f_name, suppress_extras=False)
>>> arr_names = arrs.files  # returns the list of array names inside

- g : the geo array
- arrs : the sub arrays


References
----------
See comment by Serge Tolstov in:

`List of geometry topics
<https://en.wikipedia.org/wiki/List_of_geometry_topics>`_.

`Geometry checks
<https://community.esri.com/thread/244587-check-geometry-fails-in-shared
-origin-edge-case>`_.

**Clipping,intersection references**

`Sutherland-Hodgman polygon clipping
<https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping>`_.

`<http://geomalgorithms.com/a09-_intersect-3.html>`_.

`<https://codereview.stackexchange.com/questions/166702/cythonized-
sutherland-hogman-algorithm>`_.

`Hodgman algorithm
<https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm>`_.

`Atherton clipping
<https://en.wikipedia.org/wiki/Weiler%E2%80%93Atherton_clipping_
algorithm>`_.

`<https://scicomp.stackexchange.com/questions/8895/vertical-and-horizontal
-segments-intersection-line-sweep>`_.
"""

# pylint: disable=C0103, C0302, C0326, C0415, E0611, E1136, E1121
# pylint: disable=R0904, R0914
# pylint: disable=W0201, W0212, W0221, W0612, W0621, W0105
# pylint: disable=R0902

import sys
import numpy as np

from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import repack_fields

from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay

import npGeo as npg

from npg_helpers import (_get_base_, _bit_area_, _bit_min_max_,
                         _in_extent_, _angles_3pnt_)
from npg_pip import np_wn


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 7.2f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=1, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    'pnts_to_extent', 'common_extent', 'extent_to_poly', 'find_closest',
    'eucl_dist', '_dist_along_', '_percent_along_', '_pnts_on_line_',
    'scale_by_area', 'offset_buffer',
    'mabr', '_ch_scipy_', '_ch_simple_', '_ch_',
    'polys_to_unique_pnts', 'polys_to_segments',
    'segments_to_polys', 'simplify_lines',
    'pnts_in_pnts',
    '_pnt_on_poly_', '_pnt_on_segment_', 'p_o_p',
    '_tri_pnts_', 'in_hole_check'
    ]  # 'pnts_in_Geo'


def pnts_to_extent(a, as_pair=False):
    """Return the extent of a geometry. (Left, Bottom, Right, Top).

    Parameters
    ----------
    a : array-like
        An Nx2 array of point objects expected.
    as_pair : boolean
        True, returns a point pair [LB, RT].  False, returns a ravelled array
        [L, B, R, T]

    Notes
    -----
    Uses `_bit_min_max_`.  This is faster for large arrays.
    >>> ext = np.array([a[:, 0].min(), a[:, 1].min(),
    ...                 a[:, 0].max(), a[:, 1].max()])
    """
    a = _get_base_(a)
    ext = _bit_min_max_(a)
    if as_pair:
        ext = ext.reshape(2, 2)
    return ext


def common_extent(a, b):
    """Return the extent overlap for two polygons as L, B, R, T or None"""
    a = _get_base_(a)
    b = _get_base_(b)
    ext0 = np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))
    ext1 = np.concatenate((np.min(b, axis=0), np.max(b, axis=0)))
    es = np.vstack((ext0, ext1))
    maxs = np.max(es, axis=0)
    mins = np.min(es, axis=0)
    L, B = maxs[:2]
    R, T = mins[2:]
    if (L <= R) and (B <= T):
        return np.array([L, B, R, T])  # (x1, y1, x2, y2)
    return None


def extent_to_poly(extent, kind=2):
    """Create a polygon/polyline feature from an array of x,y values.

    The array returned is ordered clockwise with the first and last point
    repeated to form a closed-loop.

    Parameters
    ----------
    extent : array-like
        The extent is specified as four float values in the form of
        L(eft), B(ottom), R(ight), T(op) eg. np.array([5, 5, 10, 10]) or a
        pair of points [LB, RT]
    kind : integer
        A value of 1 for a polyline, or 2 for a polygon.
    """
    shp = extent.shape
    if shp not in [(2, 2), (4,)]:
        print("Check the docs...\n{}".format(extent_to_poly.__doc__))
        return None
    L, B, R, T = extent.ravel()
    L, R = min(L, R), max(L, R)
    B, T = min(B, T), max(B, T)
    ext = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
    return npg.arrays_to_Geo([ext], kind=kind, info="extent to poly")


# ==== ====================================================
# ---- distance related
def find_closest(a, pnt):
    """Find the closest point within a Geo array, its index and distance."""
    dist = eucl_dist(a, pnt)
    idx = np.argmin(dist)
    return np.asarray(a[idx]), idx, dist[idx]


def eucl_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum.

    Parameters
    ----------
    a, b : array like
        Inputs, list, tuple, array in 1, 2 or 3D form.
    metric : string
        Euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),

    Notes
    -----
    Mini e_dist for 2d points array and a single point.

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
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
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
    a = _get_base_(a)
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
    a = _get_base_(a)
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
    a = _get_base_(a)
    N = len(a) - 1                                    # segments
    dxdy = a[1:, :] - a[:-1, :]                       # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    if is_percent:                                    # as percentage
        spacing = abs(spacing)
        spacing = min(spacing / 100, 1.)
        steps = (sum(leng) * spacing) / leng          # step distance
    else:
        steps = leng / spacing                        # step distance
    deltas = dxdy / (steps.reshape(-1, 1))            # coordinate steps
    pnts = np.empty((N,), dtype='O')                  # construct an `O` array
    for i in range(N):              # cycle through the segments and make
        num = np.arange(steps[i])   # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1, -1)       # add the final point and concatenate
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
    `is_Geo` from npGeo and `_bit_area_` from npg_helpers.

    Notes
    -----
    - Translate to the origin of the unique points in the polygon.
    - Determine the initial area.
    - Scale the coordinates.
    - Shift back to the original center.
    """
    def _area_scaler_(a, factor):
        """Do the work."""
        if factor <= 0.0:
            return None
        a = np.array(a)
        cent = np.mean(np.unique(a, axis=0), axis=0)
        shifted = a - cent
        area_ = _bit_area_(shifted)
        alpha = np.sqrt(factor * area_ / area_)
        scaled = shifted * [alpha, alpha]
        return scaled + cent
    # ----
    if npg.is_Geo(poly):
        final = [_area_scaler_(a, factor) for a in poly.bits]
    else:
        final = _area_scaler_(poly, factor)
    if asGeo:
        a_stack, ift, extent = npg.array_IFT(final, shift_to_origin=False)
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
        """Move the Geo array buffering separately."""
        arr = poly.bits
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
    if npg.is_Geo(poly):
        final = _buffer_Geo_(poly, buff_dist, keep_holes)
    else:
        final = _buffer_array_(poly, buff_dist)
    if asGeo:
        a_stack, ift, extent = npg.array_IFT(final, shift_to_origin=False)
        return npg.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
    return final  # fr_to, z, final


# ---- minimum area bounding rectangle (mabr)
#
def mabr(polys, p_centers, p_angles):
    """Determine the minimum area bounding rectangle for polygons.

    Called by the class method `min_area_rect` in npGeo.

    Parameters
    ----------
    polys : array
        These shapes should be the convex hull of the shape points.
    p_centers : array
        Extent centers of the convex hulls (polys).
    p_angles : array
        The perimeter/segment angles making up the shape.

    Returns
    -------
    This is the MABR... minimum area bounding rectangle.
    """

    def _LBRT_(a):
        """Extent of a sub-array in an object array."""
        return np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))

    def _extent_area_(a):
        """Area of an extent polygon."""
        LBRT = _LBRT_(a)
        dx, dy = np.diff(LBRT.reshape(2, 2), axis=0).squeeze()
        return dx * dy, LBRT

    def _rot_(a, cent, angle, clockwise):
        """Rotate shapes about their center. Specify `angle` in degrees."""
        angle = np.radians(angle)
        if clockwise:
            angle = -angle
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, -s), (s, c)))
        return np.einsum('ij,jk->ik', a - cent, R) + cent
    # ----
    # Determine their convex hulls for the outer rings.
    # Obtain the angles, extents and centers for each hull.
    rects = []
    for i, ch in enumerate(polys):  # chs_):   # first quadrant
        uni_ = np.unique(p_angles[i] % 180.)
        # uni_ = uni_[inv]  # [i for i in uni_ if i not in [90.]]
        _, LBRT = _extent_area_(ch)
        area_old = np.inf
        Xmin, Ymin, Xmax, Ymax = LBRT
        vals = [area_old, p_centers[i], np.inf, Xmin, Ymin, Xmax, Ymax]
        for angle in uni_:
            ch2 = _rot_(ch, p_centers[i], angle, False)  # translate, rotate
            area_, LBRT = _extent_area_(ch2)  # ---- determine area
            Xmin, Ymin, Xmax, Ymax = LBRT
            if area_ <= area_old:
                area_old = area_
                Xmin, Ymin, Xmax, Ymax = LBRT
                vals = [area_, p_centers[i], angle, Xmin, Ymin, Xmax, Ymax]
        rects.append(vals)
    rects = np.asarray(rects, dtype='O')
    return rects


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
def polys_to_unique_pnts(a, as_structured=True):
    """Based on `polys_to_points`.

    Allows for recreation of original point order and unique points.
    Structured arrays is used for sorting.
    """
    uni, idx, cnts = np.unique(uts(a), return_index=True,
                               return_counts=True, axis=0)
    uni = stu(uni)
    if as_structured:
        N = uni.shape[0]
        dt = [('New_ID', '<i4'), ('Xs', '<f8'), ('Ys', '<f8'), ('Num', '<i4')]
        z = np.zeros((N,), dtype=dt)
        z['New_ID'] = idx
        z['Xs'] = uni[:, 0]
        z['Ys'] = uni[:, 1]
        z['Num'] = cnts
        return z[np.argsort(z, order='New_ID')]
    return a[np.sort(idx)]


def polys_to_segments(self, as_basic=True, to_orig=False, as_3d=False):
    """Segment poly* structures into o-d pairs from start to finish.

    as_basic : boolean
        True, returns an Nx4 array (x0, y0, x1, y1) of from-to coordinates.
        False, returns a structured array
        If `as_3d` is True, then `as_basic` is set to False.
    to_origin : boolean
        True, moves the coordinates back to their original position
        defined by the `LL` property of the Geo array.
    as_3d : boolean
        True, the point pairs are returned as a 3D array in the form
        [[X_orig', Y_orig'], ['X_dest', 'Y_dest']], without the distances.

    Notes
    -----
    Use `prn_tbl` if you want to see a well formatted output.
    """
    if self.K not in (1, 2):
        print("Poly* features required.")
        return None
    # ---- basic return as ndarray used by common_segments
    if as_3d:  # The array cannot be basic if it is 3d
        as_basic = False
    if to_orig:
        tmp = self.XY + self.LL
        b_vals = [tmp[ft[0]:ft[1]] for ft in self.FT]   # shift to orig extent
    else:
        b_vals = [b for b in self.bits]
    # ---- Do the concatenation
    fr_to = np.concatenate([np.concatenate((b[:-1], b[1:]), axis=1)
                            for b in b_vals], axis=0)
    # ---- return if simple and not 3d representation
    if as_basic:
        return fr_to
    # ---- return 3d from-to representation
    if as_3d:
        fr_to = fr_to[:, :4]
        s0, s1 = fr_to.shape
        return fr_to.reshape(s0, s1//2, s1//2)
    # ----structured array section
    # add bit ids and lengths to the output array
    b_ids = self.IFT
    segs = np.asarray([[[b_ids[i][0], *(b_ids[i][-2:])], len(b) - 1]
                       for i, b in enumerate(b_vals)], dtype='O')
    s_ids = np.concatenate([np.tile(i[0], i[1]).reshape(-1, 3)
                            for i in segs], axis=0)
    dist = (np.sqrt(np.sum((fr_to[:, :2] - fr_to[:, 2:4])**2, axis=1)))
    fr_to = np.hstack((fr_to, s_ids, dist.reshape(-1, 1)))
    dt = np.dtype([('X_fr', 'f8'), ('Y_fr', 'f8'), ('X_to', 'f8'),
                   ('Y_to', 'f8'), ('Orig_id', 'i4'), ('Part', 'i4'),
                   ('Seq_ID', 'i4'), ('Length', 'f8')])
    fr_to = uts(fr_to, dtype=dt)
    return repack_fields(fr_to)


def segments_to_polys(self):
    """Return segments from one of the above to their original form."""
    return np.vstack([i.reshape(2, 2) for i in self])


def simplify_lines(a, deviation=10):
    """Simplify array. Requires, `_angles_3pnt_` from npg_helpers."""
    ang = _angles_3pnt_(a, inside=True, in_deg=True)
    idx = (np.abs(ang - 180.) >= deviation)
    sub = a[1: -1]
    p = sub[idx]
    return a, p, ang


# ----------------------------------------------------------------------------
# ---- points in, or on, geometries
#

def pnts_in_pnts(pnts, geo, just_common=True):
    """Check to see if pnts are coincident (common) with pnts in a Geo array.

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
    `npGeo.pnts_in_Geo` for Geo arrays explicitly.
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


# ==========================================================================
#
# ---- Geo, pnt on polygon
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
        seg = poly[key - 1: key + 2]  # grab the before and after closest
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
        `from-to points`, of x,y coordinates as an ndarray or equivalent.

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
    # new_pnts = new_pnts.reshape(-1, 2)
    return new_pnts.tolist()


# ----------------------------------------------------------------------------


# ---- Not included yet -----------------------------------------------------
#
def bin_pnts(pnts, x_bins=None, y_bins=None):
    """Bin points using a 2D bin.

    Parameters
    ----------
    pnts : array
        An Nx2 array of point objects.
    x_bins, y_bins : array-like
        A sequence of incrementing bin thresholds.

    Example
    -------
    >>> np.histogramdd(g4, bins=[(0, 10, 20, 30, 40), (0, 10, 20, 30, 40)])
    (array([[ 1.00,  0.00,  0.00,  0.00],
    ...     [ 4.00,  3.00,  2.00,  0.00],
    ...     [ 3.00,  3.00,  0.00,  0.00],
    ...     [ 1.00,  0.00,  0.00,  0.00]]),
    [array([ 0, 10, 20, 30, 40]), array([ 0, 10, 20, 30, 40])])

    Where the first array are the counts, and the next two arrays are the bin
    edges for the X and Y values.

    References
    ----------
    `Aggregate points
    <https://pro.arcgis.com/en/pro-app/tool-reference/geoanalytics-desktop/
    aggregate-points.htm>`_.
    """
    if x_bins is None:
        x_bins = (pnts[:, 0].max() - pnts[:, 0].min()) / 10
    if y_bins is None:
        y_bins = (pnts[:, 0].max() - pnts[:, 0].min()) / 10
    h = np.histogramdd(pnts, [x_bins, y_bins])
    return h


def in_hole_check(pnts, geo):
    """Check if points are in a hole."""
    w = np.where(geo.CW == 0)[0]
    holes = geo.bits[w]
    out = []
    for h in holes:
        inside = np_wn(pnts, h)  # crossing_num(pnts, h, False)
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
x    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
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
