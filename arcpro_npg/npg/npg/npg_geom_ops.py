# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
----------------------------------
npg_geom_ops: Geometry focused methods
----------------------------------

**Geometry focused methods that work with Geo arrays or np.ndarrays.**

----

Script :
    npg_geom_ops.py

Author :
    Dan_Patterson

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2024-07-14

Purpose
-------
Geometry focused methods that work with Geo arrays or np.ndarrays.
In the case of the former, the methods may be being called from Geo methods
in such things as a list comprehension.

Notes
-----
(1) `"C:/arcpro_npg/docs/_npgeom_notes_.txt` contains other notes of interest.

(2) See references for the origin of this quote.

"For an Esri polygon to be simple, all intersections have to occur at
vertices. In general, it follows from 'a valid Esri polygon must have
such structure that the interior of the polygon can always be unambiguously
determined to be to the right of every segment', that"::

    - segments can touch other segments only at the end points,
    - segments have non-zero length,
    - outer rings are clockwise and holes are counterclockwise,
    - each polygon ring has non-zero area.
    - order of the rings does not matter,
    - rings can be self-tangent,
    - rings can not overlap.

(3) How to flatten a searchcursor to points and/or None.

>>> in_fc = "C:/arcpro_npg/Project_npg/npgeom.gdb/Polygons"
>>> SR = npg.getSR(in_fc)
>>> with arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as c:
>>>     pnts = [[[[p for p in arr] for arr in r[1]]] for r in c]
>>> c.reset()  # don't forget to reset the cursor

Example
-------
Sample data

>>> f_name = "C:/arcpro_npg/data/g_arr.npz"
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

**Clipping, intersection references**

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


**Generating __all__ and __helpers__**

>>> not_in = [
...     '__all__', '__builtins__', '__cached__', '__doc__', '__file__',
...     '__loader__', '__name__', '__package__', '__spec__', 'np', 'npg', 'sys'
...     ] + __imports__

>>> __all__ = [i for i in dir(npg.npg_geom_ops)
...            if i[0] != "_" and i not in not_in]

>>> __helpers__ = [i for i in dir(npg.npg_geom_ops)
...                if i[0] == "_" and i not in not_in]

"""

# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121

import sys
import numpy as np

from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import repack_fields

from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay

# import npGeo
from npg import npGeo, npg_geom_hlp, npg_pip  # noqa
from npg.npg_geom_hlp import (
    _get_base_, _bit_area_, _bit_min_max_, _in_extent_, _angles_3pnt_)
from npg.npg_pip import np_wn

# np.set_printoptions(
#     edgeitems=10, linewidth=100, precision=2, suppress=True, threshold=200,
#     formatter={"bool": lambda x: repr(x.astype(np.int32)),
#                "float_kind": '{: 6.2f}'.format})
# np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# -- See script header

__all__ = [
    'bin_pnts', 'common_extent', 'densify_by_distance', 'densify_by_factor',
    'eucl_dist', 'extent_to_poly', 'find_closest', 'in_hole_check', 'mabr',
    'npGeo', 'np_wn', 'offset_buffer', 'on_line_chk', 'pnts_in_pnts',
    'pnts_on_poly', 'pnts_to_extent', 'polys_to_segments',
    'polys_to_unique_pnts', 'scale_by_area', 'segments_to_polys', 'simplify',
    'simplify_lines', 'triangulate_pnts'
]

__helpers__ = [
    '_add_pnts_on_line_', '_bit_min_max_', '_ch_', '_ch_scipy_',
    '_ch_simple_', '_closest_pnt_on_poly_', '_dist_along_', '_e_2d_',
    '_is_pnt_on_line_', '_percent_along_', '_pnt_on_segment_',
    'view_as_struct_'
]

__imports__ = [
    'CH', 'Delaunay', 'npGeo', 'np_wn', 'npg_geom_hlp', 'npg_pip', 'stu',
    'uts', '_get_base_', '_bit_area_', '_bit_min_max_', '_in_extent_',
    '_angles_3pnt_'
]


# ---- ---------------------------
# ---- general helpers
#
def _view_as_struct_(a, return_all=False):
    """Key function to get uniform 2d arrays to be viewed as structured arrays.

    A bit of trickery, but it works for all set-like functionality.
    Use `uts` for more complicated dtypes.

    Parameters
    ----------
    a : array
        Geo array or ndarray to be viewed.

    Returns
    -------
    Array view as structured/recarray, with shape = (N, 1)

    References
    ----------
    See `unstructured_to_structured` in... numpy/lib/recfunctions.py

    >>> from numpy.lib.recfunctions import unstructured_to_structured as uts
    """
    shp = a.shape
    dt = a.dtype
    a_view = a.view(dt.descr * shp[1])[..., 0]
    if return_all:
        return a_view, shp, dt
    return a_view


# ---- ---------------------------
# ---- (1) distance helpers
#
def _e_2d_(a, p):
    """Array points, `a`,  to point `p`, distance.

    Use to determine distance of a point to a segment or segments.
    """
    diff = a - p[None, :]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


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
    if dist >= cumleng[-1]:    # check for distance greater than cumulative
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
    t = percent - perleng[_start_]
    xt = x0 * (1. - t) + (x1 * t)
    yt = y0 * (1. - t) + (y1 * t)
    return np.array([xt, yt])


def _is_pnt_on_line_(start, end, xy, tolerance=0.0):
    """Perform a distance check of whether a point is on a line.

    Parameters
    ----------
    start, end, xy : points, array-like
    tolerance : float
        Acceptable distance from the line.

    Notes
    -----
    `tolerance` is normally not needed unless you want to examine points
    quite close to a segment.

    eps = 2**-52 = 2.220446049250313e-16
    np.finfo(float).eps = 2.220446049250313e-16
    np.finfo(float)
    finfo(resolution=1e-15, min=-1.7976931348623157e+308,
          max=1.7976931348623157e+308, dtype=float64)
    """
    #
    def sq_dist(a, b):
        """Add math.sqrt() for actual distance."""
        return (b[0] - a[0])**2 + (b[1] - a[1])**2
    #
    dl = sq_dist(start, end)  # -- line distance
    ds = sq_dist(start, xy)   # -- distance to start from pnt `xy`
    de = sq_dist(end, xy)     # -- distance to end from pnt `xy`
    d0, d1, d2 = np.sqrt([ds, de, dl])  # -- return the sqrt values
    if tolerance == 0.0:
        return d0 + d1 == d2
    d = (d0 + d1) - d2
    return -tolerance <= d <= tolerance


def _add_pnts_on_line_(a, spacing=1, is_percent=False):  # densify by distance
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
    Called by `densify_by_distance`.

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
    a0 = a[-1].reshape(1, -1)       # create the final point and concatenate
    return np.concatenate((*pnts, a0), axis=0)


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


def _closest_pnt_on_poly_(pnt, poly):
    """Find closest point location on a polygon/polyline.

    See : `pnts_on_poly` for batch running of multiple points to a polygon.

    Parameters
    ----------
    pnt : 2D ndarray array
        XY pair representing the point coordinates.
    poly : 2D ndarray array
        A sequence of XY pairs in clockwise order is expected.  The first and
        last points may or may not be duplicates, signifying sequence closure.

    Requires
    --------
    `_e_2d_` is required

    Returns
    -------
    A list of [(x0, y0), (x1, y1), distance, angle] values for the from point
    and the intersection point on the line. The angle is relative to north
    from the origin point to the point on the polygon.

    Notes
    -----
    `e_dist` is represented by _e_2d and pnt_on_seg by its equivalent below.

    `_line_dir_` is from it's equivalent line_dir included here.

    This may be as simple as finding the closest point on the edge, but if
    needed, an orthogonal projection onto a polygon/line edge will be done.
    This situation arises when the distance to two sequential points is the
    same.
    """
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
    # -- determine the distances
    d = _e_2d_(poly, pnt)   # abbreviated edist =>  d = e_dist(poly, pnt)
    key = np.argsort(d)[0]  # dist = d[key]
    if key == 0:            # np.vstack((poly[-1:], poly[:3]))
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


def on_line_chk(start, end, xy, tolerance=1.0e-12):
    """Perform a distance check of whether a point is on a line.

    Parameters
    ----------
    start, end, xy : array_like
        The x,y values for the points.
    tolerance : number
        Acceptable distance tolerance to account for floating point issues.

    Returns
    -------
    A boolean indicating whether the x,y point is on the line and a list of
    values as follows::
        [xy] : if start or end equals xy
        [start, xy, d0] : `xy` is closest to `start` with a distance `d0`.
        [xy, end, d1] : `xy` is closest to `end` with a distance of `d1`.
        [] : the empty list is returned when `xy` is not on the line.

    See Also
    --------
    `npg_geom_ops.is_pnt_on_line` use if just a boolean check is required.
    """
    #
    def dist(a, b):
        """Actual distance."""
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    #
    # boolean checks for start, end xy equality
    if (start == xy).all():
        return True, [xy]
    if (end == xy).all():
        return True, [xy]
    line_leng = dist(start, end)
    d0, d1 = dist(start, xy), dist(end, xy)
    d = (d0 + d1) - line_leng
    chk = -tolerance <= d <= tolerance
    if chk:  # -- xy is on line
        if d0 <= d1:  # -- closest to start
            return chk, [start, xy, d0]
        return chk, [xy, end, d1]  # -- closest to end
    return chk, []  # -- not on line


def find_closest(a, pnt):
    """Find the closest point within a Geo array, its index and distance.

    See Also
    --------
    `closest_pnt_on_poly` is used for point on poly* features and includes
    any required projection onto their edges for the new point feature.

    `_pnt_on_segment_` is similar but does point placement.
    """
    dist = _e_2d_(a, pnt)
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


# ---- ---------------------------
# ---- (2) extent functions
#
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
    """Return the extent overlap for two polygons as L, B, R, T or None."""
    a = _get_base_(a)
    b = _get_base_(b)
    ext0 = np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))
    ext1 = np.concatenate((np.min(b, axis=0), np.max(b, axis=0)))
    es = np.concatenate((ext0[None, :], ext1[None, :]), axis=0)
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
    vals = npGeo.arrays_to_Geo(
        [ext], kind=kind, info="extent to poly", to_origin=False)
    return vals


# ---- ---------------------------
# ---- (3) densify/simplify
#
def densify_by_factor(a, factor=2):
    """Densify a 2D array using np.interp.

    Parameters
    ----------
    a : array
        A 2D array of points representing a polyline/polygon boundary.
    fact : number
        The factor to density the line segments by.

    Notes
    -----
    The original construction of `c` rather than the zero's approach.

    >>> c0 = c0.reshape(n, -1)
    >>> c1 = c1.reshape(n, -1)
    >>> c = np.concatenate((c0, c1), axis=1)
    """
    a = np.squeeze(a)
    n_fact = len(a) * factor
    b = np.arange(0, n_fact, factor)
    b_new = np.arange(n_fact - 1)     # Where you want to interpolate
    c0 = np.interp(b_new, b, a[:, 0])
    c1 = np.interp(b_new, b, a[:, 1])
    n = c0.shape[0]
    c = np.zeros((n, 2))
    c[:, 0] = c0
    c[:, 1] = c1
    return c


def densify_by_distance(a, spacing):
    r"""Return the wrapper for `pnts_on_line`.

    Example
    -------
    >>> a = np.array([[0., 0.], [3., 4.], [3., 0.], [0., 0.]])  # 3x4x5 rule
    >>> a.T
    array([[0., 3., 3., 0.],
           [0., 4., 0., 0.]])
    >>> pnts_on_line(a, spacing=2).T  # take the transpose to facilitate view
    ... array([[0. , 1.2, 2.4, 3. , 3. , 3. , 1. , 0. ],
    ...        [0. , 1.6, 3.2, 4. , 2. , 0. , 0. , 0. ]])
    ... array([[0.,  . . . .   3., . .   3., . . . 0. ],
    ...        [0.,  . . . .   4., . .   0., . . . 0. ]])

    >>> letter `C` and skinny `C`
    >>> a = np.array([[ 0, 0], [ 0, 100], [100, 100], [100,  80],
                      [ 20,  80], [ 20, 20], [100, 20], [100, 0], [ 0, 0]])
    >>> b = np.array([[ 0., 0.], [ 0., 10.], [10., 10.], [10.,  8.],
                      [ 2., 8.], [ 2., 2.], [10., 2.], [10., 0.], [ 0., 0.]])

    Notes
    -----
    The return value could be np.vstack((*pnts, a[-1])) using the last point
    directly, but np.concatenate with a reshaped a[-1] is somewhat faster.
    All entries to the stacking must be ndim=2.

    References
    ----------
    `<https://stackoverflow.com/questions/54665326/adding-points-per-pixel-
    along-some-axis-for-2d-polygon>`_.

    `<https://stackoverflow.com/questions/51512197/python-equidistant-points
    -along-a-line-joining-set-of-points/51514725>`_.
    """
    return _add_pnts_on_line_(a, spacing)


def simplify(arr, tol=1e-6):
    """Remove redundant points on a poly perimeter."""
    x1, y1 = arr[:-2].T
    x2, y2 = arr[1:-1].T
    x3, y3 = arr[2:].T
    result = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    whr = np.nonzero(np.abs(result) >= tol)[0]
    bits = arr[1:-1][whr]
    keep = np.concatenate((arr[0, None], bits, arr[-1, None]), axis=0)
    return keep


# ---- ---------------------------
# ---- (4) buffer, scale
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
    `is_Geo` from npGeo and `_bit_area_` from npg_geom_hlp.

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
    # --
    if npGeo.is_Geo(poly):
        if poly.K != 2:
            print("\n Polygon geo array required.")
            return None
        final = [_area_scaler_(a, factor) for a in poly.bits]
    else:
        final = _area_scaler_(poly, factor)
    if asGeo:
        a_stack, ift, extent = npGeo.array_IFT(final, shift_to_origin=False)
        return npGeo.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
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
        if abs(c) > 1e-12:
            n1 = (a * dx2 - b * dx1) / c
            n2 = (a * dy2 - b * dy1) / c
            return (n1, n2)
        return (x2, y2)

    def _buff_(bit, buff_dist=1):
        """Offset line."""
        ft_ = []
        segs = []
        bit = np.array(bit)
        for i in range(bit.shape[0] - 1):
            x1, y1, x2, y2 = *bit[i], *bit[i + 1]
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
        r = buff_dist / np.sqrt(np.einsum('ij,ij->i', diff, diff))
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
        cw = poly.CL
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
    # --
    # Buffer Geo arrays or ndarray
    if npGeo.is_Geo(poly):
        final = _buffer_Geo_(poly, buff_dist, keep_holes)
    else:
        final = _buffer_array_(poly, buff_dist)
    if asGeo:
        a_stack, ift, extent = npGeo.array_IFT(final, shift_to_origin=False)
        return npGeo.Geo(a_stack, IFT=ift, Kind=2, Extent=extent, Info=None)
    return final  # fr_to, z, final


# ---- ----------------------------
# ---- (5) convex hulls
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
    r"""Calculate the convex hull for given points.

    Removes null_pnts, finds the unique points, then determines the hull from
    the remaining.
    """
    def _x_(o, a, b):
        """Cross product for vectors o-a and o-b... a<--o-->b."""
        xo, yo = o
        xa, ya = a
        xb, yb = b
        return (xa - xo) * (yb - yo) - (ya - yo) * (xb - xo)
    # --
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


# ---- ---------------------------
# ---- (6) mabr (min. area bounding rectangle)
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
    # --
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
            area_, LBRT = _extent_area_(ch2)  # -- determine area
            Xmin, Ymin, Xmax, Ymax = LBRT
            if area_ <= area_old:
                area_old = area_
                Xmin, Ymin, Xmax, Ymax = LBRT
                vals = [area_, p_centers[i], angle, Xmin, Ymin, Xmax, Ymax]
        rects.append(vals)
    rects = np.asarray(rects, dtype='O')
    return rects


# ---- ---------------------------
# ---- (7) triangulation, Delaunay helper
#
def triangulate_pnts(pnts):
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
    # -- indices holder, fill with indices, repeat first and roll CL
    # translate the points back
    z = np.zeros((len(simps), 4), dtype='int32')
    z[:, :3] = simps
    z[:, 3] = simps[:, 0]
    z = z[:, ::-1]                    # reorder clockwise
    new_pnts = p[z] + avg
    # new_pnts = new_pnts.reshape(-1, 2)
    return new_pnts.tolist()


# ---- ---------------------------
# ---- (8) poly* conversion
#
def polys_to_unique_pnts(a, as_structured=True):
    """Based on `polys_to_points`.

    Allows for recreation of original point order and unique points.
    Structured arrays is used for sorting.
    """
    a = _view_as_struct_(a)  # replace `uts` with an abbreviated version
    uni, idx, cnts = np.unique(a, return_index=True,
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
    # -- basic return as ndarray used by common_segments
    if as_3d:  # The array cannot be basic if it is 3d
        as_basic = False
    if to_orig:
        tmp = self.XY + self.LL
        b_vals = [tmp[ft[0]:ft[1]] for ft in self.FT]   # shift to orig extent
    else:
        b_vals = self.bits
    # -- Do the concatenation
    fr_to = np.concatenate([np.concatenate((b[:-1], b[1:]), axis=1)
                            for b in b_vals], axis=0)
    # -- return if simple and not 3d representation
    if as_basic:
        return fr_to
    # -- return 3d from-to representation
    if as_3d:
        fr_to = fr_to[:, :4]
        s0, s1 = fr_to.shape
        return fr_to.reshape(s0, s1 // 2, s1 // 2)
    # -- structured array section
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
    """Simplify array. Requires, `_angles_3pnt_` from npg_geom_hlp."""
    ang = _angles_3pnt_(a, inside=True, in_deg=True)
    idx = np.abs(ang - 180.) >= deviation
    sub = a[1: -1]
    p = sub[idx]
    return a, p, ang


# ---- ---------------------------
# ---- (9) pnts in, or on, geometries
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
        If `just_common` is True, only the points in both data sets are
        returned.  If False, then the common and unique points are returned as
        two separate arrays.
        If one of the two arrays is empty `None` will be returned for that
        array.

    See Also
    --------
    `npg_pip.pnts_in_Geo` for Geo arrays explicitly.
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


def pnts_on_poly(pnts, poly):
    """Run multiple `_closest_pnt_on_poly_`."""
    result = []
    for p in pnts:
        result.append(_closest_pnt_on_poly_(p, poly))
    result = np.asarray(result)
    dt = [('X0', '<f8'), ('Y0', '<f8'), ('X1', '<f8'), ('Y1', '<f8'),
          ('Dist', '<f8'), ('Angle', '<f8')]
    z = np.zeros((len(result),), dtype=dt)
    names = z.dtype.names
    for i, n in enumerate(names):
        z[n] = result[:, i]
    return z


# ---- ---------------------------
# ---- Not included yet --------
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
        mn_ = pnts[:, 0].min()
        mx_ = pnts[:, 0].max()
        x_bins = np.arange(mn_, mx_, (mx_ - mn_) / 10.)
    if y_bins is None:
        mn_ = pnts[:, 1].min()
        mx_ = pnts[:, 1].max()
        y_bins = np.arange(mn_, mx_, (mx_ - mn_) / 10.)
    h = np.histogram2d(pnts[:, 0], pnts[:, 1], [x_bins, y_bins])
    return h


def in_hole_check(pnts, geo):
    """Check if points are in a hole."""
    w = np.where(geo.CL == 0)[0]
    holes = [geo.bits[i] for i in w]
    out = []
    for h in holes:
        inside = np_wn(pnts, h)  # crossing_num(pnts, h, False)
        if inside.size > 0:
            out.append([h, inside])
    return out


def which_quad(line):
    """Return the quadrant a vector lies in.

    >>> q = _quad_(line)
    >>> if q in [2, 3]:
    >>>     line = line[::-1]
    """
    x_, y_ = np.sign(np.diff(line[[0, -1]], axis=0))[0]
    # right
    if x_ >= 0:
        if y_ >= 0:  # upper
            return 1
        return 4
    # left
    if y_ >= 0:  # upper
        return 2
    return 3


# ---- ---------------------------
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
