# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
------------
npg_maths
-----------

** Math helpers for use with Geo arrays or NumPy ndarrays.

----

Script :
    npg_maths.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-03-11

Purpose
-------
General purpose functions for Geo arrays or NumPy ndarrays.

References
----------
`Paul Bourke website
<https://paulbourke.net/geometry/circlesphere/>`_.

Notes
-----
Notes for the `npg_maths`

"""
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0401,E1101,E1121
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0621
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915

import sys
import numpy as np

import npg  # noqa

script = sys.argv[0]

__all__ = [
    'cross_product_2d',
    'n_largest',                       # (3) size-based functions
    'n_smallest',
    'running_count',
    'pnt_to_array_distances',
    'project_pnt_to_line',             # ( ) line methods
    'segment_crossing'
]

__helpers__ = [
    '_angles_3pnt_',
    '_angle_between_',
    '_pnt_on_segment_'
]

__imports__ = []


# ---- ---------------------------
# ---- (1) private helpers
#
def _arc_mini_(p_st, cent, p_en, radius=1., step=5.0, outside=True):
    """Create arc from a mini circle.  A circle center of 0,0 is used."""
    #
    # -- create a basic circle assuming center at (0,0)
    angles = np.deg2rad(np.arange(180.0, -180.0 - step, -step))
    ang = np.degrees(angles)
    x_s = radius * np.cos(angles)  # + xc  # X and Y values
    y_s = radius * np.sin(angles)  # + yc  # add the circle center later
    circ_pnts = np.array([x_s, y_s]).T
    # --
    d0 = p_st - cent
    d1 = p_en - cent
    start = np.atan2(*d0[::-1])  # np.degrees(
    stop = np.atan2(*d1[::-1])   # np.degrees(
    start_deg = np.degrees(start)
    stop_deg = np.degrees(stop)
    # signs = [i >= 0 for i in [start_deg, stop_deg]]
    # inner_angle = np.degrees(_angle_between_(p_st, cent, p_en))
    # outer_angle = np.degrees(_angle_between_(p_en, cent, p_st))
    #
    if start_deg < stop_deg:
        if start_deg < 0 and stop_deg >= 0:
            w0 = np.nonzero(ang <= start_deg)[0]
            w1 = np.nonzero(ang >= stop_deg)[0]
            ids = np.concatenate((w0, w1))
        else:
            ids = np.logical_and(ang >= start_deg, ang <= stop_deg)
    elif start_deg > stop_deg:
        if start_deg >= 0:
            ids = np.logical_and(ang <= start_deg, ang >= stop_deg)
        else:  # both negative ?
            ids = np.logical_and(ang <= start_deg, ang >= stop_deg)
    #
    new_pnts = circ_pnts[ids] + cent
    return new_pnts


def _angles_3pnt_(a, inside=True, in_deg=True):
    """Worker for Geo `polygon_angles`, `geom_angles` and `min_area_rect`.

    Sequential points, a, b, c for the first bit in a shape, so interior holes
    are removed in polygons and the first part of a multipart shape is used.
    Use multipart_to_singlepart if you want to  process that type.

    Parameters
    ----------
    inside : boolean
        True, for interior angles for polygons ordered clockwise.  Equivalent
        to `right-side` for polylines.
    in_deg : boolean
        True for degrees, False for radians.

    See Also
    --------
    For a simple angle between 3 consecutive points see `_angle_between_`.
    Compare the two::

      a = np.array([[2.,0], [0., 0], [10., 10.]])  # clockwise, forms a 45 deg
      _angles_3pnt_(a, inside=True, in_deg=True)
      array([ 128.66,  45.00,   6.34])  # these sum to 180
      # or, change the angle base
      b = np.array([[10., 10.], [10.,0], [0., 0]])
      _angles_3pnt_(b, inside=True, in_deg=True)
      array([ 45.00,  90.00,  45.00])
      # the middle angle is the one, versus
      p0, cent, p1 = a
      _angle_between_(p0, cent, p1, in_degrees=True)  # -> 45.0

    Notes
    -----
    Sum of interior angles of a polygon with `n` edges::

        (n − 2)π radians or (n − 2) × 180 degrees
        n = number of unique vertices

    | euler`s formula
    | number of faces + number of vertices - number of edges = 2
    | rectangle : 1 + 5 - 4 = 2
    | triangle  : 1 + 4 - 3 = 2
    """
    # if np.allclose(a[0], a[-1]):                 # closed loop, remove dupl.
    #     a = a[1:]  # a[:-1] 2024-10-20 changes to get 1st angle correct
    # ba = a - np.concatenate((a[-1][None, :], a[:-1]), axis=0)
    # bc = a - np.concatenate((a[1:], a[0][None, :]), axis=0)
    # cr = cross_product_2d(ba, bc)
    #
    if np.allclose(a[0], a[-1]):                 # closed loop, remove dupl.
        a = a[:-1]
    ba = a - np.concatenate((a[-1][None, :], a[:-1]), axis=0)
    bc = a - np.concatenate((a[1:], a[0][None, :]), axis=0)
    cr = cross_product_2d(ba, bc)
    dt = np.einsum('ij,ij->i', ba, bc)
    ang = np.arctan2(cr, dt)
    TwoPI = np.pi * 2.
    if inside:
        angles = np.where(ang < 0., ang + TwoPI, ang)
    else:
        angles = np.where(ang > 0., TwoPI - ang, ang)
    if in_deg:
        angles = np.degrees(angles)
    return angles


def _angle_between_(p0, cent, p1, inside=True, in_degrees=False):
    """Return angle between two vectors emminating from the center.

    Parameter
    ---------
    p0, cent, p1 : array_like
        These are two point coordinates forming an angle p0 -> cent -> p1

    Notes
    -----
    The vectors are created from the three points, cent -> p0 and cent -> p1
    This works::

        a = np.array([[2.,0], [0., 0], [10., 10.]])
        p0, cent, p1 = a
        _angle_between_(p0, cent, p1, in_degrees=False)
        0.7853981633974485
        _angle_between_(p0, cent, p1, in_degrees=True)
        45.000000000000014

    Note:  use np.degrees(angle).item() to get a python number.
    """
    v0 = np.array(p0) - np.array(cent)  # center to p0
    v1 = np.array(p1) - np.array(cent)  # center to p1
    ang = np.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    TwoPI = np.pi * 2.
    if inside:
        angles = np.where(ang < 0., ang + TwoPI, ang)
    else:
        angles = np.where(ang > 0., TwoPI - ang, ang)
    if in_degrees:
        angles = np.degrees(angles)
    return angles


# ---- ---------------------------
# ---- (2) geom private helpers

def _resize_segment_(a, absolute=True, value=1, direction=0, keep_all=True):
    """Return a line segment scaled by a finite distance in a chosen direction.

    Parameters
    ----------
    a : array_like
        Two point array with shape (2, 2) representing the segment start and
        end coordinates.
    absolute : Boolean
        True uses the `value` to scale the segment by the amount.
        False will use `value` as a percentage in the scaling.
    value : number
        A number representing an absolute distance or a percentage of the
        segment length depending upon `absolute`.
    direction : number
        A choice from (-1, 0, 1) specifying which end(s) are lengthened
          -  -1 the start end of the segment
          -   0 both ends
          -   1 the destination end
    keep_all :  boolean
        True, adds the extension points to the input array in their correct
        position.  False, yields a new segment consisting of just the extended
        points.

    Examples
    --------
    scale with all options::

       a = np.array([[1., 1.], [5., 4]])
       r = [_scale_segment_(a, distance=0.25, direction=i, keep_all=True)
            for i in [-1, 0, 1]]
    scale all segments::

       aoi.T
       array([[ 0.0, 0.0, 10.0, 10.0, 0.0], [ 0.0, 10.0, 10.0, 0.0, 0.0]])
       pairs = np.concatenate((aoi[:-1], aoi[1:]), axis=1).reshape(-1, 2, 2)
       r = [_resize_segment_(i, absolute=True, value=1., direction=0,
                             keep_all=True)
            for i in pairs]
       segs = [np.concatenate((i[:-1], i[1:]), axis=1) for i in r]
       plot_segments(segs)
    """
    if direction not in [-1, 0, 1]:
        print("\nRead the script header.\n")
        return None
    # --
    dxy = a[1:] - a[:-1]
    leng = np.sqrt(np.einsum('ij,ij->i', dxy, dxy)).item()  # just get the val
    distance = value if absolute else leng * value/100.
    st_ = a[0] - (dxy[0] / leng) * distance
    en_ = a[1] + (dxy[0] / leng) * distance
    # --
    # ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if direction == 0:
        arr = [st_, a[0], a[1], en_] if keep_all else [st_, en_]
    elif direction == 1:
        arr = [a[0], a[1], en_] if keep_all else [a[0], en_]
    elif direction == -1:
        arr = [st_, a[0], a[1]] if keep_all else [st_, a[1]]
    return np.array(arr)


# ---- ---------------------------
# ---- (3) private helpers
def cross_product_2d(a, b):
    """Replace `np.cross` with this for 2D vectors (deprecated in NumPy 2.0).

    a, b : array_like
        2D vectors
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def dot_product_2d(a, b):
    """Return the dot product for two 2D arrays using einsum.

    Parameters
    ----------
    a, b : array_like
        The 2D arrays.

    Example
    -------
    A square `a` and a rotated square `b`
    a =                         b =
    array([[  0.00,   0.00],    array([[  0.00,   5.00],
           [  0.00,  10.00],           [  5.00,  10.00],
           [ 10.00,  10.00],           [ 10.00,   5.00],
           [ 10.00,   0.00],           [  5.00,   0.00],
           [  0.00,   0.00]])          [  0.00,   5.00]])

    dot_product(a, b)  -> array([  0.00,  100.00,  150.00,  50.00,   0.00])
    """
    return np.einsum('ij,ij->i', a, b)


def norm_2d(a, b):
    """Return.  Fill in the information.

    A norm is the distance from the origin to a point. In effect, the first
    point is taken as the origin

    https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Notes
    -----
    >>> np.linalg.vector_norm(aoi, axis=1)
    ... array([  0.00,  10.00,  14.14,  10.00,   0.00])

    >>> np.linalg.norm(aoi, axis=1)
    ... array([  0.00,  10.00,  14.14,  10.00,   0.00])

    # Euclidean Distance Caculator
    def dist(a, b, axis=1):
        return np.linalg.norm(a - b, axis=ax)

    >>> dist(a, b, axis=1)  -> array([  5.00,   5.00,   5.00,   5.00,   5.00])
    >>> dist(a, b, axis=0)  -> array([  7.07,   8.66])
    ... # --
    ... # Shapes must be the same, or broadcastable
    >>> dist(a, b[:3][:, None], axis=0)
    ... array([[ 11.18,  12.25],
    ...        [ 11.18,   7.07],
    ...        [ 11.18,   7.07],
    ...        [ 11.18,  12.25],
    ...        [ 11.18,  12.25]])
    >>> dist(a, b[:3][:, None], axis=1)
    ... array([[ 14.14,  11.18],
    ...        [ 11.18,  17.32],
    ...        [ 17.32,  11.18]])
    """
    return None


def pnt_to_array_distances(a, p):
    """Array points, `a`,  to point `p`, distance(s).

    Use to determine distance of a point to a segment or segments.  No error
    checking, so make sure `a` is a set of Nx2 points and p is a single point.
    """
    if hasattr(a, 'IFT'):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        a = np.array(a, ndmin=2)  # ensure a 2d array of coordinates
    diff = a - p[None, :]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def project_pnt_to_line(x0, y0, x1, y1, xp, yp):
    """Project a point on to a line to get the perpendicular location."""
    x01 = x1 - x0
    y01 = y1 - y0
    dotp = x01 * (xp - x0) + y01 * (yp - y0)
    dot01 = x01 * x01 + y01 * y01
    if dot01:
        coeff = dotp / dot01
        lx = x0 + x01 * coeff
        ly = y0 + y01 * coeff
        return lx, ly
    return None


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

    np.cross for 2D arrays was deprecated in NumPy 2.0 use

    def cross2d(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    >>> d = np.linalg.norm(np.cross(p1 - p0, p0 - p))/np.linalg.norm(p1 - p0)
    >>> # becomes
    >>> d = np.linalg.norm(cross2d(p1 - p0, p0 - p))/np.linalg.norm(p1 - p0)
    """
    x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
    dist_ = dx * dx + dy * dy  # squared length
    u = ((x0 - x1) * dx + (y0 - y1) * dy) / dist_
    u = max(min(u, 1), 0)
    xy = np.array([dx, dy]) * u + [x1, y1]
    d = xy - pnt
    return xy, np.hypot(d[0], d[1])


# ---- (2) shape-based intersections
#
def circ_circ_intersection(c0, r0, c1, r1, return_arcs=False, step=1):
    """Return circle-circle intersection points.

    Parameters
    ----------
    c0, c1 : array_like
        Circle centers.
    r0, r1 : numbers
        Radius of each circle
    return_arcs : boolean
        True, to return the intersection points as well as the arc points.
    step : number
        Used to create the arc spacing, in degrees, if arcs are being returned.

    References
    ----------
    `Paul Bourke website
    <https://paulbourke.net/geometry/circlesphere/>` _.

    Example
    -------
    centers = [1., 12.], [4., 13], [5., 15.]
    radius = 2.
    c0 = [1., 12.]; r0 = 2.
    c1 = [4., 13.]; r1 = 2.
    result = circ_circ_intersection(c0, r0, c1, r1, return_arcs=True, step=10)
    c0c1 = np.array([c0, c1])
    if result:  # not an empty list
        result += [c0c1]  # put c0c1 into a list
        plot_polylines(result)
    """
    c0 = np.array(c0)
    c1 = np.array(c1)
    x0, y0 = c0
    x1, y1 = c1
    d = ((x1 - x0)**2 + (y1 - y0)**2)**0.5  # -- x**0.5 = np.sqrt(x)
    if d > r0 + r1:  # distance greater than combined radius, non-intersecting
        return []
    if d < abs(r0 - r1):  # one circle within other
        return []
    if d == 0 and r0 == r1:  # coincident circles
        return []
    a = (r0**2 - r1**2 + d**2) / (2.0 * d)
    h = (r0**2 - a**2)**0.5  # -- equal to np.sqrt(r0**2 - a**2)
    dx, dy = c1 - c0
    x2 = x0 + a * dx / d  # x2,y2 is the chord midpoint
    y2 = y0 + a * dy / d  # x2,y2 = c0 + a*(c1 - c0)/d
    # -- see image in reference
    h_dx_d = h * dx / d
    h_dy_d = h * dy / d
    x3 = x2 + h_dy_d  # h * dy / d  # x3, x4 are the intersection pnts
    y3 = y2 - h_dx_d  # h * dx / d  # mirrored along the chord at the
    x4 = x2 - h_dy_d  # h * dy / d  # intersection points
    y4 = y2 + h_dx_d  # h * dx / d
    x_pnts = np.array([[x3, y3], [x4, y4]])
    if return_arcs:
        if x_pnts.size == 4:
            p_st, p_en = x_pnts
            arc0 = _arc_mini_(p_st, c0, p_en, radius=r0, step=step)
            arc1 = _arc_mini_(p_en, c1, p_st, radius=r1, step=step)
            arc0 = np.concatenate(
                (x_pnts[-1][None, :], arc0, x_pnts[0][None, :]), axis=0)
            arc1 = np.concatenate(
                (x_pnts[0][None, :], arc1, x_pnts[-1][None, :]), axis=0)
            return [x_pnts, arc0, arc1]
    return x_pnts


def line_circ_intersection(c_cent, st_pnt, en_pnt, radius=1):
    """Return the intersection points between a line and a circle.

    Parameters
    ----------
    c_cent, st_pnt, en_pnt : array_like
        The circle center (cx,cy) and the segment start (x0,y0) and end points
        (x1,y1).
    radius : number
        Circle radius.

    """
    cx, cy = c_cent
    r = radius
    x0, y0 = st_pnt
    x1, y1 = en_pnt

    cx = x0 - cx  # reorient to center
    cy = y0 - cy
    dx = x1 - x0
    dy = y1 - y0
    a = dx * dx + dy * dy
    b = 2 * (dx * cx + dy * cy)
    c = cx * cx + cy * cy - r * r
    det = b * b - (4 * a * c)

    if abs(a) <= 1.0e-6 or det < 0:
        return None  # No real solutions.
    elif det == 0:   # Line is tangent to the circle
        t = -b / (2 * a)
        p0_x = x0 + t * dx
        p0_y = y0 + t * dy
        return [st_pnt, [p0_x, p0_y], en_pnt]
    else:             # Line intersects circle
        det = det**0.5  # sqrt(det)
        t1 = (-b - det) / (2 * a)
        t2 = (-b + det) / (2 * a)
        # First point is closest to [x0, y0]
        p0_x = x0 + t1 * dx
        p0_y = y0 + t1 * dy
        #
        p1_x = x0 + t2 * dx
        p1_y = y0 + t2 * dy
        return [st_pnt, [p0_x, p1_y], [p1_x, p1_y], en_pnt]


def segment_crossing(args):
    """Return a two line intersection point from 4 points.

    Parameters
    ----------
    args : array_like
        The input shape needs to be one of (8,), (2, 4) or (2, 2, 2).

    Notes
    -----
    `c` is the denominator, and `a` and `b` are ua and ub in Paul Bourkes

    See Also
    --------
    This is a duplicate, for now, of `npg_buffer._x_ings_`
    Returns
    -------
    The intersection point if two segments cross, or an extrapolation to a
    point where they would meet.
    """
    #
    if len(args) == 8:
        x0, y0, x1, y1, x2, y2, x3, y3 = args
    elif np.prod(args.shape) == 8:
        args = np.ravel(args)
    else:
        print("\n{}".format(segment_crossing.__doc__))
        return None
    #
    x0, y0, x1, y1, x2, y2, x3, y3 = args
    dx_10, dy_10 = x1 - x0, y1 - y0
    dx_32, dy_32 = x3 - x2, y3 - y2
    #
    a = x0 * y1 - x1 * y0  # 2d cross product `cross2d`
    b = x2 * y3 - x3 * y2
    c = dy_10 * dx_32 - dy_32 * dx_10  # (y1-y0)*(x3-x2) - (y3-y2)*(x1-x0)
    if abs(c) > 1e-12:  # -- return the intersection point
        n1 = (a * dx_32 - b * dx_10) / c
        n2 = (a * dy_32 - b * dy_10) / c
        return (n1, n2)
    # -- if no intersection exists, return the end of the first line
    return None


# ---- (3) counts or size-based .... n largest, n_smallest
#
def n_largest(a, num=1, col_sort=True):
    """Return the`num` largest entries in an array.

    The results are either:
        - by row, sorted by column
        - by column, sorted by row

    Parameters
    ----------
    a : ndarray
        Array dimensions <=3 supported
    num : integer
        The number of elements to return
    col_sort : boolean
        True to determine by column.
    """
    assert a.ndim <= 2, "Only arrays with ndim <=2 supported"
    num = min(num, a.shape[-1])
    _axis_ = 0 if col_sort else 1
    if a.ndim == 1:
        b = np.sort(a)[-num:]
    elif a.ndim == 2:
        b = np.sort(a, axis=_axis_)[-num:]
    else:
        return None
    return b


def n_smallest(a, num=1, col_sort=True):
    """Return the `num` smallest entries in an array.

    see `n_largest` for parameter description
    """
    assert a.ndim <= 3, "Only arrays with ndim <=3 supported"
    num = min(num, a.shape[-1])
    _axis_ = 0 if col_sort else 1
    if a.ndim == 1:
        b = np.sort(a)[:num]
    elif a.ndim == 2:
        b = np.sort(a, axis=_axis_)[:num]
    else:
        return None
    return b


def running_count(a, to_label=False):
    """Perform a running count on a 1D array.

    The order number of the value in the sequence is returned.

    Parameters
    ----------
    a : array
        1D array of values, int, float or string
    to_label : boolean
        Return the output as a concatenated string of value-sequence numbers if
        True, or if False, return a structured array with a specified dtype.

    Examples
    --------
    >>> a = np.random.randint(1, 10, 10)
    >>> #  [3, 5, 7, 5, 9, 2, 2, 2, 6, 4] #
    >>> running_count(a, False)
    array([(3, 1), (5, 1), (7, 1), (5, 2), (9, 1), (2, 1), (2, 2),
           (2, 3), (6, 1), (4, 1)],
          dtype=[('Value', '<i4'), ('Count', '<i4')])
    >>> running_count(a, True)
    array(['3_001', '5_001', '7_001', '5_002', '9_001', '2_001', '2_002',
           '2_003', '6_001', '4_001'],
          dtype='<U5')

    >>> b = np.array(list("zabcaabbdedbz"))
    >>> #  ['z', 'a', 'b', 'c', 'a', 'a', 'b', 'b', 'd', 'e', 'd','b', 'z'] #
    >>> running_count(b, False)
    array([('z', 1), ('a', 1), ('b', 1), ('c', 1), ('a', 2), ('a', 3),
           ('b', 2), ('b', 3), ('d', 1), ('e', 1), ('d', 2), ('b', 4),
           ('z', 2)], dtype=[('Value', '<U1'), ('Count', '<i4')])
    >>> running_count(b, True)
    array(['z_001', 'a_001', 'b_001', 'c_001', 'a_002', 'a_003', 'b_002',
           'b_003', 'd_001', 'e_001', 'd_002', 'b_004', 'z_002'], dtype='<U5')
    """
    dt = [('Value', a.dtype.str), ('Count', '<i4')]
    N = a.shape[0]  # used for padding
    z = np.zeros((N,), dtype=dt)
    idx = a.argsort(kind='mergesort')
    s_a = a[idx]
    neq = np.where(s_a[1:] != s_a[:-1])[0] + 1
    run = np.ones(a.shape, int)
    run[neq[0]] -= neq[0]
    run[neq[1:]] -= np.diff(neq)
    out = np.empty_like(run)
    out[idx] = run.cumsum()
    z['Value'] = a
    z['Count'] = out
    if to_label:
        pad = int(round(np.log10(N)))
        z = np.array(["{}_{:0>{}}".format(*i, pad) for i in list(zip(a, out))])
    return z


# ---- (4)To incorporate
def _point_along_a_line(x0, y0, x1, y1, d):
    """
    Return the point on the line connecting (*x0*, *y0*) -- (*x1*, *y1*) whose
    distance from (*x0*, *y0*) is *d*.
    line 3155 in
    `<https://github.com/matplotlib/matplotlib/blob/v3.10.1/lib/matplotlib/
    patches.py#L1962`_.
    """
    dx, dy = x0 - x1, y0 - y1
    ff = d / (dx * dx + dy * dy) ** .5
    x2, y2 = x0 - ff * dx, y0 - ff * dy

    return x2, y2

# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # optional controls here
