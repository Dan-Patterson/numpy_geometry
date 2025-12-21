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
    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-12-09

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
    'flip_left_right',                 # (2) geom helpers
    'flip_up_down',
    'cross_product_2d',                # (3) math helpers
    'dot_product_2d',
    'norm_2d',
    'pnt_to_array_distances',
    'project_pnt_to_line',
    'circ_circ_intersection',          # (4) shape-based intersectons
    'line_circ_intersection',
    'segment_crossing',
    'n_largest',                       # (5) counts or size-based functions
    'n_smallest',
    'running_count',
]

__helpers__ = [
    '_area_centroid_2',                 # (1) geom private helpers
    'trans_rot_2',
    '_arc_mini_',
    '_angles_3pnt_',
    '_angle_between_',
    '_offset_segment_',
    '_resize_segment_',
    '_pnt_on_segment_',
    '_point_along_a_line'
]

__imports__ = []


# ---- ---------------------------
# ---- (1) geom private helpers
#
rot90 = np.array([[0, -1], [1, 0]], dtype='float')
rot180 = np.array([[-1, 0], [0, -1]], dtype='float')
rot270 =  np.array([[0, 1], [-1, 0]], dtype='float')  #noqa
# -- rotations by angle
# c, s = np.cos(angle), np.sin(angle)
# rot = np.array(((c, s), (-s, c)))


def _area_centroid_2(a):
    r"""Calculate area and centroid for a singlepart polygon, `a`.

    This is also used to calculate area and centroid for a Geo array's parts.

    Notes
    -----
    **See npg.npg_geom_hlp  ... _area_centroid_ ** for the main
    For multipart shapes, just use this syntax:

    >>> # rectangle with hole
    >>> a0 = np.array([[[0., 0.], [0., 10.], [10., 10.], [10., 0.], [0., 0.]],
                      [[2., 2.], [8., 2.], [8., 8.], [2., 8.], [2., 2.]]])
    >>> [_area_centroid_2(i) for i in a0]
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


def _trans_rot_2(a, angle=0.0, clockwise=False):
    """Rotate shapes about their center or individually."""
    if clockwise:
        angle = -angle
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s, c)))
    # cent = np.mean(np.unique(a, axis=0), axis=0)
    area_, cent = _area_centroid_2(a)
    return np.einsum('ij,jk->ik', a - cent, R) + cent


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
    #
    if np.allclose(a[0], a[-1]):                 # closed loop, remove dupl.
        a = a[:-1]
    ba = a - np.concatenate((a[-1][None, :], a[:-1]), axis=0)  # centre - start
    bc = a - np.concatenate((a[1:], a[0][None, :]), axis=0)  # centre - end
    cr = cross_product_2d(ba, bc)       # -- use cross_product_2d for `cross`
    dt = np.einsum('ij,ij->i', ba, bc)  #
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
    determinant : for 2D array [[a, b], [c, d]] == sum(a*d)
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


def _offset_segment_(poly, value=1.):
    """Return offset polygon segments by a finite value.

    Notes
    -----
    This can be combined with `resizing` using the following syntax::

        pairs = offsets.reshape(-1, 2, 2)  # the results from this function
        r = [_resize_segment_(
                i, absolute=True,
                value=2.,
                direction=0,
                keep_all=True)
             for i in pairs
             ]
        segs = [np.concatenate((i[:-1], i[1:]), axis=1) for i in r]
        # plot_segments(segs)  optional function

    You can resize after offsetting or visa versa
    """
    dxdy = poly[1:] - poly[:-1]
    r = value / np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))
    rr = np.concatenate((r[:, None], -r[:, None]), axis=1)
    dx_dy = dxdy * rr
    dy_dx = dx_dy[:, [1, 0]]  # -- swap order yielding dy, dx
    pnt0 = poly[:-1] + dy_dx     # new start and end points for offset segments
    pnt1 = poly[1:] + dy_dx      #
    # -- offset segments
    offsets = np.concatenate((pnt0, pnt1), axis=1)
    return offsets


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


def _pnt_on_segment_(pnt, seg, full=True):
    """Orthogonal projection of a point onto a 2 point line segment.

    Returns the intersection point, if the point is between the segment end
    points, otherwise, it returns the distance to the closest endpoint.

    Parameters
    ----------
    pnt : array-like
        `x,y` coordinate pair as list or ndarray
    seg : array-like
        `from-to points`, of x,y coordinates as an ndarray or equivalent.
    full : boolean
        True returns the location and the distance.  False, just the location.
    Notes
    -----
    >>> seg = np.array([[0, 0], [10, 10]])  # p0, p1
    >>> p = [10, 0]
    >>> _pnt_on_segment_(p, seg)
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
    if full:
        d = xy - pnt
        return xy, np.hypot(d[0], d[1])
    else:
        return xy


# ---- ---------------------------
# ---- (2) geom helpers
# !!! these two don't work for Nx2 array
def flip_left_right(a, shift_back=True):
    """Return an array flipped left to right.

    Parameters
    ----------
    a :  array_like
    shift_back : boolean
        True, returns the array to the original x-axis baseline.  False, flips
        along a line middle of the new y-values.
    """
    m = np.array([[-1, 0], [0, -1]], dtype='float')  # rotate 180
    vals = a @ m
    if shift_back:
        mins_ = np.min(vals, axis=1)
        vals = vals - mins_
    return vals


def flip_up_down(a, shift_back=True):
    """Return an array flipped vertically.

    Parameters
    ----------
    a :  array_like
    shift_back : boolean
        True, returns the array to the original x-axis baseline.  False, flips
        along a line middle of the new y-values.
    """
    m = np.array([[1, 0], [0, -1]], dtype='float')
    vals = a @ m
    if shift_back:
        mins_ = np.min(vals, axis=0)
        vals[:, 1] -= mins_[1]
        # print("not implemented")
    return vals


# ---- ---------------------------
# ---- (3) math helpers

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

    # Euclidean Distance Calculator
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


# ---- (4) shape-based intersections
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
    def _e_2d_(a, p):
        """Return arc,`a` start-end point distance to point `p`."""
        diff = a - p[None, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    def _fix_(arc, p_st, en):
        """Check arc st end pnts."""
        _d0, _d1 = _e_2d_(arc[[0, -1]], p_st)
        if _d0 < _d1:  # or to_ to  np.array([p_en, p_st]) for polygons
            fr_, to_ = p_st[None, :], p_en[None, ]  # see above
        else:
            to_, fr_ = p_st[None, :], p_en[None, ]
        return np.concatenate((fr_, arc, to_), axis=0)
    #
    x0, y0 = c0 = np.array(c0)
    x1, y1 = c1 = np.array(c1)
    dx, dy = c1 - c0
    d = (dx**2 + dy**2)**0.5  # -- centre-centre distance x**0.5 = np.sqrt(x)
    # -- cases
    if d > r0 + r1:           # non-intersecting, `d` > combined radii
        return []
    if d < abs(r0 - r1):      # one circle within other
        return []
    if d == 0 and r0 == r1:   # coincident circles
        return []
    #
    # -- intersections exist -- see image in reference  --
    #    x2 = x0 + a * dx / d   and    y2 = y0 + a * dy / d
    #    h_dx_d = h * dx / d    and    h_dy_d = h * dy / d
    #
    a = (r0**2 - r1**2 + d**2) / (2.0 * d)
    h = (r0**2 - a**2)**0.5  # -- equal to np.sqrt(r0**2 - a**2)
    #
    x2, y2 = c0 + (c1 - c0) * a/d     # the chord midpoint
    h_dx_d, h_dy_d = (c1 - c0) * h/d
    #   the intersection pnts mirrored along the chord
    x3 = x2 + h_dy_d
    y3 = y2 - h_dx_d
    x4 = x2 - h_dy_d
    y4 = y2 + h_dx_d
    x_pnts = np.array([[x3, y3], [x4, y4]])
    # --
    if return_arcs and x_pnts.size == 4:
        p_st, p_en = x_pnts
        arc0 = _arc_mini_(p_st, c0, p_en, radius=r0, step=step)
        arc1 = _arc_mini_(p_en, c1, p_st, radius=r1, step=step)
        # -- fix arcs
        arc0 = _fix_(arc0, p_st, p_en)
        arc1 = _fix_(arc1, p_st, p_en)
        #
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
    #
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


# ---- (5) counts or size-based .... n largest, n_smallest
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
#
def _point_along_a_line(x0, y0, x1, y1, d):
    """Return a point on a line.

    The point on the line connects (*x0*, *y0*) -- (*x1*, *y1*) with a
    distance of *d* from (*x0*, *y0*).
    line 3155 in
    `<https://github.com/matplotlib/matplotlib/blob/v3.10.1/lib/matplotlib/
    patches.py#L1962`_.
    """
    dx, dy = x0 - x1, y0 - y1
    ff = d / (dx * dx + dy * dy)**0.5
    x2, y2 = x0 - ff * dx, y0 - ff * dy
    return x2, y2


# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # optional controls here
