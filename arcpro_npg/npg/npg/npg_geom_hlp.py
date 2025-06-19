# -*- coding: utf-8 -*-
# noqa: D205, D208, D400, F403
r"""
-----------
npg_geom_hlp
-----------

**General helper functions**

----

Script :
    npg_geom_hlp.py

Author :
    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-06-01

Purpose
-------
Helper functions for Geo arrays and used by npg_geom_ops.py.

Notes
-----
To suppress runtime warnings for errors that you know will happen.

`<https://stackoverflow.com/questions/29950557/ignore-divide-by-0-warning-
in-numpy>`_.

Generally:  np.seterr(divide='ignore', invalid='ignore')

For a section::

    with np.errstate(divide='ignore', invalid='ignore'):
        # some code here

References
----------
`Compare_geometry to check for identical rows
<https://stackoverflow.com/questions/51352527/check-for-identical-rows-in-
different-numpy-arrays>`_.

`Dealing with Duplicates blog in 2023
<https://community.esri.com/t5/python-blog/dealing-with-duplicates/ba-p/
1258351>`_.

Extras
------
**Generating `__all__` and `__helpers__`**

>>> not_in = [
...     '__all__', '__builtins__', '__cached__', '__doc__', '__file__',
...     '__loader__', '__name__', '__package__', '__spec__', 'np', 'npg',
...     'sys', 'script'
...     ] + __imports__

>>> __all__ = [i for i in dir(npg.npg_helpers)
...            if i[0] != "_" and i not in not_in]

>>> __helpers__ = [i for i in dir(npg.npg_helpers)
...                if i[0] == "_" and i not in not_in]

"""

# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121


import sys
# from textwrap import dedent

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

from npg.npg_prn import prn_tbl  # used by `shape_finder`

script = sys.argv[0]  # print this should you need to locate the script

nums = 'efdgFDGbBhHiIlLqQpP'

# -- See script header
#
# ---- __all__, __helpers__
#
__imports__ = ['uts', 'prn_tbl']

__all__ = [
    'geom_angles',                     # (4) Geo / ndarray stuff
    'segment_angles',
    'a_eq_b',                          # (5) compare, remove, keep geometry
    'close_pnts',
    'classify_pnts',
    'pnt_segment_info',
    'common_pnts',
    'compare_segments',
    'compare_geom',
    'keep_geom',
    'remove_geom',
    'del_seq_dups',
    'multi_check',
    'sort_xy',                         # (6) sort coordinates
    'swap_segment_pnts',
    'sort_segment_pairs',
    'dist_angle_sort',
    'radial_sort',
    'interweave',                      # (7) others functions
    'shape_finder',
    'coerce2array',
    'flat',
    'reclass_ids',
]

__helpers__ = [
    '_adj_within_',                    # (1) Geo Helpers
    '_base_',
    '_bit_check_',
    '_clean_segments_',
    '_e_2d_',
    '_from_to_pnts_',
    '_is_clockwise_',                  # (2) Condition checking
    '_is_ccw_',
    '_is_convex_',
    '_is_turn',
    '_in_extent_',
    '_in_LBRT_',
    '_pnts_in_extent_',
    '_angle_between_',                 # (3) Geometry helpers
    '_angles_from_north_',
    '_angles_from_xaxis_',
    '_od_angles_dist_',
    '_area_centroid_',
    '_bit_area_',
    '_bit_crossproduct_',
    '_bit_min_max_',
    '_bit_length_',
    '_bit_segment_angles_'
    '_rotate_',
    '_scale_',
    '_translate_',
    '_trans_rot_',
    '_perp_'
    # not classes
    '_iterate_'
]


#
# ---- ---------------------------
# ---- core bit functions
# ---- (1) Geo Helpers
#
def _adj_within_(a, full=False):
    """Determine adjacency within a Geo array's outer rings.

    Parameters
    ----------
    a : Geo array
    full : boolean
        True, returns a full report of the shapes being compared (f_shp, t_shp)
        and the input point ids.  The common points are returned from t_shp and
        their point and polygon id values.
        False, simply returns the adjacent polygon ids (f_shp, t_shp).

    Returns
    -------
    Two arrays, the adjacency information and the outer_rings.
    """
    out = []
    ids = a.shp_pnt_ids
    fr_to = a.FT
    f_shps = a.IDs
    arr = a.XY
    for i, f_t in enumerate(fr_to[:-1]):
        f_shp = f_shps[i]
        f, t = f_t
        w = (arr[f:t][:, None] == arr[t:]).any(0).all(-1)  # .tolist()
        if np.sum(w) > 0:
            pnt_shp = ids[t:][w]
            t_shp = np.unique(pnt_shp[:, 1])  # t_shp = pnt_shp[0][1]
            if full:
                common = arr[t:][w]
                out.append([[f_shp] + t_shp.tolist(), f_t, common, pnt_shp])
            else:
                out.append(np.asarray([f_shp] + t_shp.tolist()))
    return out


def _base_(a):
    """Return the base array of a Geo array.  Shave off microseconds."""
    if hasattr(a, "IFT"):
        return a.XY
    return a


def _bit_check_(a, just_outer=False):
    r"""Check for bits and convert if necessary.

    Parameters
    ----------
    a : array-like
        Either a Geo array or a list of lists.  Conversion to bits or outer
        rings as desired.
    just_outer : boolean
        True, removes holes from the geometry.

    """
    if hasattr(a, "IFT"):
        if just_outer:
            a = a.outer_rings(asGeo=False)
        else:
            a = a.bits
    elif isinstance(a, (list, tuple)):
        a = np.asarray(a, dtype='O')
    if len(a) == 1:
        a = [a]
    return a


def _clean_segments_(a, tol=1e-06):
    """Remove overlaps and extra points on poly* segments. In `npg_geom_hlp`.

    Parameters
    ----------
    a : array
        The input array or a bit from a Geo array.
    tol : float
        The tolerance for determining whether a point deviates from a line.

    Notes
    -----
    - Segments along a straight line can overlap (a construction error).
          [[0,0], [5, 5], [2, 2], [7, 7]]  # points out of order
    - Extraneous points can exist along a segment.
          [[0,0], [2, 2], [5, 5], [7, 7]]  # extra points not needed for line.
    """
    cr, ba, bc = _bit_crossproduct_(a, extras=True)
    # -- avoid duplicating the 1st point (0).
    whr = [i for i in np.nonzero(np.abs(cr) > tol)[0] if i != 0]
    vals = np.concatenate((a[0][None, :], a[whr], a[-1][None, :]), axis=0)
    return vals


def _e_2d_(a, p):
    """Return point to point distance for array (mini e_dist)."""
    diff = a - p[None, :]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def _from_to_pnts_(a, as_pairs=False):
    """Convert polygon/polyline shapes to from-to points.

    Parameters
    ----------
    a : array-like
        A Geo array bit or ndarray representing a singlepart shape.
    as_pairs : boolean
        True, returns the pairs as an Nx2x2 array. False, returns an Nx4 array.

    >>> [[[ X0, Y0],
          [ X1, Y1]], ...]        # True
    >>> [[X0, Y0, X1, Y1], ...]   # False

    See Also
    --------
    The Geo method `od_pairs` returns the proper traversal removing possible
    connections between inner and outer rings.
    """
    a = _base_(a)
    return np.concatenate((a[:-1], a[1:]), axis=1)


# ---- ---------------------------
# ---- (2) Condition checking
# ---- (a) angles, orientation, sidedness
def _is_clockwise_(a):
    """Return whether the sequence (polygon) is clockwise oriented or not."""
    return 1 if _bit_area_(a) > 0. else 0


def _is_ccw_(a):
    """Counterclockwise."""
    return 0 if _bit_area_(a) > 0. else 1


def _is_turn(pnts, tol=1e-6):
    """Return whether 3 points create a right, or left turn or straight.

    Parameters
    ----------
    pnts : points
        pnts can be a 3 point array with a shape of (3, 2)
    tol : float
        The tolerance to accommadate floating point errors in calculations.

    Returns
    -------
    -1 : right
     0 : straight
     1 : left

    Notes
    -----
    No error checking.  It is assumed you can provide an array of 3 points.
    Do any required conversion elsewhere.
    """
    pnts = np.array(pnts)
    if len(pnts) != 3:
        print("\nThree points required.")
        return None
    x0, y0, x1, y1, x2, y2 = pnts.ravel()
    v = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    v = 0 if abs(v) < tol else v
    # out = True if v < 0 else False
    out = int(np.sign(v))  # clever use of sign
    return out


def _is_convex_(a, is_closed=True):
    """Return whether a polygon is convex."""
    check = _bit_crossproduct_(a, is_closed)  # cross product
    return np.all(check >= 0)


# ---- (b) location, position, shape
#
def _in_extent_(pnts, extent):
    """Return points in, or on the line of an extent. See `_in_LBRT_` also.

    Parameters
    ----------
    pnts : array
        An Nx2 array representing point objects.
    extent : array-like
        A 2x2 array, as [[x0, y0], [x1, y1] where the first pair is the
        left-bottom and the second pair is the right-top coordinate.
    """
    if extent.shape[0] == 4:
        extent = extent.reshape(2, 2)
    LB, RT = extent
    comp = np.logical_and(LB <= pnts, pnts <= RT)  # using <= and <=
    idx = np.logical_and(comp[..., 0], comp[..., 1])
    return pnts[idx]


def _in_LBRT_(pnts, extent):
    """Return points in, or on the line of an extent.

    Parameters
    ----------
    pnts : array
        An Nx2 array representing point objects.
    extent : array-like
        A 1x4, as [x0, y0, x1, y1] where the first tw is the left-bottom
        and the second two are the right-top coordinate.

    See Also
    --------
    `np_helpers._in_extent_`
    """
    if hasattr(pnts, "IFT"):
        pnts = pnts.XY
    LB, RT = extent[:2], extent[2:]
    case_ = np.all(np.logical_and(LB < pnts, pnts <= RT), axis=1)
    if np.sum(case_) > 0:
        return True
    return False


def _pnts_in_extent_(pnts, extent=None, return_index=False):
    """Check, and return points within a defined extent.

    Parameters
    ----------
    pnts, extent : ndarray
        Nx2 arrays. `extent` needs a minimum shape of (2, 2).  If not,
        the left, bottom, top and right in that shape, then the extent is
        derived from the input points.
    return_index : boolean
        True, returns the indices in `pnts` that are inside the extent.
        False, returns whether at least one point is inside.

    >>> LB = np.min(a, axis=0)  # left bottom
    >>> RT = np.max(a, axis=0)  # right top
    >>> extent = np.asarray([LB, RT])
    """
    msg = "\nExtent in error... 2x2 array required not:\n{}\n"
    if extent is None:
        print(msg.format(extent))
        return None
    shp = np.asarray(extent).shape
    if shp == (2, 2):
        LB, RT = extent
    elif shp[0] == 4:
        extent = extent.reshape(2, 2)
        LB, RT = np.min(extent, axis=0), np.max(extent, axis=0)
    else:
        print(msg.format(extent))
        return None
    idx = np.all(np.logical_and(LB < pnts, pnts <= RT), axis=1)
    if return_index:
        if idx.base is not None:
            return idx.base
        return idx
    return np.all(idx)


# ---- ---------------------------
# ---- (3) geometry helpers
#
def _angle_between_(p0, cent, p1, in_degrees=False):
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
        angle_between(p0, cent, p1, in_degrees=False)
        0.7853981633974485
        angle_between(p0, cent, p1, in_degrees=True)
        45.000000000000014

    Note:  use np.degrees(angle).item() to get a python number.

    https://community.esri.com/t5/spatial-data-science
    -questions/how-to-identify-polygons-with-approximate-right/m-p/752828
        ba = p1 - p0     p1 is the center
        bc = p1 - p2
        cr = np.cross(ba, bc)
        dt = np.dot(ba, bc)
        ang = np.arctan2(np.linalg.norm(cr), dt)
    """
    v0 = np.array(p0) - np.array(cent)  # center to p0
    v1 = np.array(p1) - np.array(cent)  # center to p1
    angle = np.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    if in_degrees:
        angle = np.degrees(angle)
    return angle


def _angles_from_north_(angles):
    """Convert x-axis based angles to North-based, ergo clockwise.

    Parameters
    ----------
    angles : numbers, array-like
        x-axis based angles are counterclockwise ranging from -180 to 180 with
        0 E, 90 N, +/-180 W and -90 S

    See Also
    --------
    `_angles_from_xaxis_` for discussion.
    """
    return np.mod((450.0 - angles), 360.)


def _angles_from_xaxis_(angles):
    """Convert North referenced angles to x-axis based angles.

    Parameters
    ----------
    angles : numbers, array-like
        North-based angles, clockwise 0 to 360 degrees.

    Returns
    -------
    Angles relative to the x-axis (-180 - 180) which are counterclockwise.

    Example::

        >>> a = np.arange(180., -180 - 45, -45, dtype=np.float64)
        >>> a
        >>> array([ 180., 135., 90., 45., 0., -45., -90., -135., -180.])
        >>> b = (-a + 90.) % 360.
        >>> b
        >>> array([ 270., 315., 0., 45., 90., 135., 180., 225., 270.])
    """
    return np.mod(-angles + 90., 360.)


def _od_angles_dist_(arr, is_polygon=True):
    """Return origin-destination angles and distances.

    Parameters
    ----------
    is_polygon : boolean
        The first and last point are sliced off.  The first point is used
        as the origin and the last is a duplicate of the first.

    Notes
    -----
    The pnts array is rotated to the LL of the extent.  The first point is
    used as the origin and is sliced off of the remaining list.  If
    `is_polygon` is True, then the duplicated last point will be removed.
    """

    def _LL_(arr):
        """Return the closest point to the lower left of the array."""
        LL = np.min(arr, axis=0)
        idx = np.argmin(_e_2d_(arr, LL))
        return idx

    num = _LL_(arr)
    min_f = arr[num]
    arr_ordered = np.concatenate((arr[num:-1], arr[:num], [arr[num]]), axis=0)
    if is_polygon:
        arr = arr_ordered[1:-1]
    else:
        arr = arr_ordered[1:]
    dxdy = arr - min_f
    ang = np.degrees(np.arctan2(dxdy[:, 1], dxdy[:, 0]))
    dist = _e_2d_(arr, min_f)
    return arr_ordered, ang, dist


def _area_centroid_(a):
    r"""Calculate area and centroid for a singlepart polygon, `a`.

    This is also used to calculate area and centroid for a Geo array's parts.

    Notes
    -----
    For multipart shapes, just use this syntax:

    >>> # rectangle with hole
    >>> a0 = np.array([[[0., 0.], [0., 10.], [10., 10.], [10., 0.], [0., 0.]],
                      [[2., 2.], [8., 2.], [8., 8.], [2., 8.], [2., 2.]]])
    >>> [_area_centroid_(i) for i in a0]
    >>> [(100.0, array([ 5.00,  5.00])), (-36.0, array([ 5.00,  5.00]))]
    """
    a = _base_(a)
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    t = e1 - e0
    area = np.sum((e0 - e1) * 0.5)
    x_c = np.sum((x1 + x0) * t, axis=0) / (area * 6.0)
    y_c = np.sum((y1 + y0) * t, axis=0) / (area * 6.0)
    return area, np.asarray([-x_c, -y_c])


def _bit_area_(a):
    """Mini e_area, used by `areas` and `centroids`.

    Negative areas are holes.  This is intentionally reversed from
    the `shoelace` formula.
    """
    a = _base_(a)
    x0, y1 = (a.T)[:, 1:]  # cross set up as follows
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)  # 2024-03-28 modified
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.sum((e0 - e1) * 0.5)


def _bit_crossproduct_(a, is_closed=True, extras=False):
    """Cross product.

    Used by `is_convex`, `_angles_3pnt_` and `_clean_segments_`.

    .. note::

        np.cross for 2D arrays was deprecated in numpy 2.0, use `cross_2d`
    """
    def cross2d(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    a = _base_(a)
    if is_closed:
        if np.allclose(a[0], a[-1]):  # closed loop, remove dupl.
            a = a[:-1]
    ba = a - np.concatenate((a[-1][None, :], a[:-1]), axis=0)
    bc = a - np.concatenate((a[1:], a[0][None, :]), axis=0)
    # cr = np.cross(ba, bc) + 0.0  # deprecated
    cr = cross2d(ba, bc)
    if extras:
        return cr, ba, bc
    return cr


def _bit_min_max_(a):
    """Extent of a sub-array in an object array."""
    a = _base_(a)
    a = np.atleast_2d(a)
    return np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))


def _bit_length_(a):
    """Calculate segment lengths of poly geometry."""
    a = _base_(a)
    diff = a[1:] - a[:-1]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def _bit_segment_angles_(a, fromNorth=False):
    """Geo array, object or ndarray segment angles for polygons or polylines.

    Used by `segment_angles` and `min_area_rect`.

    Example
    -------
    # clockwise oriented connected segments
    a = np.array([[  2.00,   0.00], [  1.00,   0.00], [  0.00,   1.00]])
    _bit_segment_angles_(a, False)  # -- array([ 180.00,  135.00])
    _bit_segment_angles_(a, True)   # -- array([ 270.00,  315.00])
    """
    a = _base_(a)
    dxy = a[1:] - a[:-1]
    ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


def _rotate_(a, R, as_group):
    """Rotation helper.

    Parameters
    ----------
    geo_arr : array
        The input geo array, which is split here.
    as_group : boolean
        True, all shapes are rotated about the extent center.  False, each
        shape is rotated about its center.
    R : array
        The rotation matrix, passed on from Geo.rotate.
    clockwise : boolean
    """
    if not hasattr(a, "IFT"):
        print("Geo array required")
        return None
    shapes = _bit_check_(a)
    out = []
    if as_group:  # -- rotate as a whole
        cent = np.mean(a.XY, axis=0)
        return np.einsum('ij,jk->ik', a.XY - cent, R) + cent
    #
    uniqs = []
    for chunk in shapes:  # -- rotate individually
        _, idx = np.unique(chunk, True, axis=0)
        uniqs.append(chunk[np.sort(idx)])
    cents = [np.mean(i, axis=0) for i in uniqs]
    for i, chunk in enumerate(shapes):
        ch = np.einsum('ij,jk->ik', chunk - cents[i], R) + cents[i]
        out.append(ch)
    return out


def _scale_(a, factor=1):
    """Scale a geometry equally."""
    a = _base_(a)
    cent = np.min(a, axis=0)
    shift_orig = a - cent
    scaled = shift_orig * [factor, factor]
    return scaled + cent


def _translate_(a, dx=0, dy=0):
    """Move/shift/translate by dx, dy to a new location.

    Parameters
    ----------
    a : array-like
        A 2D array of coordinates with (x, y) shape or (N, x, y).
    dx, dy : numbers, list
        If dy is None, then dx must be array-like consisting of two values.

    Notes
    -----
    >>> dx, dy = np.mean(a, axis=0)  # to center about the x,y origin.
    """
    a = _base_(a)
    if a.ndim == 1:
        a = a.reshape(1, a.shape[0], 2)
        return np.array([i + [dx, dy] for i in a])
    return a + [dx, dy]


def _trans_rot_(a, angle=0.0, clockwise=False):
    """Rotate shapes about their center or individually."""
    a = _base_(a)
    if clockwise:
        angle = -angle
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s, c)))
    cent = np.mean(np.unique(a, axis=0), axis=0)
    return np.einsum('ij,jk->ik', a - cent, R) + cent


def _perp_(a):
    """Perpendicular to array."""
    b = np.empty_like(a)
    b_dim = b.ndim
    if b_dim == 1:
        b[0] = -a[1]
        b[1] = a[0]
    elif b_dim == 2:
        b[:, 0] = -a[:, 1]
        b[:, 1] = a[:, 0]
    return b


# ---- ---------------------------
# ---- (4) Geo / ndarray functions
#
def geom_angles(a, fromNorth=False):
    """Polyline/segment angles.

    Parameters
    ----------
    a : array-like
        A Geo array or a list of arrays representing the polyline shapes.
    """
    bits = _bit_check_(a, just_outer=False)
    out = []
    for b in bits:
        dxy = b[1:] - b[:-1]
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        if fromNorth:
            ang = np.mod((450.0 - ang), 360.)
        out.append(ang)
    return out


def segment_angles(a, fromNorth=False):
    """Return segment angles for Geo, object or ndarrays.

    It is assumed that `a` represents polygons or polylines.

    See Also
    --------
    `angles_segment` in Geo class which is quicker for them.

    splitter = geo.To - np.arange(1, geo.N + 1)
    """
    a = _bit_check_(a, just_outer=False)
    ang = [_bit_segment_angles_(i, fromNorth) for i in a]
    return ang


# ---- ---------------------------
# ---- (5) compare, remove, keep geometry
#
def a_eq_b(a, b, atol=1.0e-8, rtol=1.0e-5, return_pnts=False):
    r"""Return indices of, or points where, two point arrays are equal.

    Parameters
    ----------
    a, b : 2d arrays
        No error checking so ensure that a and b are 2d, but their shape need
        not be the same.

    Notes
    -----
    Modified from `np.isclose`, stripping out the nan and ma checks.
    Adjust atol and rtol appropriately.

    >>> np.nonzero(a_eq_b(a, b))[0]

    One could use::

    >>> np.equal(A, B).all(1).nonzero()[0]
    >>> np.where((a[:, None] == b).all(-1).any(1))[0]

    If you aren't worried about floating-point issues in equality checks.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if b.size > 2:
        b = b[:, None]
    w = np.less_equal(np.abs(a - b), atol + rtol * abs(b)).all(-1)
    if w.ndim > 1:
        w = w.any(-1).squeeze()
    if return_pnts:
        return b[w].squeeze()
    return w.squeeze()


def close_pnts(a, b, tol=.0001, ret_whr=True):
    """Alternate to `a_eq_b`. Returns indices or common points within `tol`."""
    ba = np.abs(b - a[:, None])
    whr = (ba.sum(axis=-1, keepdims=True) < tol).any(-1)
    whr = np.nonzero(whr)
    if ret_whr:
        return whr
    w0, w1 = whr[0], whr[1]
    return a[w0], b[w1]


def classify_pnts(a):
    """Classify Geo array points.

    Use np.argsort to sort the array and get the indices.
    >>> z = uts(sq2)  # convert to a structured array
    >>> perm = z.argsort(kind='mergesort')
    >>> aux = z[perm]  # sorted
    >>> sq2.shp_pnt_ids[perm]  # gives the ids and their shape
    """
    result = np.unique(
        a,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        axis=0
        )
    uni, idx, inv, cnt = result
    n0 = np.arange(a.shape[0] + 1, dtype='int')
    n1 = a.Fr
    n2 = a.To - 1  # duplicate start/end ids
    ft = np.sort(np.concatenate((n1, n2)))
    rem_st_end = set(n0).symmetric_difference(ft)
    # rem_st_end = np.sort(np.asarray(list(set(n0).symmetric_difference(ft))))
    eq_1 = set(idx[cnt == 1])
    gt_2 = set(idx[cnt > 2])
    eq_2 = rem_st_end.symmetric_difference(eq_1.union(gt_2))
    #
    eq_1 = np.sort(np.asarray(list(eq_1)))
    eq_2 = np.sort(np.asarray(list(eq_2)))
    gt_2 = np.sort(np.asarray(list(gt_2)))
    return ft, eq_1, eq_2, gt_2  # rem_strt


def pnt_segment_info(arr):
    """Return point segmentation information.

    Notes
    -----
    p_ids : Nx2 Geo array
        The first column is the point number and the second column is the
        feature they belong to.  The point IDs are in the sequence order of
        the poly feature construction.  This information is used to determine
        where duplicates occur and to what features they belong.
    """
    uni, idx, inv, cnts = np.unique(arr, True, True, True, axis=0)
    # index = sorted(idx)
    p_ids = arr.pnt_indices()  # get the point IDs, they will not be sorted
    out = []
    uni_cnts = np.unique(cnts).tolist()
    for i in uni_cnts:
        sub0 = uni[cnts == i]
        sub1 = idx[cnts == i]
        sub2 = p_ids[sub1]
        out.append([i, sub0, sub1, sub2])
    return out


def common_pnts(pnts, self, remove_common=True):
    """Check for coincident points between `pnts` and the Geo array.

    Parameters
    ----------
    pnts : 2D ndarray
        The points (N, 2) that you are looking for in the Geo array.
    remove_common : boolean
        True, returns an ndarray with the common points removed.
        False, returns the indices of the unique entries in `pnts`, aka,
        the indices of the common points between the two are not returned.

    See Also
    --------
    `np_geom.pnts_in_pnts` for a variant with slightly different returns.
    """
    w = np.where(np.equal(pnts[:, None], self).all(-1))[0]
    if len(w) > 0:
        uni = np.unique(pnts[w], axis=0)
        w1 = np.where((pnts == uni[:, None]).all(-1))[1]
        idx = [i for i in np.arange(len(pnts)) if i not in w1]
        if remove_common:  # equals... return np.delete(pnts, w1, axis=0)
            return pnts[idx]
        return idx
    print("{} not found".format(pnts))
    return pnts


def compare_segments(a, b, both_ways=True, invert=False, return_idx=False):
    """Return common segments between two poly* features.

    Parameters
    ----------
    a, b : arrays, 2D with ndim=2
        Arrays that may share common segments.  If the arrays are Nx2, from-to
        points will be constructed from the arrays.  If the arrays are Nx4, it
        is assumed that the from-to pairs have already been constructed.
    both_ways : boolean
        True, check for directionality in shared segments.
    unique : boolean
        True, return unique values.
    invert : boolean
        True, look for those not in.
    return_idx : boolean
        True, return indices in `a` that have common segments in `b`.

    Notes
    -----
    The segments of `b` are compared to `a` in both the forward and reverse
    directions in case there is directionality in shared segments.
    """
    ids_ = []
    segs_ = []
    shp0 = a.shape[1]
    shp1 = b.shape[1]
    if (a.ndim != 2 & b.ndim != 2) or shp0 != shp1:
        print("Arrays do not have to correct number of dimensions or shape.")
        return None
    if shp0 == 2:
        a_segs = np.concatenate((a[:-1], a[1:]), axis=1)
        b_segs = np.concatenate((b[:-1], b[1:]), axis=1)
    elif shp0 == 4:
        a_segs = np.copy(a)
        b_segs = np.copy(b)
    result0 = (a_segs[:, None] == b_segs).all(-1).any(-1)  # ** see reference
    if invert:
        result0 = ~result0
    idx_a = np.nonzero(result0)[0].tolist()
    if len(idx_a) > 0:
        segs_.append(a_segs[idx_a])
        ids_.append(idx_a)
    if both_ways:
        tmp = b_segs[:, [2, 3, 0, 1]]
        result1 = (a_segs[:, None] == tmp).all(-1).any(-1)
        if invert:
            result1 = ~result1
        idx_a_rev = np.nonzero(result1)[0].tolist()
        if len(idx_a_rev) > 0:
            segs_.append(tmp[idx_a_rev])
            ids_.append(idx_a_rev)
    if len(segs_) == 0:
        return [None, []]
    return [segs_, ids_]


def compare_geom(arr, look_for, unique=True, invert=False, return_idx=False):
    """Look for duplicates or common points in two 2D arrays.

    ** Used to find points or segments that are common between geometries. **

    Parameters
    ----------
    arr : array, 2D
        The main array, preferably the larger of the two.
    look_for : array, 2D
        The array to compare with.
    unique : boolean
        True, return unique values.
    invert : boolean
        True, look for those not in.
    return_idx : boolean
        True, return both the values and the indices of the comparison.

    Returns
    -------
    The intersection or difference in both arrays, depending on `invert`.
    The indices of where the objects were found

    >>> a = np.array([[ 5.,  5.], [ 6.,  6.], [10., 10.], [12., 12.]])
    >>> b = np.array([[ 6.,  6.], [12., 12.]])

    >>> compare_geom(a, b, invert=False)
    ... array([[ 6.,  6.],
    ...        [12., 12.]])

    >>> compare_geom(a, b, invert=True)
    ... array([[ 5.,  5.],
    ...        [10., 10.]])
    """
    result = (arr[:, None] == look_for).all(-1).any(-1)  # ** see reference
    if sum(result) == 0:
        return None
    if invert:
        keep = ~result
        idx = np.where(keep)[0]
        w = np.nonzero((idx[1:] - idx[:-1] > 1))[0] + 1
        bits = np.array_split(idx, w)
        if 0 == bits[0][0]:
            tmp = arr[np.concatenate(bits[::-1])]
        else:
            tmp = arr[np.concatenate(bits)]
    else:
        idx = np.where(result)[0]
        tmp = arr[result]
    if unique:
        out, ids = np.unique(tmp, return_index=True, axis=0)
        idx = sorted(ids)
        tmp = tmp[idx]
    if return_idx:
        return tmp, idx
    return tmp


def keep_geom(arr, look_for, **kwargs):
    """Keep points in `arr` that match those in `look_for`."""
    return compare_geom(arr, look_for, invert=False, return_idx=False)


def remove_geom(arr, look_for, **kwargs):
    """Remove points from `arr` that match those in `look_for`."""
    return compare_geom(arr, look_for, unique=False,
                        invert=True, return_idx=False)


def del_seq_dups(arr, poly=True):
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
    # -- view data, reshape to (N,)
    dt = [('f{i}'.format(i=i), dt_in) for i in range(arr.shape[1])]
    tmp = arr.view(dtype=dt, type=np.ndarray).squeeze()
    # -- mask and check for sequential equality.
    mask = np.empty((shp_in[0],), np.bool_)
    mask[0] = True
    mask[1:] = tmp[:-1] != tmp[1:]
    # wh_ = np.nonzero(mask)[0]
    # sub_arrays = np.array_split(arr, wh_[wh_ > 0])
    tmp = arr.base[mask]  # -- slice original array sequentially unique points
    if poly:  # -- polygon source check
        if not (tmp[0] != tmp[-1]).all(-1):
            arr = np.concatenate((tmp, tmp[0, None]), axis=0)
            return arr
    return tmp


def multi_check(arr):
    """Check for possible multiple parts in an array.

    Parameters
    ----------
    arr : array_like
        Nx2 array of clockwise coordinates with the first and last being equal.

    Notes
    -----
    If there is a duplicate vertex, in a polygon array, other than the
    start/end vertex, this may represent the presence of a multipart shape.
    In a `Geo` array, this really isnt necessary.
    """
    uni, idx, cnt = np.unique(arr, return_index=True, return_counts=True,
                              axis=0)
    chk = uni[cnt > 1]
    # whr = np.nonzero(cnt > 1)[0]
    parts = []
    if len(chk) == 1:
        val = chk[1]
        result = (arr[:, None] == val).all(-1).any(-1)
        idxs = np.nonzero(result)[0]
        bits = np.array_split(arr, idxs.tolist())
        parts = [np.concatenate((bits[0], bits[-1]), axis=0),
                 np.concatenate((bits[1], np.atleast_2d(bits[1][0])), axis=0)]
        # -- assert idxs only contains 2 ids for this to work
    elif len(chk) > 2:
        result = (arr[:, None] == chk).all(-1).any(-1)
        idxs = np.nonzero(result)[0]
        bits = np.array_split(arr, idxs.tolist())
        parts = []
        for i in range(1, len(bits), 2):
            sub = np.concatenate((bits[i], bits[i + 1]), axis=0)
            parts.append(sub)
    return parts


# ---- ---------------------------
# ---- (6) sort coordinates
#
def sort_xy(a, x_ascending=True, y_ascending=True, return_order=True):
    """Sort point coordinates lexicographically.

    Return sorted points or just the order.

    Parameters
    ----------
    a : ndarray or Geo array
        Convert lists of points to ndarrays or Geo arrays first.
    x_ascending, y_ascending : boolean
        If False, sort is done in decending order by axis.
    return_order : boolean
        True returns the order and not the sorted points.  False returns the
        sorted points.
    """
    a = _base_(a)
    x_s = a[:, 0]
    y_s = a[:, 1]
    order = None
    if x_ascending:
        if y_ascending:
            order = np.lexsort((y_s, x_s))
        else:
            order = np.lexsort((-y_s, x_s))
    elif not x_ascending:
        if y_ascending:
            order = np.lexsort((y_s, -x_s))
        else:
            order = np.lexsort((-y_s, -x_s))
    if return_order:
        return order
    return a[order]


def swap_segment_pnts(a):
    """Swap point pairs so that the x-values are increasing.

    Parameters
    ----------
    a : array
        The array is usually an Nx2 array of points representing the boundary
        of a polygon or a polyline.  An Nx4 array representing (x0,y0, x1,y1)
        from-to pairs may also be used.

    Returns
    -------
    An Nx4 array representing the from-to point pairs is returned.  The angles
    of these sequences will be in the first two quadrants (I, II).
    """
    shp0, shp1 = a.shape
    shp0 -= 1
    if shp1 == 2:
        tmp = np.concatenate((a[:-1], a[1:]), axis=1)
        idx_ = np.zeros((shp0, 2), dtype=int)
        idx_[:, 0] = np.arange(shp0)
        idx_[:-1, 1] = np.arange(1, shp0)
    elif shp1 == 4:
        tmp = np.copy(a)
    else:
        print("Array shape of Nx2 or Nx4 required.")
        return None
    #
    # -- compare x columns, fill output array
    gte_idx = tmp[:, 0] >= tmp[:, 2]  # compare x-coordinates
    # check y if x is equal
    eq_idx = np.logical_and(tmp[:, 0] == tmp[:, 2], tmp[:, 1] <= tmp[:, 3])
    gte_idx = gte_idx ^ eq_idx
    em = np.zeros(tmp.shape)
    em[gte_idx == 0] = tmp[gte_idx == 0]  # correct orientation, keep values
    em[gte_idx == 1, :2] = tmp[gte_idx == 1, -2:]  # swap needed
    em[gte_idx == 1, -2:] = tmp[gte_idx == 1, :2]
    whr = np.nonzero(gte_idx == 1)[0]  # index check
    idx_[whr] = idx_[whr][:, [1, 0]]
    return em, idx_


def sort_segment_pairs(a):
    """Sort point pairs.

    Parameters
    ----------
    a : array
        An Nx2 array of points representing the boundary of a polygon or a
        polyline since it is assumed that the segments are somehow connected.

    Returns
    -------
    The new segments and their from-to indices.
    The segments are sorted lexicographically with ascending x-values for the
    point pairs.  The segments are oriented so they lie in quadrant I or II.

    Notes
    -----
    See `swap_segment_pnts` for parameters and details.
    """
    tmp, idx_ = swap_segment_pnts(a)
    ind = np.lexsort((tmp[:, 2], tmp[:, 0]))
    return tmp[ind], idx_[ind]  # ind, idx_


def dist_angle_sort(a, sort_point=None, close_poly=True):
    """Return a radial and distance sort of points relative to `sort_point`.

    Parameters
    ----------
    a : array-like
        The array to sort.
    sort_point : list
        The [x, y] value of the sort origin.  If `None`, then the minimum
        x,y value from the inputs is used.

    Useful for polygons.  First and last point equality is checked.
    """
    a = np.array(a)
    min_f = np.array([np.min(a[:, 0]), np.min(a[:, 1])])
    dxdy = np.subtract(a, np.atleast_2d(min_f))
    ang = np.degrees(np.arctan2(dxdy[:, 1], dxdy[:, 0]))
    dist = _e_2d_(a, min_f)
    ang_dist = np.vstack((ang, dist)).T
    keys = np.argsort(uts(ang_dist))
    rev_keys = keys[::-1]   # works
    arr = a[rev_keys]
    if np.all(arr[0] == arr[-1]):
        arr = np.concatenate((arr, arr[0][None, :]), axis=0)
    return arr


def radial_sort(a, extent_center=True, close_poly=True, clockwise=True):
    """Return a radial sort of points around their center.

    Useful for polygons.  First and last point equality is checked.
    """
    def extent_cent(a):
        """Extent center."""
        extent = np.concatenate(
            (np.min(uniq, axis=0), np.max(uniq, axis=0))).reshape(2, 2)
        return np.mean(extent, axis=0)

    def cent_dist(a, cent):
        """Docs."""
        diff = a - cent[None, :]
        dist_arr = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        return dist_arr

    a = _base_(a)
    uniq = np.unique(a, axis=0)
    if extent_center:
        cent = extent_cent(uniq)
    cent = np.mean(uniq, axis=0)
    dxdy = uniq - cent
    angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])
    idx = angles.argsort()
    srted = uniq[idx]
    if close_poly:
        srted = np.concatenate((srted, [srted[0]]), axis=0)
    if clockwise:
        srted = srted[::-1]
    return srted


# ---- ---------------------------
# ---- (7) others functions
#
def interweave(arr, as_3d=False):
    """Weave an array to produce from-to pairs.  Returns a copy.

     >>> np.array(list(zip(arr[:-1], arr[1:])))  # returns a copy

    Parameters
    ----------
    arr : ndarray
        A 2d ndarray.
    as_3d : boolean
        If True, an (N, 2, 2) shaped array is returned, otherwise the
        `from-to` values appear on the same line with a (N, 4) shape.
    """
    fr_to = np.concatenate((arr[:-1], arr[1:]), axis=1)
    if as_3d:
        return fr_to.reshape((-1, 2, 2))  # for ndim=3d
    return fr_to


def shape_finder(arr, start=0, prn=False, structured=True):
    """Provide the structure of an array/list which may be uneven and nested.

    Parameters
    ----------
    arr : array-like
        An list/tuple/array of objects. In this case points. Shapes are
        usually formed as parts with/without holes and they may have multiple
        parts.
    start : integer
        Start number to use for the counter.  Usually 0 or 1.
    prn : boolean
        True, to print the output.  False returns a structured array.
    structured : boolean
        True, returns a structured array if prn is False, otherwise an ndarray
        is returned.

    Requires
    --------
    `prn_tbl` from npg_prn.

    Used by
    -------
    `npg_prn.prn_lists` and `npg_prn.prn_arrays`.

    Notes
    -----
    >>> # -- 3 shapes, (1, 2, 3)
    >>> array([[1, 0, 0, (5, 2)],
               [1, 0, 1, (5, 2)],  shape 1 has 2 parts (0, 1)
               [1, 0, 2, (4, 2)],     part 0 has 4 parts (0, 1, 2, 3)
               [1, 0, 3, (4, 2)],
               [1, 1, 0, (5, 2)],     part 1 has 2 parts (0, 1)
               [1, 1, 1, (4, 2)],
               [2, 0, 0, (9, 2)],  shape 2 has 2 parts
               [2, 1, 0, (10, 2)],    part 1 has 4 parts (0, 1, 2, 3)
               [2, 1, 1, (4, 2)],
               [2, 1, 2, (4, 2)],
               [2, 1, 3, (4, 2)],
               [3, 0, 0, (4, 2)]], shape 3 has 1 part consisting of 4 points
              dtype=object)
    """
    def _len_check_(arr):
        """Check iterator lengths."""
        if len(arr) == 1:
            return False
        q = [len(a) == len(arr[0])      # check subarray and array lengths
             if hasattr(a, '__iter__')  # if it is an iterable
             else False                 # otherwise, return False
             for a in arr]              # for each subarray in the array
        return np.all(q)

    def _arr_(arr):
        """Assign dtype based on nested array lengths from `_len_check_`."""
        dt = 'float' if _len_check_(arr) else 'O'
        return np.asarray(arr, dtype=dt).squeeze()
    #
    cnt = start
    info = []
    arrs = []
    if isinstance(arr, (list, tuple)):
        if len(arr[0]) == 2 and isinstance(arr[0][0], (int, float)):
            arrs = [arr]
        else:
            arrs = [i for i in arr if hasattr(i, '__len__')]
    elif isinstance(arr, np.ndarray):
        if arr.dtype.kind == "O":
            arrs = arr
        elif len(arr.shape) == 2:
            arrs = [arr]
    for ar in arrs:
        a0 = _arr_(ar)  # -- create an appropriate array
        if a0.dtype.kind in 'efdg' or len(a0.shape) > 1:
            info.append([cnt, 0, 0, a0.shape[0]])
        else:
            i = 0
            for a1 in a0:
                a1 = _arr_(a1)
                j = 0
                if a1.dtype.kind in 'efdg' or len(a1.shape) > 1:
                    info.append([cnt, i, j, a1.shape[0]])
                else:
                    for a2 in a1:
                        a2 = _arr_(a2)
                        info.append([cnt, i, j, a2.shape[0]])
                        j += 1
                i += 1
        cnt += 1
    dt = [("shape", "i8"), ("part", "i8"), ("bit", "i8"), ("Nx2", "i8")]
    info = np.array(info)
    if structured:
        out = np.zeros((len(info),), dtype=dt)
        names = out.dtype.names
        for i, nme in enumerate(names):
            out[nme] = info[:, i]
        if prn:
            prn_tbl(out)
            return None
        return out
    return info


def coerce2array(arr, start=0):
    """Return arrays using the principles of `shape_finder`.

    Parameters
    ----------
    arr : array-like
        Either lists of lists, lists or arrays, arrays of lists or similar.
        There is the expectation that the objects represent Nx2 geometry
        objects with potentially multiple parts and bits.
    start : integer
        Either use 0 or 1 for array part numbering purposes.

    Returns
    -------
    Most likely an object array will be returned if the input structure is
    also nested.
    """
    #
    def _len_check_(arr):
        """Check iterator lengths."""
        arr = np.asarray(arr, dtype='O').squeeze()
        if arr.shape[0] == 1:
            return False, len(arr)
        q = [len(a) == len(arr[0])      # check subarray and array lengths
             if hasattr(a, '__iter__')  # if it is an iterable
             else False                 # otherwise, return False
             for a in arr]              # for each subarray in the array
        return np.all(q), len(arr)

    def _arr_(arr):
        """Assign dtype based on nested array lengths from `_len_check_`."""
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                return arr
        result = [_len_check_(i) for i in arr]
        out = []
        if len(result) == 1:
            return np.asarray(arr, dtype='float').squeeze()
        for i, r in enumerate(result):
            if r[0]:
                out.append(np.asarray(arr[i], dtype='float').squeeze())
            else:
                sub = []
                chk2 = _len_check_(arr[i])
                if chk2[0]:
                    out.append(np.asarray(arr[i], dtype='float').squeeze())
                else:
                    # chk3 = _len_check_(arr[i])  # testing, keep for now
                    # print("chk3 {}".format(chk3))
                    for ar in arr[i]:
                        sub.append(np.asarray(ar, dtype='float').squeeze())
                    dt1 = 'O' if len(sub) > 1 else 'float'
                    out.append(np.asarray(sub, dtype=dt1))
        return np.asarray(out, dtype='O')
    #
    out = []
    chk = _len_check_(arr)
    if chk[0] and chk[1] > 2:
        return np.array(arr)
    for ar in arr:
        out.append(_arr_(ar))
    return np.array(out, dtype="O")


def flat(lst):
    """Flatten input. Basic flattening but doesn't yield where things are."""
    def _flat(lst, r):
        """Recursive flattener."""
        if len(lst) > 0:
            if not isinstance(lst[0], (list, np.ndarray, tuple)):  # added [0]
                r.append(lst)
            else:
                for i in lst:
                    r = r + flat(i)
        return r
    return _flat(lst, [])


def reclass_ids(vals=None):
    """Reclass a 1d sequence of integers.  Usually used to reclass ID values.

    Parameters
    ----------
    vals : integers
        A sequence of integers representing ID values for objects.  They may
        contain gaps in the sequence and duplicates of values.

    Notes
    -----
    Reclass an array to integer classes based upon unique values.
    Unique values are easily determined by sorting, finding successive
    differences as a reverse boolean, then producing and assigning a
    cumulative sum.
    """
    if vals is None:
        return None
    idx = np.arange(len(vals), dtype="int32")
    ordr = np.zeros(len(vals), dtype="int32")
    dt = [("ID", "int32"), ("Values", "U5"), ("Order", "int32")]
    a = np.array(list(zip(idx, vals, ordr)), dtype=dt)
    #
    # sort the array, determine where consecutive values are equal
    # reclass them using an inverse boolean and produce a cumulative sum
    s_idx = np.argsort(a, order=['Values', 'ID'])
    final = a[s_idx]
    boolv = final['Values'][:-1] == final['Values'][1:]  # read very carefully
    w = np.where(boolv, 0, 1)   # find indices where they are the same
    # sort back
    final["Order"][1:] = np.cumsum(w)  # csum[1:] = np.cumsum(w)
    final = final[np.argsort(final, order=['ID'])]
    # frmt = "{}\nInput array...\n{!r:}\n\nFinal array\n{!r:}"
    # args = [dedent(recl_ids.__doc__), a.reshape(-1, 1), final.reshape(-1, 1)]
    # print(dedent(frmt).format(*args))
    return final['Order']  # a, final


# ===========================================================================
# ---- ==== main section
if __name__ == "__main__":
    """optional location for parameters"""
    # in_fc = r"C:\arcpro_npg\Project_npg\npgeom.gdb\Polygons"
    # in_fc = r"C:\arcpro_npg\Project_npg\npgeom.gdb\Polygons2"
