# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
-----------
npg_helpers
-----------

**General helper functions**

----

Script :
    npg_helpers.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-12-25

Purpose
-------
Helper functions for Geo arrays and used by npg_geom.py.

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


Extras
------
**Generating ``__all__`` and ``__helpers__``**

>>> not_in = [
...     '__all__', '__builtins__', '__cached__', '__doc__', '__file__',
...     '__loader__', '__name__', '__package__', '__spec__', 'np', 'npg',
...     'sys', 'script'
...     ]

>>> __all__ = [i for i in dir(npg.npg_helpers)
...            if i[0] != "_" and i not in not_in]

>>> __helpers__ = [i for i in dir(npg.npg_helpers)
...                if i[0] == "_" and i not in not_in]

"""

# pylint: disable=C0103,C0415
# pylint: disable=R0912, R0913, R0914, R1710, R1705
# pylint: disable=W0105,W0212,W0612,W0613

import sys
# from textwrap import dedent

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

# if 'npg' not in list(locals().keys()):
#     import npg

# import npg_prn  # noqa
from npg_prn import prn_tbl  # used by ``shape_finder``

script = sys.argv[0]  # print this should you need to locate the script

nums = 'efdgFDGbBhHiIlLqQpP'

# -- See script header
__all__ = [
    'common_pnts', 'compare_geom', 'flat', 'interweave', 'keep_geom',
    'polyline_angles',
    'radial_sort', 'remove_geom', 'segment_angles', 'shape_finder',
    'coerce2array', 'dist_angle_sort', 'sort_xy', 'stride_2d', 'reclass_ids'
    ]

__helpers__ = [
    'prn_tbl', '_angles_3pnt_', '_od_angles_dist_', '_area_centroid_',
    '_bit_area_', '_bit_check_', '_bit_crossproduct_', '_bit_length_',
    '_bit_min_max_', '_bit_segment_angles_', '_from_to_pnts_', '_get_base_',
    '_in_LBRT_', '_in_extent_', '_is_ccw_', '_is_clockwise_', '_is_right_side',
    '_isin_2d_', '_pnts_in_extent_', '_rotate_', '_scale_', '_to_lists_',
    '_trans_rot_', '_translate_', '_perp_'
    ]  # ---- core bit functions

__all__ = __helpers__ + __all__


# ---------------------------------------------------------------------------
# ---- (1) Helpers
#
def _get_base_(a):
    """Return the base array of a Geo array.  Shave off microseconds."""
    if hasattr(a, "IFT"):
        return a.XY
    return a


def _bit_check_(a, just_outer=False):
    """Check for bits and convert if necessary.

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
    a = _get_base_(a)
    return np.concatenate((a[:-1], a[1:]), axis=1)


def _to_lists_(a, outer_only=True):
    """Return list or list of lists for a Geo or ndarray.

    Parameters
    ----------
    a : array-like
        Either a Geo array or ndarray.
    outer_only : boolean
        True, returns the outer-rings of a Geo array.  False, returns the bit.

    See Also
    --------
    ``Geo_to_lists``, ``Geo_to_arrays`` if you want to maintain the potentially
    nested structure of the geometry.
    """
    if hasattr(a, "IFT"):
        if outer_only:
            return a.outer_rings(False)  # a.bits
        return a.bits
    elif isinstance(a, np.ndarray):
        if a.dtype.kind == 'O':
            return a
        elif a.ndim == 2:
            return [a]
        elif a.ndim == 3:
            return [i for i in a]
    else:  # a list already
        return a


# ---------------------------------------------------------------------------
# ---- (2) geometry helpers
#
def _angles_3pnt_(a, inside=True, in_deg=True):
    """Worker for Geo `polygon_angles`, `polyline_angles` and `min_area_rect`.

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

    Notes
    -----
    Sum of interior angles of a polygon with ``n`` edges::

        (n − 2)π radians or (n − 2) × 180 degrees
        n = number of unique vertices

    | euler`s formula
    | number of faces + number of vertices - number of edges = 2
    | rectangle : 1 + 5 - 4 = 2
    | triangle  : 1 + 4 - 3 = 2
    """
    if np.allclose(a[0], a[-1]):                 # closed loop, remove dupl.
        a = a[:-1]
    cr, ba, bc = _bit_crossproduct_(a, extras=True)
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
    def _e_2d_(a, p):
        """See npg_helpers ``_e_2d_``."""
        diff = a - p[None, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

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
    a = _get_base_(a)
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
    """Mini e_area, used by `areas` and `centroids`."""
    a = _get_base_(a)
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.sum((e0 - e1)*0.5)


def _bit_crossproduct_(a, extras=False):
    """Cross product.  Used by `is_convex` and `_angles_3pnt_`."""
    a = _get_base_(a)
    ba = a - np.concatenate((a[-1][None, :], a[:-1]), axis=0)
    bc = a - np.concatenate((a[1:], a[0][None, :]), axis=0)
    if extras:
        return np.cross(ba, bc), ba, bc
    return np.cross(ba, bc)


def _bit_min_max_(a):
    """Extent of a sub-array in an object array."""
    a = _get_base_(a)
    a = np.atleast_2d(a)
    return np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))


def _bit_length_(a):
    """Calculate segment lengths of poly geometry."""
    a = _get_base_(a)
    diff = a[1:] - a[:-1]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def _bit_segment_angles_(a, fromNorth=False):
    """Geo array, object or ndarray segment angles for polygons or polylines.

    Used by `segment_angles` and `min_area_rect`.
    """
    a = _get_base_(a)
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
    a = _get_base_(a)
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
    a = _get_base_(a)
    if a.ndim == 1:
        a = a.reshape(1, a.shape[0], 2)
        return np.array([i + [dx, dy] for i in a])
    else:
        return a + [dx, dy]


def _trans_rot_(a, angle=0.0, clockwise=False):
    """Rotate shapes about their center or individually."""
    a = _get_base_(a)
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


# ---------------------------------------------------------------------------
# ---- (3) Condition checking
#
def _is_clockwise_(a):
    """Return whether the sequence (polygon) is clockwise oriented or not."""
    return 1 if _bit_area_(a) > 0. else 0


def _is_ccw_(a):
    """Counterclockwise."""
    return 0 if _bit_area_(a) > 0. else 1


def _isin_2d_(a, b, as_integer=False):
    """Perform a 2d `isin` check for 2 arrays.

    Parameters
    ----------
    a, b : arrays
        The arrays to compare.
    as_integer : boolean
        False, returns a boolean array.  True, returns integer array which may
        useful for some operations.

    Example
    -------
    >>> a = np.array([[ 5.00,  10.00], [ 5.00,  12.00], [ 6.00,  12.00],
                      [ 8.00,  12.00], [ 8.00,  11.00], [ 5.00,  10.00]])
    >>> b = np.array([[ 5.00,  12.00], [ 5.00,  15.00], [ 7.00,  14.00],
                      [ 6.00,  12.00], [ 5.00,  12.00]])
    >>> w0 = (a[:, None] == b).all(-1).any(-1)
    array([0, 1, 1, 0, 0, 0])
    >>> a[w0]
    array([[ 5.00,  12.00], [ 6.00,  12.00]])
    >>> w1 = (b[:, None] == a).all(-1).any(-1)
    >>> b[w1]
    array([[ 5.00,  12.00], [ 6.00,  12.00], [ 5.00,  12.00]])

    Reference
    ---------
    `<https://stackoverflow.com/a/51352806/6828711>`_.
    """
    a = _get_base_(a)
    b = _get_base_(b)
    out = (a[:, None] == b).all(-1).any(-1)
    if as_integer:
        return out.astype('int')
    return out


def _is_right_side(p, strt, end):
    """Determine if a point (p) is `inside` a line segment (strt-->end).

    Parameters
    ----------
    p, strt, end : array-like
        X,Y coordinates of the subject point and the start and end of the line.

    See Also
    --------
    line_crosses, in_out_crosses

    Notes
    -----
    A point is on the inside (right-side) if `position` is negative.  This
    assumes polygons are oriented clockwise.

    position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))

    So in essence, the reverse of _is_left_side with the outcomes reversed ;).
    """
    x, y, x0, y0, x1, y1 = *p, *strt, *end
    return (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)


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
    case = np.all(np.logical_and(LB < pnts, pnts <= RT), axis=1)
    if np.sum(case) > 0:
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
        return
    shp = np.asarray(extent).shape
    if shp == (2, 2):
        LB, RT = extent
    elif shp[0] == 4:
        extent = extent.reshape(2, 2)
        LB, RT = np.min(extent, axis=0), np.max(extent, axis=0)
    else:
        print(msg.format(extent))
        return
    idx = np.all(np.logical_and(LB < pnts, pnts <= RT), axis=1)
    if return_index:
        return idx.base
    return np.all(idx)


# ---------------------------------------------------------------------------
# ---- (4) Geo / ndarray stuff
#
def polyline_angles(a, fromNorth=False):
    """Polyline/segment angles.

    Parameters
    ----------
    a : array-like
        A Geo array or a list of arrays representing the polyline shapes
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

    It is assumed that ``a`` represents polygons or polylines.

    See Also
    --------
    `angles_segment` in Geo class which is quicker for them.

    splitter = geo.To - np.arange(1, geo.N + 1)
    """
    a = _bit_check_(a, just_outer=False)
    ang = [_bit_segment_angles_(i, fromNorth) for i in a]
    return ang


# ---------------------------------------------------------------------------
# ---- (5) compare, remove, keep geometry
#
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
    The intersection or difference in both arrays, depending on ``invert``.
    The indices of where the objects were found

    >>> a = np.array([[ 5.,  5.], [ 6.,  6.], [10., 10.], [12., 12.]])
    >>> b = np.array([[ 6.,  6.], [12., 12.]])

    >>> compare_2d(a, b, invert=False)
    ... array([[ 6.,  6.],
    ...        [12., 12.]])

    >>> compare_2d(a, b, invert=True)
    ... array([[ 5.,  5.],
    ...        [10., 10.]])
    """
    result = (arr[:, None] == look_for).all(-1).any(-1)  # ** see reference
    if sum(result) == 0:
        return None
    if invert:
        result = ~result
    idx = np.where(result)[0]
    out = arr[result]
    if unique:
        out = np.unique(out, axis=0)
    if return_idx:
        return out, idx
    return out


def keep_geom(arr, look_for, **kwargs):
    """Keep points in ``arr`` that match those in ``look_for``."""
    return compare_geom(arr, look_for, invert=False, return_idx=False)


def remove_geom(arr, look_for, **kwargs):
    """Remove points from ``arr`` that match those in ``look_for``."""
    return compare_geom(arr, look_for, unique=False,
                        invert=True, return_idx=False)


# ---------------------------------------------------------------------------
# ---- (6) sort coordinates
#
def sort_xy(a, x_ascending=True, y_ascending=True):
    """Sort points by coordinates.

    Parameters
    ----------
    a : ndarray or Geo array
        Convert lists of points to ndarrays or Geo arrays first.
    x_ascending, y_ascending : boolean
        If False, sort is done in decending order by axis.
    """
    a = _get_base_(a)
    x_s = a[:, 0]
    y_s = a[:, 1]
    if x_ascending:
        if y_ascending:
            return a[np.lexsort((y_s, x_s))]
        return a[np.lexsort((-y_s, x_s))]
    if y_ascending:
        if x_ascending:
            return a[np.lexsort((x_s, y_s))]
        return a[np.lexsort((-x_s, y_s))]


def dist_angle_sort(a, sort_point=None, close_poly=True):
    """Return a radial and distance sort of points relative to point.

    Parameters
    ----------
    a : array-like
        The array to sort.
    sort_point : list
        The [x, y] value of the sort origin.  If ``None``, then the minimum
        x,y value from the inputs is used.

    Useful for polygons.  First and last point equality is checked.
    """
    def _e_2d_(a, p):
        """Array points to point distance."""
        diff = a - p[None, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

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

    a = _get_base_(a)
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


def interweave(arr, as_3d=False):
    """Weave an arrays to produce from-to pairs.  Returns a copy.

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


def stride_2d(a, win=(2, 2), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of an 2D array.

    Parameters
    ----------
    a : 2D array
    win : tuple
        Window size in terms of rows and columns.
    stepby : tuple
        The steps to take in the X and Y direction along the array.

    Notes
    -----
    You can ravel a 2D array to facilitate moving from row to row.  See the
    examples.  Alternately, you can use moving blocks to obtain things like
    statistical parameters on `raster` data.

    This function is coming in version 1.20 in numpy.lib.stride_tricks.py::

        sliding_window_view(x, window_shape, axis=None, *,
                            subok=False, writeable=False):

    Example
    -------
    Create from-to points::

        # -- produces a `view` and not a copy
        >>> a = np.array([[0, 1], [2, 3], [4, 5]])
        >>> stride_2d(a.ravel(), win=(4,), stepby=(2,))

        array([[0, 1, 2, 3],
               [2, 3, 4, 5]])

        # -- alternatives, but they produce copies
        >>> np.concatenate((a[:-1], a[1:]), axis=1)
        >>> np.asarray(list(zip(a[:-1], a[1:])))

        # -- concatenate is faster, with 500 points in `s`.
        %timeit stride_2d(s.ravel(), win=(4,), stepby=(2,))
        21.7 µs ± 476 ns per loop (mean ± std. dev. of 7 runs, 10000 loops

        %timeit np.concatenate((s[:-1], s[1:]), axis=1)
        8.41 µs ± 158 ns per loop (mean ± std. dev. of 7 runs, 100000 loops

    A different stride::

        >>> stride_2d(b, win=(2, 2), stepby=(1, 1))
        array([[[0, 1],
                [2, 3]],

               [[2, 3],
                [4, 5]]])

    """
    from numpy.lib.stride_tricks import as_strided
    shp = np.array(a.shape)    # array shape 2D (r, c) or 3D (d, r, c)
    win_shp = np.array(win)    # window    (4,) (3, 3) or    (1, 3, 3)
    ss = np.array(stepby)      # step by   (2,) (1, 1) or    (1, 1, 1)
    newshape = tuple(((shp - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides, subok=True)
    return a_s.squeeze()


# ---------------------------------------------------------------------------
# ---- (7) others functions
#
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
    ``prn_tbl`` from npg_prn.

    Used by
    -------
    ``npg_prn.prn_lists`` and ``npg_prn.prn_arrays``.

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
            return
        return out
    else:
        return info


def coerce2array(arr, start=0):
    """Return arrays using the principles of ``shape_finder``.

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
                    chk3 = _len_check_(arr[i])
                    print("chk3 {}".format(chk3))
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
        if not isinstance(lst[0], (list, np.ndarray, tuple)):  # added [0]
            r.append(lst)
        else:
            for i in lst:
                r = r + flat(i)
        return r
    return _flat(lst, [])


def project_pnt_to_line(x1, y1, x2, y2, xp, yp):
    """Project a point on to a line to get the perpendicular location."""
    x12 = x2 - x1
    y12 = y2 - y1
    dotp = x12 * (xp - x1) + y12 * (yp - y1)
    dot12 = x12 * x12 + y12 * y12
    if dot12:
        coeff = dotp / dot12
        lx = x1 + x12 * coeff
        ly = y1 + y12 * coeff
        return lx, ly
    else:
        return None


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


def _iterate_(N, n):
    """Return combinations for array lengths."""
    import itertools
    combos = itertools.combinations(np.arange(N), n)
    return combos


# ===========================================================================
# ---- ==== main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
