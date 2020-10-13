# -*- coding: utf-8 -*-
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
    2020-10-07

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

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
# from textwrap import dedent

import numpy as np
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

# if 'npg' not in list(locals().keys()):
#     import npgeom as npg
import npGeo as npg

script = sys.argv[0]  # print this should you need to locate the script

nums = 'efdgFDGbBhHiIlLqQpP'

__all__ = [
    '_bit_check_', '_from_to_pnts_',
    '_angles_3pnt_', '_area_centroid_', '_bit_area_', '_bit_crossproduct_',
    '_bit_extent_', '_bit_length_', '_bit_segment_angles_',
    '_is_clockwise_', '_is_ccw_', '_isin_2d_', '_is_right_side', '_in_extent_',
    '_pnts_in_extent_', '_translate_', '_rot_', '_scale_', '_rotate_',
    'polyline_angles', 'segment_angles',
    'line_crosses', 'in_out_crosses', 'crossings',
    'common_pnts', 'compare_geom', 'keep_geom', 'remove_geom',
    'sort_xy', 'radial_sort', 'interweave', 'shape_finder'
]  # 'crossing_num', 'pnts_in_poly', '_bit_area_',  '_bit_length_',


# ---- core bit functions
def _bit_check_(a, just_outer=False):
    """Check for bits and convert if necessary.

    a : array_like
        Either a Geo array or a list of lists.  Conversion to bits or outer
        rings as desired.
    just_outer : boolean
        True, removes holds from the geometry.
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


def _from_to_pnts_(a):
    """Convert polygon/polyline shapes to from-to points.

    Parameters
    ----------
    a : array-like
        A Geo array bit or ndarray representing a singlepart shape.
    """
    return np.concatenate((a[:-1], a[1:]), axis=1)


# ---- bit helpers ----
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
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.sum((e0 - e1)*0.5)


def _bit_crossproduct_(a, extras=False):
    """Cross product.  Used by `is_convex` and `_angles_3pnt_`."""
    ba = a - np.concatenate((a[-1][None, :], a[:-1]), axis=0)
    bc = a - np.concatenate((a[1:], a[0][None, :]), axis=0)
    if extras:
        return np.cross(ba, bc), ba, bc
    return np.cross(ba, bc)


def _bit_extent_(a):
    """Extent of a sub-array in an object array."""
    a = np.atleast_2d(a)
    return np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))


def _bit_length_(a):
    """Calculate segment lengths of poly geometry."""
    diff = a[1:] - a[:-1]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def _bit_segment_angles_(a, fromNorth=False):
    """Geo array, object or ndarray segment angles for polygons or polylines.
    Used by `segment_angles` and `min_area_rect`.
    """
    dxy = a[1:] - a[:-1]
    ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


# ---- Condition checking
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


def _in_extent_(pnts, ext):
    """Return points in, or on the line of an extent. See `_in_LBRT_` also.

    Parameters
    ----------
    pnts : array
        An Nx2 array representing point objects.
    extent : array-like
        A 2x2 array, as [[x0, y0], [x1, y1] where the first pair is the
        left-bottom and the second pair is the right-top coordinate.
    """
    if ext.shape[0] == 4:
        ext = ext.reshape(2, 2)
    LB, RT = ext
    comp = np.logical_and(LB <= pnts, pnts <= RT)  # using <= and <=
    idx = np.logical_and(comp[..., 0], comp[..., 1])
    return pnts[idx]


def _pnts_in_extent_(pnts, ext=None, return_index=False):
    """Check, and return points within a defined extent.

    Parameters
    ----------
    pnts, ext : ndarray
        Nx2 arrays. `ext` needs a minimum shape of (2, 2).  If not,
        the left, bottom, top and right in that shape, then the extent is
        derived from the input points.
    return_index : boolean
        True, returns the indices in `pnts` that are inside the extent.
        False, returns whether at least one point is inside.

    >>> LB = np.min(a, axis=0)  # left bottom
    >>> RT = np.max(a, axis=0)  # right top
    >>> ext = np.asarray([LB, RT])
    """
    msg = "\nExtent in error... 2x2 array required not:\n{}\n"
    if ext is None:
        print(msg.format(ext))
        return
    shp = np.asarray(ext).shape
    if shp == (2, 2):
        LB, RT = ext
    elif shp[0] == 4:
        ext = ext.reshape(2, 2)
        LB, RT = np.min(ext, axis=0), np.max(ext, axis=0)
    else:
        print(msg.format(ext))
        return
    idx = np.all(np.logical_and(LB < pnts, pnts <= RT), axis=1)
    if return_index:
        return idx.base
    return np.all(idx)


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
    a = np.array(a)
    if a.ndim == 1:
        a = a.reshape(1, a.shape[0], 2)
        return np.array([i + [dx, dy] for i in a])
    else:
        return a + [dx, dy]


def _rot_(a, angle=0.0, clockwise=False):
    """Rotate shapes about their center or individually."""
    if clockwise:
        angle = -angle
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s, c)))
    cent = np.mean(np.unique(a, axis=0), axis=0)
    return np.einsum('ij,jk->ik', a - cent, R) + cent


def _scale_(a, factor=1):
    """Scale a geometry equally."""
    a = np.array(a)
    cent = np.min(a, axis=0)
    shift_orig = a - cent
    scaled = shift_orig * [factor, factor]
    return scaled + cent


# ---- Geo or ndarray stuff
#
def _rotate_(geo_arr, R, as_group):
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
    shapes = _bit_check_(geo_arr)
    out = []
    if as_group:  # ---- rotate as a whole
        cent = np.mean(geo_arr, axis=0)
        return np.einsum('ij,jk->ik', geo_arr - cent, R) + cent
    # ----
    uniqs = []
    for chunk in shapes:  # ---- rotate individually
        _, idx = np.unique(chunk, True, axis=0)
        uniqs.append(chunk[np.sort(idx)])
    cents = [np.mean(i, axis=0) for i in uniqs]
    for i, chunk in enumerate(shapes):
        ch = np.einsum('ij,jk->ik', chunk - cents[i], R) + cents[i]
        out.append(ch)
    return out


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
    """Return segment angles for Geo, object or ndarrays which represent
    polygons or polylines.

    See Also
    --------
    `angles_segment` in Geo class which is quicker for them.

    splitter = geo.To - np.arange(1, geo.N + 1)
    """
    a = _bit_check_(a, just_outer=False)
    ang = [_bit_segment_angles_(i, fromNorth) for i in a]
    return ang


# ---- `crossing` and related methods ----------------------------------------
# related functions
# See : line_crosses, in_out_crosses
#  pnt_right_side : single point relative to the line
#  line_crosses   : checks both segment points relative to the line
#  in_out_crosses # a variante of the above, with a different return signature

def line_crosses(p0, p1, p2, p3):
    """Determine if a line is `inside` another line segment.

    Parameters
    ----------
    p0, p1, p2, p3 : array-like
        X,Y coordinates of the subject (p0-->p1) and clipping (p2-->p3) lines.

    Returns
    -------
    The result indicates which points, if any, are on the inward bound side of
    a polygon (aka, right side). The clip edge (p2-->p3) is for clockwise
    oriented polygons and its segments. If `a` and `b` are True, then both are
    inside.  False for both means that they are on the outside of the clipping
    segment.
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3
    dc_x = x3 - x2
    dc_y = y3 - y2
    # ---- check p0 and p1 separately and return the result
    a = (y0 - y2) * dc_x <= (x0 - x2) * dc_y
    b = (y1 - y2) * dc_x <= (x1 - x2) * dc_y
    return a, b


def in_out_crosses(*args):
    """
    Determine if a line segment (p0-->p1) is crossed by a cutting/clipping
    segment (p2-->p3).  `inside` effectively means `right side` for clockwise
    oriented polygons.

    Parameters
    ----------
    p0p1, p2p3 : line segments
        Line segments with their identified start-end points, as below
    p0, p1, p2, p3 : array-like
        X,Y coordinates of the subject (p0-->p1) and clipping (p2-->p3) lines.

    Requires
    --------
    `_line_crosses_` method

    Returns
    -------
    - -1 both segment points are outside the clipping segment.
    - 0  the segment points cross the clipping segment with one point inside.
         and one point outside.
    - 1  both segment points are inside the clipping segment.

    """
    msg = "\nPass 2, 2-pnt lines or 4 points to the function\n"
    args = np.asarray(args)
    if np.size(args) == 8:
        if len(args) == 2:  # two lines
            p0, p1, p2, p3 = *args[0], *args[1]
        elif len(args) == 4:  # four points
            p0, p1, p2, p3 = args
        else:
            print(msg)
            return
    else:
        print(msg)
        return
    # ----
    a, b = line_crosses(p0, p1, p2, p3)
    if a and b:
        return 1
    elif a or b:
        return 0
    elif not a and not b:
        return -1


def crossings(geo, clipper):
    """Determine if lines cross. multiline implementation of above"""
    bounds = npg.dissolve(geo)  # **** need to fix dissolve
    p0s = bounds[:-1]
    p1s = bounds[1:]
    p2s = clipper[:-1]
    p3s = clipper[1:]
    n = len(p0s)
    m = len(p2s)
    in_ = []
    out_ = []
    crosses_ = []
    x_pnts = []
    for j in range(m):
        p2, p3 = p2s[j], p3s[j]
        for i in range(n):
            p0, p1 = p0s[i], p1s[i]
            ar = np.asarray([p0, p1, p2, p3])
            a, b = line_crosses(p0, p1, p2, p3)
            if a and b:
                # return 1
                in_.append(ar)
            elif a or b:
                # return 0
                crosses_.append(ar)
                x0 = npg._intersect_(p0, p1, p2, p3)
                print(p0, p1, p2, p3, x0)
                x_pnts.append(x0)
            # elif not a and not b:
            else:
                # return -1
                out_.append(ar)
    return in_, out_, crosses_, x_pnts


# ---- compare, remove, keep geometry ----------------------------------------
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
    w = np.where(np.equal(pnts, self).all(-1))[0]
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
    """Look for duplicates in two 2D arrays.  This can be points or segments.

    ** can use to find duplicates between 2 arrays ie compare_arrays **

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
    def _iterate_(N, n):
        """Return combinations for array lengths"""
        import itertools
        combos = itertools.combinations(np.arange(N), n)
        return combos

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


# ---- sort coordinates
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
    if hasattr(a, "IFT"):
        x_s = a.X
        y_s = a.Y
    else:
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


def radial_sort(a, close_poly=True, clockwise=True):
    """Return a radial sort of points around their center.

    Useful for polygons.  First and last point equality is checked.
    """
    uniq = np.unique(a, axis=0)
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
    """Weave an arrays to produce from-to pairs.  Returns a view

     >>> np.array(list(zip(arr[:-1], arr[1:])))  # returns a copy

    Parameters
    ----------
    a, b : ndarrays
        The array shapes must be the same.
    as_3d : boolean
        If True, an (N, 2, 2) shaped array is returned, otherwise the
        `from-to` values appear on the same line with a (N, 4) shape.
    """
    a = arr[:-1]
    b = arr[1:]
    fr_to = np.concatenate((a, b), axis=1)
    if as_3d:
        return fr_to.reshape(-1, 2, 2)  # for ndim=3d
    return fr_to


# ---- others ---------------------------------------------------------------
#
def shape_finder(arr, ids=None):
    """Provide the structure of an array/list which may be uneven and nested.

    Parameters
    ----------
    arr : array-like
        An array of objects. In this case points.
    ids : integer
        The object ID values for each shape. If ``None``, then values will be
        returned as a sequence from zero to the length of ``arr``.
    """
    main = []
    if ids is None:
        ids = np.arange(len(arr))
    if isinstance(arr, (list, tuple)):
        arr = np.asarray(arr, dtype='O')  # .squeeze()
    elif isinstance(arr, np.ndarray):
        shp = arr.shape
        if len(shp) > 1:
            arr = [arr]
    cnt = 0
    for i, a in enumerate(arr):
        info = []
        if hasattr(a, '__len__'):
            a0 = np.asarray(a, dtype='O')
            for j, a1 in enumerate(a0):
                if hasattr(a1, '__len__'):
                    a1 = np.asarray(a1, dtype='float')
                    if len(a1.shape) >= 2:
                        info.append([ids[i], j, *a1.shape])
                    else:  # a pair
                        info.append([ids[i], j, *a0.shape])
                        break
        main.append(np.asarray(info))
        cnt += 1
    return np.vstack(main)


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
