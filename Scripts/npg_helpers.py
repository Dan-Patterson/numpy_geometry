# -*- coding: utf-8 -*-
r"""
-----------
npg_helpers
-----------

General helper functions.

----

Script :
    npg_helpers.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-01-09

Purpose
-------
Helper functions for npgeom.

Notes
-----
To suppress runtime warnings for errors that you know will happen

`<https://stackoverflow.com/questions/29950557/ignore-divide-by-0-warning-
in-numpy>`_.

Generally:  np.seterr(divide='ignore', invalid='ignore')
For a section:
    with np.errstate(divide='ignore', invalid='ignore'):
        # some code here

References
----------
None

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
# from textwrap import dedent

import numpy as np
import npgeom as npg

script = sys.argv[0]  # print this should you need to locate the script

nums = 'efdgFDGbBhHiIlLqQpP'

__all__ = [
    '_area_bit_', '_in_extent_', '_is_ccw_', '_is_clockwise_',
    '_is_right_side', '_length_bit_', '_rotate_', '_scale_', '_translate_',
    'compare_geom', 'crossings', 'in_out_crosses', 'interweave', 'is_Geo',
    'keep_geom', 'line_crosses', 'pnts_in_extent_', 'poly_cross_product_',
    'polyline_angles', 'radial_sort', 'remove_geom', 'sort_xy'
]  # 'crossing_num', 'pnts_in_poly'


# ---- common helpers ----
#
def _area_bit_(a):
    """Mini e_area, used by areas and centroids."""
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.sum((e0 - e1)*0.5)


def _length_bit_(ar):
    """Calculate segment lengths of poly geometry."""
    diff = ar[:-1] - ar[:-1]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def _is_clockwise_(a):
    """Return whether the sequence (polygon) is clockwise oriented or not."""
    if _area_bit_(a) > 0.:
        return True
    return False


def _is_ccw_(a):
    """Counterclockwise."""
    return 0 if _area_bit_(a) > 0. else 1


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
    Returns whether or not a point is inside (right-side) the current
    clip edge for a clockwise oriented polygon and its segments.

    position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))

    negative for right of clockwise line, positive for left. So in essence,
    the reverse of _is_left_side with the outcomes reversed ;)
    """
    x, y, x0, y0, x1, y1 = *p, *strt, *end
    return (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)


def is_Geo(obj, verbose=False):
    """Check the input to see if it is a Geo array."""
    if ('Geo' in str(type(obj))) & (issubclass(obj.__class__, np.ndarray)):
        return True
    if verbose:
        msg = "`{}`, is not a Geo array`. Use `arrays_toGeo` to convert."
        print(msg.format(obj.__class__))
    return False


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
    LB, RT = ext
    comp = np.logical_and(LB <= pnts, pnts <= RT)  # using <= and <=
    idx = np.logical_and(comp[..., 0], comp[..., 1])
    return pnts[idx]


def pnts_in_extent_(pnts, ext=None, return_index=False):
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
    elif shp[0] > 2:
        LB, RT = np.min(ext, axis=0), np.max(ext, axis=0)
    else:
        print(msg.format(ext))
        return
    idx = np.all(np.logical_and(LB < pnts, pnts <= RT), axis=1)
    if return_index:
        return idx
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


def _rotate_(a, angle=0.0, clockwise=False):
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


# ---- Geo array stuff
def polyline_angles(geo, fromNorth=False):
    """Polyline/segment angles.  *** needs work***."""
    ft = np.concatenate([np.concatenate((b[:-1], b[1:]), axis=1)
                         for b in geo.bits], axis=0)
    dxy = ft[1:] - ft[:-1]
    ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


def poly_cross_product_(a):
    """Cross product for poly features.

    Used by geom._angles_ with the optional ba, bc returns.  Kept for future
    reference.

    Parameters
    ----------
    a : ndarray
        The first and last points are checked in case

    Alternate::

        ba = a - np.roll(a, 1, 0)   # just as fastish as concatenate
        bc = a - np.roll(a, -1, 0)  # but definitely cleaner
    """
    dx, dy = a[0] - a[-1]
    if np.allclose(dx, dy):     # closed loop, remove duplicate
        a = a[:-1]
    ba = a - np.concatenate((a[-1, None], a[:-1]), axis=0)
    bc = a - np.concatenate((a[1:], a[0, None]), axis=0)
    return np.cross(ba, bc)  # ---  ba, bc


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
    bounds = npg.dissolve(geo)
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
def compare_geom(arr, look_for, unique=True, invert=False, return_idx=False):
    """Look for duplicates in two 2D arrays.  This can be points or segments.

    ** can use to find duplicates between 2 arrays ie compare_arrays **

    Parameters
    ----------
    arr : array, 2D
        The main array, preferably the larger of the two.
    look_for : array, 2D
        The array to compare with.

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
    result = (arr[:, None] == look_for).all(-1).any(-1)
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
    if hasattr(a, 'X') and hasattr(a, 'Y'):
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
    arr = np.asarray(arr).squeeze()
    cnt = 0
    for i, a in enumerate(arr):
        info = []
        if hasattr(a, '__len__'):
            a0 = np.asarray(a)
            for j, a1 in enumerate(a0):
                if hasattr(a1, '__len__'):
                    a1 = np.asarray(a1)
                    if len(a1.shape) >= 2:
                        info.append([ids[i], cnt, j, *a1.shape])
                    else:  # a pair
                        info.append([ids[i], cnt, j, *a0.shape])
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
