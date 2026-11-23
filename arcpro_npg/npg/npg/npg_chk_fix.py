# -*- coding: utf-8 -*-
# noqa: D205, D208, D400, F403
r"""
----------------------------------------------
npg_chk_fix: Functions for checking and fixing
----------------------------------------------

**Checking/fixing methods that work with Geo arrays or np.ndarrays.**

----

Script :
    npg_chk_fix.py

Author :
    Dan_Patterson

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2026-11-20
    
Purpose
-------
Checking/fixing methods that work with Geo arrays or np.ndarrays.
In the case of the former, the methods may be being called from Geo methods
in such things as a list comprehension.

See Also
--------

"""
# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121

# import sys
import numpy as np

# import npg
# from npg.npGeo import Geo_to_arrays, arrays_to_Geo
# from npg.npg_prn import prn_tbl

__all__ = [
    'clean_segments',                  # (2) check/fix functions
    'del_seq_dupl_pnts',
    'multipart_check',                 # (3) geometry functions
    'self_intersection_check'
]

__helpers__ = [                        # (1) general functions
    '_bit_crossproduct_',
    '_poly_area_'
]
    

# ----  ---------------------------
# ---- (1) general functions
#
def _bit_crossproduct_(a, is_closed=True, extras=False):
    """Cross product.

    Used by `is_convex`, `_angles_3pnt_` and `clean_segments`.

    .. note::

        np.cross for 2D arrays was deprecated in numpy 2.0, use `cross2d`
    """
    def cross2d(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    if hasattr(a, "IFT"):  # `npg_geom_hlp._base_` equivalent
        a = a.XY
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


def _poly_area_(a):
    """Mini e_area, used by `areas` and `centroids`.

    Negative areas are holes.  This is intentionally reversed from
    the `shoelace` formula.
    """
    x0, y1 = (a.T)[:, 1:]  # cross set up as follows
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)  # 2024-03-28 modified
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.sum((e0 - e1) * 0.5)

# ----  ---------------------------
# ---- (2) check/fix functions
#
def clean_segments(a, tol=1e-06):
    """Remove overlaps and extra points on poly* segments.

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


def del_seq_dupl_pnts(arr, poly=True):
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
    `npg_helpers.uniq_2d` which can be used in situations where genuine
    uniqueness is desired.
    """
    # -- like np.unique but not sorted
    arr = np.ascontiguousarray(arr)
    shp_in, dt_in = arr.shape, arr.dtype
    # arr = np.ascontiguousarray(arr)
    dt = [(f'f{i}', dt_in) for i in range(arr.shape[1])]
    tmp = arr.view(dt).squeeze()  # -- view data and reshape to (N,)
    # -- mask and check for sequential equality.
    mask = np.empty((shp_in[0],), np.bool_)
    mask[0] = True
    mask[1:] = tmp[:-1] != tmp[1:]
    # wh_ = np.nonzero(mask)[0]
    # sub_arrays = np.array_split(arr, wh_[wh_ > 0])
    tmp = arr[mask]  # -- slice the original array sequentially unique points
    if poly:  # -- polygon source check
        if (tmp[0] != tmp[-1]).any():  # any? all?
            arr = np.concatenate((tmp, tmp[0, None]), axis=0)
            return arr
    return tmp

# ---- ---------------------------
# ---- (3) geometry checks
#
def multipart_check(arr):
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


def self_intersection_check(arr):
    """Return the results of a self intersection check for a polygon.

    Parameters
    ----------
    a : array_like
        Nx2 array of x,y polygon coordinates.

    Example
    -------
    >>> crossed = np.array([[0., 0.], [5., 5.], [0., 5.], [5., 0.], [0., 0.]])
    >>> self_intersection_check(crossed)
    >>> [[0, 2, [2.5, 2.5]]]  # segments 0, 2 intersect at 2.5, 2.5
    >>> npg.npg_plots.plot_segments(a, title=None, label_segs=True)

    """
 
    def seg_intersection(p0, p1, p2, p3):
        """Return segment intersection ids and x, y values or None""" 
        x0, y0, x1, y1, x2, y2,x3, y3 = *p0, *p1, *p2, *p3
        denom = (x0 - x1) * (y2 - y3) - (y0 - y1) * (x2 - x3)
        if denom == 0:  # Parallel or collinear
            return None
        t = ((x0 - x2) * (y2 - y3) - (y0 - y2) * (x2 - x3)) / denom
        u = -((x0 - x1) * (y0 - y2) - (y0 - y1) * (x0 - x2)) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            return [x, y]
        return None  # Segments do not overlap
   
    a = np.concatenate((arr[:-1], arr[1:]), axis=1)
    N = len(a)
    out = []
    for i, v in enumerate(a[:-1]):
        frst = a[i]
        for j in range(i + 2, N-1):  # skip adjoining segment
            secn = a[j]
            p0, p1 = frst[:2], frst[-2:]
            p2, p3 = secn[:2], secn[-2:]
            chk = seg_intersection(p0, p1, p2, p3)
            if chk is not None:
                out.append([i, j, chk])
    return out
