# -*- coding: UTF-8 -*-
"""
========
pntinply
========

Script : pntinply.py

Author : Dan.Patterson@carleton.ca

Modified : 2019-03-08

Purpose
-------
Incarnations of point in polygon searches.  Includes, points in extent and
crossing number.

References
----------

`<http://geomalgorithms.com/a03-_inclusion.html>`_.

`<https://stackoverflow.com/questions/33051244/numpy-filter-points-within-
bounding-box/33051576#33051576>`_.

`<https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html>`_.  ** good

Notes
-----
Remove points that are outside of the polygon extent, then filter those
using the crossing number approach to test whether a point is within.

**Sample run**

>>> a, ext = array_demo()
>>> poly = extent_poly(ext)
>>> p0 = np.array([341999, 5021999])  # just outside
>>> p1 = np.mean(poly, axis=0)        # extent centroid
>>> pnts - 10,000 points within the full extent, 401 points within the polygon

(1) pnts_in_extent:

>>> %timeit pnts_in_extent(pnts, ext, in_out=False)
143 µs ± 2.16 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> %timeit pnts_in_extent(pnts, ext, in_out=True)
274 µs ± 9.12 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

(2) crossing_num with pnts_in_extent check (current version):

>>> %timeit crossing_num(pnts, poly)
9.68 ms ± 120 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

(3) pure crossing_num:

>>> %timeit crossing_num(pnts, poly)
369 ms ± 19.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ----10| ------20| ------30| ------40| ------50| ------60| ------70| ------80|
import numpy as np
# import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

__all__ = ['extent_poly',
           'pnts_in_extent',
           'p_in_ext',
           '_crossing_num_',
           'pnts_in_poly'
           ]


def extent_poly(ext):
    """Construct the extent rectangle from the extent points which are the
    lower left and upper right points [LB, RT]
    """
    LB, RT = ext
    L, B = LB
    R, T = RT
    return np.array([LB, [L, T], RT, [R, B], LB])


def pnts_in_extent(pnts, ext, in_out=True):
    """Point(s) in polygon test using numpy and logical_and to find points
    within a box/extent.

    Parameters:
    --------
    pnts : array
        an array of points, ndim at least 2D
    ext : numbers
        the extent of the rectangle being tested as an array of the left bottom
        (LB) and upper right (RT) coordinates
    in_out : boolean
       True, returns separate point arrays `inside, outside`.
       False returns `inside, None`.

    Notes
    -----
    comp :
        np.logical_and( great-eq LB, less RT)  condition check
    case :
        comp returns [True, False], so you take the product
    idx_in :
        indices derived using where since case will be 0 or 1
    inside :
        np.where(np.prod(comp, axis=1) == 1) if both true, product = 1
        then slice the pnts using idx_in

    """
    pnts = np.atleast_2d(pnts)  # account for single point
    outside = None
    LB, RT = ext
    comp = np.logical_and((LB < pnts), (pnts <= RT))
    idx_in = np.logical_and(comp[..., 0], comp[..., 1])
    inside = pnts[idx_in]
    if in_out:
        outside = pnts[~idx_in]  # invert case
    return inside, outside





def p_in_ext(pnts, ext):
    """Same as pnts_in_ext without the `outside` check and return.

    Example
    -------
    >>> ext = np.array([[400, 400], [600, 600]])
    >>> pnts = np.random.randint(0, `N`, size=(1000,2))

    time, `N` : 20 µs 1,000, 135 µs for 10,000, 640 µs for 50,000
    1.28 ms for 100,000
    """
    LB, RT = ext
    comp = np.logical_and(LB < pnts, pnts <= RT)
    idx = np.logical_and(comp[..., 0], comp[..., 1])
    return pnts[idx]


def _crossing_num_(pnts, poly, in_out=False):
    """Used by `pnts_in_poly`.  The implementation of pnply

    Parameters
    ----------
    pnts : array
        A numpy array of points (x,y pairs)
    poly : array
        Same as `pnts`, but a duplicate point representing the first and last
        to ensure close of the polygon.  Polygons are ordered clockwise for
        outer rings and counter-clockwise for inner rings.  The polygons are
        assumed to be single-part polgons.
    in_out : boolean
        True, retains both sets of points (inside and outside the polygon).
        False retains the outside points

    Returns
    -------
    The points inside the polygon
    Notes
    -----
    The functions, ``pnts_in_ext``, and  ``ext_poly`` are required

    Base logical check in parts
    >>> u = ys[i] <= y < ys[i+1]
    >>> d = ys[i] >= y > ys[i+1]
    >>> np.logical_or(u, d)
    """
    xs = poly[:, 0]
    ys = poly[:, 1]
    dx = np.diff(xs)
    dy = np.diff(ys)
    ext = np.array([poly.min(axis=0), poly.max(axis=0)])
    inside, outside = pnts_in_extent(pnts, ext, in_out=in_out)
    is_in = []
    for pnt in inside:
        cn = 0    # the crossing number counter
        x, y = pnt
        for i in range(len(poly)-1):      # edge from V[i] to V[i+1]
            if np.logical_or((ys[i] <= y < ys[i+1]), (ys[i] >= y > ys[i+1])):
                vt = (y - ys[i]) / dy[i]  # compute x-coordinate
                if x < (xs[i] + vt * dx[i]):
                    cn += 1
        is_in.append(cn % 2)  # either even or odd (0, 1)
    return inside[np.nonzero(is_in)], outside


def pnts_in_poly(pnts, poly, in_out=False):
    """Points in polygon, implemented using crossing number largely derived
    from **pnpoly** in its various incarnations.

    This version also does a within extent
    test to pre-process the points, keeping those within the extent to be
    passed on to the crossing number section.

    Parameters:
    ---------
    pnts : array
        point array
    poly : polygon
        closed-loop as an array.  The last and first point will be the same in
        a correctly formed polygon.

    Requires:
    ---------
    pnts_in_extent : function
        Method to limit the retained points to those within the polygon extent.
        See `pnts_in_extent` for details
    """
    inside, outside = _crossing_num_(pnts, poly, in_out=in_out)
    return inside, outside


def _demo():
    """ used in the testing
    : polygon layers
    : C:/Git_Dan/a_Data/testdata.gdb/Carp_5x5km  full 25 polygons
    : C:/Git_Dan/a_Data/testdata.gdb/subpoly     centre polygon with 'ext'
    : C:/Git_Dan/a_Data/testdata.gdb/centre_4    above, but split into 4
    """
    ext = np.array([[400, 400], [600, 600]])
    c = np.array([[0, 0], [0, 100], [100, 100], [100, 80],
                  [20, 80], [20, 20], [100, 20], [100, 0], [0, 0]])
    pnts = np.random.randint(0, 1000, size=(1000, 2))
    inside, outside = pnts_in_poly(pnts, poly=ext, in_out=True)
    return pnts, ext, c, inside, outside


if __name__ == "__main__":
    """Make some points for testing, create an extent, time each option.
    :
    :Profiling functions
    : %load_ext line_profiler
    : %lprun -f pnts_in_extent pnts_in_extent(pnts, ext)  # -f means function
    """
#    pnts, ext, poly, p0, p1 = _demo1()

# u = np.logical_and((y < ys[:-1, None]), (y < ys[1:, None]))
# d = np.logical_and((y >= ys[:-1, None]), (y > ys[1:, None]))
