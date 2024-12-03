# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""---.

------------
npg_analysis
------------

Script :
    npg_analysis.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2023-11-03

Purpose
-------
Analysis tools for the Geom class.

Notes
-----
Functions use np.lib.recfunctions methods

References
----------
Derived from arraytools ``convex_hull, mst, near, n_spaced``

"""
# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121


# import sys
from textwrap import dedent
import numpy as np
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields
# from numpy.lib.recfunctions import structured_to_unstructured as stu

np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 7.3f}'.format})
np.ma.masked_print_option.set_display('-')  # change to a single -

__all__ = [
    'closest_n', 'distances', 'not_closer', 'n_check', 'n_near',
    'n_spaced', '_x_sect_2', 'intersection_pnt', 'knn', 'knn0',
    'mst', 'connect', 'concave'
]
__helpers__ = ['_dist_arr_', '_e_dist_']

# __all__ = __helpers__ + __all__


# ===========================================================================
# ---- (1) distance related
def closest_n(a, N=3, ordered=True):
    """See the `n_near` docstring."""
    coords, dist, n_array = n_near(a, N=N, ordered=ordered)
    return coords, dist, n_array


def distances(a, b):
    """Distances for 2D arrays using einsum.

    Based on a simplified version of e_dist in arraytools.
    """
    diff = a[:, None] - b
    return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))


def not_closer(a, min_d=1, ordered=False):
    """Find the points separated by a distance greater than min_d.

    This ensures a degree of point spacing.

    Parameters
    ----------
    a : coordinates
        2D array of coordinates.
    min_d : number
        Minimum separation distance.
     ordered : boolean
        Order the input points.

    Returns
    -------
    - b : points where the spacing condition is met
    - c : the boolean array indicating which of the input points were valid.
    - d : the distance matrix

    """
    if ordered:
        a = a[np.argsort(a[:, 0])]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = b - a
    d = np.einsum('ijk,ijk->ij', diff, diff)
    d = np.sqrt(d).squeeze()
    c = ~np.triu(d <= min_d, 1).any(0)
    b = a[c]
    return b, c, d


def n_check(a):  # N=3, order=True):
    """Run n_check prior to running n_near analysis.

    Parameters
    ----------
    Two 2D array of X,Y coordinates required.  Parse your data to comply.
    """
    if isinstance(a, (list, tuple, np.ndarray)):
        if (hasattr(a[0], '__len__')) and (len(a[0]) == 2):
            return True
        return False
    print(n_check.__doc__)
    return False


def n_near(a, N=3, ordered=True, return_all=False):
    """Nearest N point analysis.

    Return the coordinates and distances to the nearest ``N`` points in a
    2D numpy array, ``a``.  The results can be ordered if needed.

    Parameters
    ----------
    a : array
        An ndarray of uniform int or float dtype.  Extract the fields
        representing the x,y coordinates before proceeding.
    N : number
         Number of closest points to return.
    ordered : boolean
        True, return results sorted by distance.
    return_all : boolean
        True, returns coordinates, distance and combined array.  False, return
        a structured array containing the from-to coordinates and their
        distances.

    Returns
    -------
    A structured array is returned containing an ID number.  The ID number
    is the ID of the points as they were read.  The ID values are zero-based.
    The array will contain (C)losest fields and distance fields
    (C0_X, C0_Y, C1_X, C1_Y, Dist0, Dist1 etc) representing coordinates
    and distance to the required 'closest' points.
    """
    if not (isinstance(a, (np.ndarray)) and (N > 1)):
        print("\nInput error...read the docs\n\n{}".format(n_near.__doc__))
        return a
    rows, _ = a.shape
    dt_near = [('Xo', '<f8'), ('Yo', '<f8')]
    dt_new = [('C{}'.format(i) + '{}'.format(j), '<f8')
              for i in range(N)
              for j in ['_X', '_Y']]
    dt_near.extend(dt_new)
    dt_dist = [('Dist{}'.format(i), '<f8') for i in range(N)]
    # dt = [('ID', '<i4')]  + dt_near + dt_dist # python 2.7
    dt = [('ID', '<i4'), *dt_near, *dt_dist]
    n_array = np.zeros((rows,), dtype=dt)
    n_array['ID'] = np.arange(rows)
    # -- distance matrix calculation using einsum ----
    if ordered:
        a = a[np.argsort(a[:, 0])]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = b - a
    dist = np.einsum('ijk,ijk->ij', diff, diff)
    d = np.sqrt(dist).squeeze()
    # -- format for use in structured array output ----
    # steps are outlined as follows....
    #
    kv = np.argsort(d, axis=1)       # sort 'd' on last axis to get keys
    coords = a[kv]                   # pull out coordinates using the keys
    s0, s1, s2 = coords.shape
    coords = coords.reshape((s0, s1 * s2))
    dist = np.sort(d)[:, 1:]         # slice sorted distances, skip 1st
    # -- construct the structured array ----
    dt_names = n_array.dtype.names
    s0, s1, s2 = (1, (N + 1) * 2 + 1, len(dt_names))
    for i in range(0, s1):           # coordinate field names
        nm = dt_names[i + 1]
        n_array[nm] = coords[:, i]
    dist_names = dt_names[s1:s2]
    for i in range(N):               # fill n_array with the results
        nm = dist_names[i]
        n_array[nm] = dist[:, i]
    if return_all:
        return coords, dist, n_array
    return n_array


def n_spaced(L=0, B=0, R=10, T=10, min_space=1, num=10, verbose=True):
    """Produce `num` points within specified (L,B,R,T).

    Parameters
    ----------
    L(eft), B, R, T(op) : int, float
        Extent coordinates.
    min_space : number
        Minimum spacing between points.
    num : number
        Number of points... this value may not be reached if the extent
        is too small and the spacing is large relative to it.
    """
    #
    def _pnts(L, B, R, T, num):
        """Create the points."""
        xs = (R - L) * np.random.random_sample(size=num) + L
        ys = (T - B) * np.random.random_sample(size=num) + B
        return np.array(list(zip(xs, ys)))

    def _not_closer(a, min_space=1):
        """Find the points that are greater than min_space in the extent."""
        b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        diff = b - a
        dist = np.einsum('ijk,ijk->ij', diff, diff)
        dist_arr = np.sqrt(dist).squeeze()
        case_ = np.any(~np.triu(dist_arr <= min_space, 1), axis=0)
        return a[case_]
    #
    cnt = 1
    n = num * 2  # check double the number required as a check
    result = 0
    frmt = "Examined: {}  Found: {}  Need: {}"
    a0 = []
    while (result < num) and (cnt < 6):  # keep using random points
        a = _pnts(L, B, R, T, num)
        if cnt > 1:
            a = np.concatenate((a0, a), axis=0)  # np.vstack((a0, a))
        a0 = _not_closer(a, min_space)
        result = len(a0)
        if verbose:
            print(dedent(frmt).format(n, result, num))
        cnt += 1
        n += n
    # perform the final sample and calculation
    use = min(num, result)
    a0 = a0[:use]  # could use a0 = np.random.shuffle(a0)[:num]
    a0 = a0[np.argsort(a0[:, 0])]
    return a0


# ---- (2) intersection
#
def _x_sect_2(args):
    """Line intersection with extrapolation if needed.

    Inputs are two lines or 4 points.

    Example
    -------
    >>> s0 = np.array([[ 0.0, 0.0], [ 5.0, 5.0], [ 2.0,  0.0], [ 2.0, -2.0]])
    >>> s1 = np.array([[ 0.0, 0.0], [ 5.0, 5.0], [ 0.0,  2.5], [ 2.5, 0.0]])
    >>> a0, a1, a2, a3 = s0
    >>> b0, b1, b2, b3 = s1
    >>> npg._x_sect_2(s0)  # (False, None)
    >>> npg._x_sect_2(s1)  # (True, [1.25, 1.25])

    See npg.geom.intersect for details.  This is a variant for concave hulls.
    """
    if isinstance(args, np.ndarray):
        args = args.tolist()
    if len(args) == 2:
        p0, p1, p2, p3 = *args[0], *args[1]
    elif len(args) == 4:
        p0, p1, p2, p3 = args
    else:
        raise AttributeError("Pass 2, 2-pnt lines or 4 points to the function")
    #
    # -- First check
    # Given 4 points, if there are < 4 unique, then the segments intersect
    u, cnts = np.unique((p0, p1, p2, p3), return_counts=True, axis=0)
    if len(u) < 4:
        intersection_pnt = u[cnts > 1]
        return True, intersection_pnt

    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]
    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]
    #
    # -- Second check ----   np.cross(p1-p0, p3-p2)
    denom = s10_x * s32_y - s32_x * s10_y  # .item()
    if denom == 0.0:  # collinear
        return False, None
    #
    # -- Third check ----  np.cross(p1-p0, p0-p2)
    positive_denom = denom > 0.0  # denominator greater than zero
    s_numer = s10_x * s02_y - s10_y * s02_x  # .item()
    #
    # -- Fourth check ----  np.cross(p3-p2, p0-p2)
    t_numer = s32_x * s02_y - s32_y * s02_x
    #
    if positive_denom in (s_numer > denom, t_numer > denom):
        return False, None
    # if ((s_numer > denom) == positive_denom) or \
    #    ((t_numer > denom) == positive_denom):
    #     return False, None
    #
    # -- check to see if the intersection point is one of the input points
    # substitute p0 in the equation  These are the intersection points
    t = t_numer / denom
    intersection_point = [p0[0] + (t * s10_x), p0[1] + (t * s10_y)]
    return True, intersection_point


def intersection_pnt(p0, p1, p2, p3):
    """Return an intersection point.

    The intersection point is for a polygon segment (p0->p1) and a
    clipping polygon segment (p2->p3).

    References
    ----------
    `<https://en.wikipedia.org/wiki/Line–line_intersection>`_.
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = (*p0, *p1, *p2, *p3)
    dc_x, dc_y = np.subtract(p2, p3)
    dp_x, dp_y = np.subtract(p0, p1)
    n1 = x2 * y3 - y2 * x3
    n2 = x0 * y1 - y0 * x1
    n3 = 1.0 / (dc_x * dp_y - dc_y * dp_x)
    arr = np.array([(n1 * dp_x - n2 * dc_x), (n1 * dp_y - n2 * dc_y)])
    return arr * n3


# ---- (3) k-nearest neighbors
# knn0 used by concave hulls
#
def knn(p, pnts, k=1, return_dist=True):
    """Calculate the `k` nearest neighbours for a given point.

    Parameters
    ----------
    p : array
        x,y reference point.
    pnts : array
        Points array to examine.
    k : integer
        The `k` in k-nearest neighbours.

    Returns
    -------
    Array of k-nearest points and optionally their distance from the source.
    """
    def _remove_self_(p, pnts):
        """Remove a point which is duplicated or itself from the array."""
        keep = ~np.all(pnts == p, axis=1)
        return pnts[keep]

    def _e_2d_(p, a):
        """Return point to point distance for array (mini e_dist)."""
        diff = a - p[np.newaxis, :]
        return np.einsum('ij,ij->i', diff, diff)

    p = np.asarray(p)
    k = max(1, min(abs(int(k)), len(pnts)))
    pnts = _remove_self_(p, pnts)
    d = _e_2d_(p, pnts)
    idx = np.argsort(d)
    if return_dist:
        return pnts[idx][:k], d[idx][:k]
    return pnts[idx][:k]


def knn0(p, pnts, k):
    """Calculate the `k` nearest neighbours for a given point, `p`.

    Parameters
    ----------
    points : array
        List of points.
    p : array-like
        Reference point, two numbers representing x, y.
    k : integer
        Number of neighbours.

    Notes
    -----
    Removing the point ``p`` from the ``pnts`` list will take time, so trickery
    is used in argsort to just omit the first index since its distance will be
    zero.  Add 1 to the ``k`` value to ensure that you slice 3 points.

    Returns
    -------
    List of the k nearest neighbours, based on squared distance.
    """
    p = np.asarray(p)
    pnts = np.asarray(pnts)
    diff = pnts - p[np.newaxis, :]
    d = np.einsum('ij,ij->i', diff, diff)
    idx = np.argsort(d)[1:k + 1]  # Note remove the first point
    return pnts[idx].tolist()


# ---- (4) minimum spanning tree
#
def _dist_arr_(a, verbose=False):
    """Minimum spanning tree preparation."""
    idx = np.lexsort((a[:, 1], a[:, 0]))  # sort X, then Y
    # idx= np.lexsort((a[:, 0], a[:, 1]))  # sort Y, then X
    a_srt = a[idx, :]
    d = _e_dist_(a_srt)
    if verbose:
        frmt = """\n    {}\n    :Input array...\n    {}\n\n    :Sorted array...
        {}\n\n    :Distance...\n    {}
        """
        args = [_dist_arr_.__doc__, a, a_srt, d]  # d.astype('int')]
        print(dedent(frmt).format(*args))
    return d, idx, a_srt


def _e_dist_(a):
    """Return a 2D square-form euclidean distance matrix.

    For other dimensions, use e_dist in ein_geom.py.  For a 2D array, array
    shapes can be determined using either method below.  The
    shapes are as follows::

    >>> # a.shape => (N, 2)
    >>> b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])  # for any shape
    >>> b = a[:, None, :]  # faster for 2D
    >>> # b.shape => (N, 1, 2)
    """
    b = a[:, None, :]
    diff = a - b
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
    # d = np.triu(d)
    return d


def mst(arr, calc_dist=True):
    """Determine the minimum spanning tree for a set of points or weights.

    The spanning tree uses the inter-point distances as their `W`eights.

    Parameters
    ----------
    arr : array, normally an interpoint distance array
        Edge weights for example, distance, time, for a set of points.
        `arr` needs to be a square array or a np.triu perhaps.

    calc_dist : boolean
        True, if `arr` is a points array, the `arr` will be converted to the
        interpoint distance.
        False means that `arr` is not a points array, but some other `weight`
        representing the interpoint relationship.

    Returns
    -------
    pairs - the pair of nodes that form the edges.

    Example
    -------
    >>> a = np.array(
            [[0.4, 0.5], [1.2, 9.1], [1.2, 3.6], [1.9, 4.6], [2.9, 5.9],
             [4.2, 5.5], [4.3, 3.0], [5.1, 8.2] [5.3, 9.5], [5.5, 5.7],
             [6.1, 4.0], [6.5, 6.8], [7.1, 7.6], [7.3, 2.0], [7.4, 1.0],
             [7.7, 9.6], [8.5, 6.5], [9.0, 4.7], [9.6, 1.6], [9.7, 9.6]])
    """
    tmp, idx = np.unique(arr, return_index=True, axis=0)
    W = arr[idx]  # retain input order
    # W = arr[~np.isnan(arr[:, 0])]
    a_copy = np.copy(W)
    if calc_dist:
        W = _e_dist_(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError("W needs to be square matrix of edge weights")
    Np = W.shape[0]
    pairs = []
    pnts_seen = [0]  # Add the first point
    n_seen = 1
    # exclude self connections by assigning inf to the diagonal
    diag = np.arange(Np)
    W[diag, diag] = np.inf
    #
    while n_seen != Np:
        new_edge = np.argmin(W[pnts_seen], axis=None)
        new_edge = divmod(new_edge, Np)
        new_edge = [pnts_seen[new_edge[0]], new_edge[1]]
        pairs.append(new_edge)
        pnts_seen.append(new_edge[1])
        W[pnts_seen, new_edge[1]] = np.inf
        W[new_edge[1], pnts_seen] = np.inf
        n_seen += 1
    # pairs = np.array(pairs)
    pairs = np.vstack(pairs)
    frum = a_copy[pairs[:, 0]]
    too = a_copy[pairs[:, 1]]
    fr_to = np.concatenate((frum, too), axis=1)  # np.vstack(pairs)
#    fr_to_2 = uts(fr_to, names=['X_orig', 'Y_orig', 'X_dest', 'Y_dest'])
    return pairs, fr_to  # repack_fields(fr_to_2)


def connect(a, dist_arr, edges):
    """Return the full spanning tree, with points, connections and distance.

    Parameters
    ----------
    a : array
        A point array
    dist : array
        The distance array, from _e_dist
    edge : array
        The edges derived from mst
    """
    a = a[~np.isnan(a[:, 0])]
    p_f = edges[:, 0]
    p_t = edges[:, 1]
    d = dist_arr[p_f, p_t]
    n = p_f.shape[0]
    dt = [('Orig', '<i4'), ('Dest', 'i4'), ('Dist', '<f8')]
    out = np.zeros((n,), dtype=dt)
    out['Orig'] = p_f
    out['Dest'] = p_t
    out['Dist'] = d
    return out, p_f, p_t


"""
    a = np.array([[0, 0], [0, 8], [10, 8],  [10, 0], [3, 4], [7, 4]])
    a = np.unique(a, axis=0)
    idx= np.lexsort((a[:,1], a[:,0]))  # sort X, then Y
    a_srt = a[idx,:]                   # slice the sorted array
    d = _e_dist_(a_srt)                # determine the square form distances
    pairs, fr_to = mst(d)              # get the orig-dest pairs for the mst
    plot_mst(a_srt, pairs)             # a little plot
    o_d = connect(a_srt, d, pairs)     # produce an o-d structured array

"""


# ---- (5) concave hull
#
def concave(points, k, pip_check=False):
    """Return the concave hull for given points.

    Parameters
    ----------
    points : array-like
        Initially the input set of points with duplicates removes and
        sorted on the Y value first, lowest Y at the top (?).
    k : integer
        Initially the number of points to start forming the concave hull,
        `k` will be the initial set of neighbors.
    pip_check : boolean
        Whether to do the final point in polygon check.  Not needed for closely
        spaced dense point patterns.
    knn0, intersects, angle, point_in_polygon : functions
        Functions used by `concave`.

    Requires
    --------
    knn0 : function
        Performs the nearest neighbors search.

    Notes
    -----
    This recursively calls itself to check concave hull.

    p_set : The working copy of the input points

    70,000 points with final pop check removed, 1011 pnts on ch
        23.1 s ± 1.13 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
        2min 15s ± 2.69 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    PI = np.pi

    def _x_sect_(*args):
        """Line intersection check.  Two lines or 4 points that form the lines.

        Requires
        --------
        intersects(line0, line1) or intersects(p0, p1, p2, p3)
        - p0, p1 -> line 1
        - p2, p3 -> line 2

        Returns
        -------
        Boolean, if the segments do intersect.

        References
        ----------
        `<https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
        line-segments-intersect#565282>`_.

        """
        if len(args) == 2:
            p0, p1, p2, p3 = *args[0], *args[1]
        elif len(args) == 4:
            p0, p1, p2, p3 = args
        else:
            raise AttributeError("Use 2, 2-pnt lines or 4 points.")
        #
        # -- First check ----   np.cross(p1-p0, p3-p2 )
        p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = *p0, *p1, *p2, *p3
        s10_x = p1_x - p0_x
        s10_y = p1_y - p0_y
        s32_x = p3_x - p2_x
        s32_y = p3_y - p2_y
        denom = s10_x * s32_y - s32_x * s10_y
        if denom == 0.0:
            return False
        #
        # -- Second check ----  np.cross(p1-p0, p0-p2 )
        den_gt0 = denom > 0
        s02_x = p0_x - p2_x
        s02_y = p0_y - p2_y
        s_numer = s10_x * s02_y - s10_y * s02_x
        if (s_numer < 0) == den_gt0:
            return False
        #
        # -- Third check ----  np.cross(p3-p2, p0-p2)
        t_numer = s32_x * s02_y - s32_y * s02_x
        if (t_numer < 0) == den_gt0:
            return False
        #
        if den_gt0 in (s_numer > denom, t_numer > denom):
            return False
        # if ((s_numer > denom) == den_gt0) or ((t_numer > denom) == den_gt0):
        #     return False
        #
        # -- check if the intersection point is one of the input points
        t = t_numer / denom
        # substitute p0 in the equation
        x = p0_x + (t * s10_x)
        y = p0_y + (t * s10_y)
        # be careful that you are comparing tuples to tuples, lists to lists
        if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
            return False
        return True

    def _angle_(p0, p1, prv_ang=0):
        """Return the angle between two points and the previous angle, or."""
        ang = np.arctan2(p0[1] - p1[1], p0[0] - p1[0])
        a0 = ang - prv_ang
        a0 = a0 % (PI * 2) - PI
        return a0

    def _point_in_polygon_(pnt, poly):  # pnt_in_poly(pnt, poly):  #
        """Point in polygon check. ## fix this and use pip from arraytools."""
        x, y = pnt
        N = len(poly)
        for i in range(N):
            x0, y0, xy = [poly[i][0], poly[i][1], poly[(i + 1) % N]]
            c_min = min([x0, xy[0]])
            c_max = max([x0, xy[0]])
            if c_min < x <= c_max:
                p = y0 - xy[1]
                q = x0 - xy[0]
                y_cal = (x - x0) * p / q + y0
                if y_cal < y:
                    return True
        return False
    # --
    k = max(k, 3)  # Make sure k >= 3
    if isinstance(points, np.ndarray):  # Remove duplicates if not done already
        p_set = np.unique(points, axis=0).tolist()
    else:
        pts = []
        p_set = [pts.append(i) for i in points if i not in pts]  # Remove dupls
        p_set = np.array(p_set)
        del pts
    if len(p_set) < 3:
        raise Exception("p_set length cannot be smaller than 3")
    if len(p_set) == 3:
        return p_set  # Points are a polygon already
    k = min(k, len(p_set) - 1)  # Make sure k neighbours can be found
    frst_p = cur_p = min(p_set, key=lambda x: x[1])
    hull = [frst_p]       # Initialize hull with first point
    p_set.remove(frst_p)  # Remove first point from p_set
    prev_ang = 0
    # --
    while (cur_p != frst_p or len(hull) == 1) and len(p_set) != 0:
        if len(hull) == 3:
            p_set.append(frst_p)          # Add first point again
        # ---- Find nearest neighbours
        knn_pnts = knn0(cur_p, p_set, k)
        cur_pnts = sorted(knn_pnts, key=lambda x: -_angle_(x, cur_p, prev_ang))
        # --
        its = True
        i = -1
        while its and i < len(cur_pnts) - 1:
            i += 1
            last_point = 1 if cur_pnts[i] == frst_p else 0
            j = 1
            its = False
            while not its and j < len(hull) - last_point:
                its = _x_sect_(hull[-1], cur_pnts[i], hull[-j - 1], hull[-j])
                j += 1
        if its:  # All points intersect, try a higher number of neighbours
            return concave(points, k + 1)
        prev_ang = _angle_(cur_pnts[i], cur_p)
        cur_p = cur_pnts[i]
        hull.append(cur_p)  # Valid candidate was found
        p_set.remove(cur_p)
    if pip_check:
        for point in p_set:
            if not _point_in_polygon_(point, hull):
                return concave(points, k + 1)
    #
    hull = np.array(hull)
    return hull


# ---- demos, extras
def _demo():
    """Demonstration."""
    # L, R, B, T = [300000, 300100, 5025000, 5025100]
    L, B, R, T = [1, 1, 10, 10]
    tol = 1
    N = 10
    a = n_spaced(L, B, R, T, tol, num=N, verbose=True)
    return a


# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # print("")
