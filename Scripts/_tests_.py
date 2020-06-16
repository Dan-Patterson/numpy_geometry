# -*- coding: utf-8 -*-
"""
-----------------------------------------------
Testing functions for the npgeom (npg) package.
-----------------------------------------------

File paths to data sources are listed at the end of the script.
There are two main test functions ``test(in_fc)`` and ``test2(g, kind=2)``
The first requires a featureclass and is run when this script is first run.
It produces ``g`` for use in ``test2``.

----
Script :
    _tests_.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-01-09

Purpose
-------
Tests for the Geo class.


ast.literal_eval(  *** find examples
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
from textwrap import dedent

# import importlib
import numpy as np
from numpy.lib.recfunctions import repack_fields
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts

if 'npg' not in list(locals().keys()):
    import npgeom as npg
    from npgeom.npGeo import *
from npg_arc_npg import get_SR, get_shapes, fc_to_Geo, poly2array, fc_data

# import npGeo
# from npGeo import fc2na_geo, id_fr_to, _fill_float_array
from arcpy.da import FeatureClassToNumPyArray  # SearchCursor

# importlib.reload(npg)
# importlib.reload(npg_arc)

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ===========================================================================
# ---- current problem

from npgeom.npg_plots import plot_polygons
from npgeom.npg_overlay import (
    _intersect_, intersects
)
from npgeom.npg_helpers import line_crosses, _in_extent_, polyline_angles
from npgeom.npg_geom import _angles_
r"""
directional buffer
point_move_x = vertex_coords_x -
    (distance) * math.cos(math.radians(direction))
point_move_y = vertex_coords_y -
    (distance) * math.cos(math.radians(90 - direction))

# Make list of points
new_line = ([[vertex_coords_x, vertex_coords_y], [point_move_x, point_move_y]])
lines_list.append(new_line)
"""

"""
diff = s00[:-1] - s00[1:]  #  coordinate difference around perimeter
sq_dist = np.einsum('ij,ij->i', diff, diff)
dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
array([ 8.6,  10.0,  10.0,  8.6])

same as

np.hypot(diff[:, 0], diff[:, 1])
array([ 8.6,  10.0,  10.0,  8.6])

segments = []
for i in range(len(coords) - 1):
    x1, y1, x2, y2 = coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]
    r = offset / math.hypot(x2 - x1, y2 - y1)
    vx, vy = (x2 - x1) * r, (y2 - y1) * r
    segments.append(((x1 - vy, y1 + vx), (x2 - vy, y2 + vx)))

# Set the resultant offset line to the input feature.
feature.setGeometry(fmeobjects.FMELine([segments[0][0]]
    + [intersection(s, t) for s, t in zip(segments[:-1], segments[1:])]
    + [segments[-1][1]]))

fr_to = np.concatenate((s00[:-1], s00[1:]), axis=1)  # produce from-to pairs
ft_pairs = fr_to.reshape(-1, 2, 2)



fr_to = np.concatenate((pnts0, pnts1), axis=1)
sze = (fr_to[:-1].size * 2)
e = np.zeros((sze), dtype=fr_to.dtype).reshape(-1, 4)
e[:-1][0::2] = fr_to[:-1]
e[1:][0::2] = fr_to[1] # for ndim=2
e.reshape(-1, 2, 2)  # for ndim=3


"""


"""  MST ....
connect shape points to follow the outline of a polygon
Thinking about using mst to do this... see npg_analysis, mst there


https://community.esri.com/blogs/dan_patterson/2017/01/31/spanning-trees


"""


# ---- common helpers
#
def polarToCartesian(centerX, centerY, radius, angleInDegrees):
    """Polar coordinate conversion"""
    angleInRadians = (angleInDegrees-90) * math.pi / 180.0
    x = centerX + (radius * math.cos(angleInRadians))
    y = centerY + (radius * math.sin(angleInRadians))
    return (x, y)


# ======
def angle_calculator(p0, cent, p1):
    """Angles given 3 points.

    A variety of things are returned.
    Angles are relative to the x-axis.  upper quadrants cover 180 to.
    Consider the points as listed in the order above.

    c_p0 : angle from cent to p0
    c_p1 : angle from cent to p1
    btwn_cp0_cp1 : angle formed between c_p0 and c_p1
    p0_c_p01_seq : sequential angle from p0==>cent==>p1
        outside/left inside/right
    """
    x_c, y_c = cent
    x_0, y_0 = p0
    x_1, y_1 = p1
    c_p0 = np.degrees(np.arctan2(y_0 - y_c, x_0 - x_c))
    c_p1 = np.degrees(np.arctan2(y_1 - y_c, x_1 - x_c))
    btwn_cp0_cp1 = (c_p0 - c_p1 + 180) % 360 -180  # signed angle
    return c_p0, c_p1, btwn_cp0_cp1


def buffer_circ(poly, buff_dist=1):
    """Buffer a polygon with circular ends.

    Stack the corner angles in clockwise order
    np.hstack((pnts0[1:], p0[1:], pnts1[:-1]))

    d = np.arange(0, 360, 10.)
    xys = [c[0] + (p[0]-c[0])* np.cos(1), c[1] + (c[1]-p[1])*np.sin(1)]

    ycal = c[1] + (p[1]-c[1])* np.cos(1) + (p[0]-c[0])*np.sin(1)
    """
    def intersection(p0, p1, p2, p3):
        """Line intersections."""
        x1, y1, x2, y2, x3, y3, x4, y4 = *p0, *p1, *p2, *p3
        dx1, dy1, dx2, dy2 = x2 - x1, y2 - y1, x4 - x3, y4 - y3
        a = x1 * y2 - x2 * y1
        b = x3 * y4 - x4 * y3
        c = dy1 * dx2 - dy2 * dx1
        if 1e-12 < abs(c):
            n1 = (a * dx2 - b * dx1) / c
            n2 = (a * dy2 - b * dy1) / c
            return (n1, n2)
        return (x2, y2)
    # -- ******* not done
    p0 = poly[:-1]
    p1 = poly[1:]
    diff = p1 - p0  # poly[1:] - poly[:-1]
    r = buff_dist/np.sqrt(np.einsum('ij,ij->i', diff, diff))
    vy_vx = (diff * r[:, None] * [1, -1])[:, ::-1]  # so adding for [1, -1]
    pnts0 = p0 + vy_vx
    pnts1 = p1 + vy_vx
    # fr_to = np.hstack((pnts0, pnts1)).reshape(-1, 2, 2)
    fr_to = np.concatenate((pnts0, pnts1), axis=1).reshape(-1, 2, 2)
    z = list(zip(fr_to[:-1], fr_to[1:]))
    z.append([z[-1][-1], z[0][0]])
    z = np.array(z)
    segs = [intersection(i[0], i[1], j[0], j[1]) for i, j in z]
    frst = np.atleast_2d(segs[-1])
    final = np.concatenate((frst, np.array(segs)), axis=0)
    return fr_to, z, final


def buff_(poly, buff_dist=1):
    """Offset line"""
    def intersection(p0, p1, p2, p3):
        """Line intersections."""
        x1, y1, x2, y2, x3, y3, x4, y4 = *p0, *p1, *p2, *p3
        dx1, dy1, dx2, dy2 = x2 - x1, y2 - y1, x4 - x3, y4 - y3
        a = x1 * y2 - x2 * y1
        b = x3 * y4 - x4 * y3
        c = dy1 * dx2 - dy2 * dx1
        if 1e-12 < abs(c):
            n1 = (a * dx2 - b * dx1) / c
            n2 = (a * dy2 - b * dy1) / c
            return (n1, n2)
        return (x2, y2)
    ft = []
    segs = []
    poly = np.array(poly)
    for i in range(poly.shape[0] - 1):
        x1, y1, x2, y2 = *poly[i], *poly[i+1]
        r = buff_dist / np.hypot(x2 - x1, y2 - y1)
        vx, vy = (x2 - x1) * r, (y2 - y1) * r
        pnt0 = (x1 - vy, y1 + vx)
        pnt1 = (x2 - vy, y2 + vx)
        ft.append([pnt0, pnt1])
    f_t = np.array(ft)
    z = list(zip(f_t[:-1], f_t[1:]))
    z.append([z[-1][-1], z[0][0]])
    z = np.array(z)
    for i, j in z:
        x_tion = intersection(i[0], i[1], j[0], j[1])
        segs.append(x_tion)  # np.array([i[0], middle]))
    frst = np.atleast_2d(segs[-1])
    final = np.concatenate((frst, np.array(segs)), axis=0)
    return f_t, z, final


# ===========================================================================
"""
# ----
# https://stackoverflow.com/questions/2158395/flatten-an-irregular-
# list-of-lists/2158532#2158532  # alfasin
"""


def flat1(l):
    """Just does the basic flattening but doesnt yield where things are."""
    def _flat(l, r):
        """Flatten sequence."""
        if not isinstance(l[0], (list, np.ndarray, tuple)):  # added [0]
            r.append(l)
        else:
            for i in l:
                r = r + flat(i)
        return r
    return _flat(l, [])


def flat(container, cnt=0, flat_list=None, sze_lst=None):
    """Flatten an object array, a list of lists, a list of arrays.

    Flattening is done using recursion.
    """
    if flat_list is None:
        sze_lst = ["cnt {}".format(cnt)]
        flat_list = []
    for item in container:
        if not isinstance(item, np.ndarray):
            item = np.asarray(item)
            cnt = cnt + 1
            if item.dtype.kind == 'O':
                flat(item, cnt, flat_list, sze_lst)
            else:
                flat_list.append(item)
                sze_lst.append([cnt, item])
        else:
            cnt = cnt + 1
            if item.dtype.kind == 'O':
                flat(item, cnt, flat_list, sze_lst)
            else:
                flat_list.append(item)
                sze_lst.append([cnt, item])
    return flat_list, sze_lst


# ===========================================================================
# ---- demos

# *** note   np.clip(np.clip(g, 0, 5), None, 10)

def perp(a):
    """Perpendicular to array"""
    b = np.empty_like(a)
    b_dim = b.ndim
    if b_dim == 1:
        b[0] = -a[1]
        b[1] = a[0]
    elif b_dim == 2:
        b[:, 0] = -a[:, 1]
        b[:, 1] = a[:, 0]
    return b

# ============================================================================


def stride_2d(a, win=(2, 2), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of an 2D array.  See arraytools.

    >>> np.asarray(list(zip(a[:-1], a[1:])))  # faster, but not a view
    """
    from numpy.lib.stride_tricks import as_strided
    shp = np.array(a.shape)    # array shape 2D (r, c) or 3D (d, r, c)
    win_shp = np.array(win)    # window         (3, 3) or    (1, 3, 3)
    ss = np.array(stepby)      # step by        (1, 1) or    (1, 1, 1)
    newshape = tuple(((shp - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides, subok=True)
    return a_s.squeeze()


def this_in_seq(this, seq):
    """Find a subset in a sequence.

    Parameters
    ----------
    seq, this : array
        Text or numbers. ``seq`` is the main array to search using ``this``
    """
    this = this.ravel()
    seq = seq.ravel()
    idx_srt = seq.argsort()
    idx = np.searchsorted(seq, this, sorter=idx_srt)
    idx[idx == len(seq)] = 0
    idx2 = idx_srt[idx]
    return idx2, (np.diff(idx2) == 1).all() & (seq[idx2] == this).all()


def this_in_seq2(this, seq):
    """Alternate to above, but about 3x slower"""
    n = len(this)
    s = stride_2d(seq, win=(n,), stepby=(1,))
    w = np.array([np.array_equal(i, this) for i in s])
    w = np.where(w)[0]
    out = []
    if w.size > 0:
        out = seq[w[0]:w[0] + n]
    return np.any(w), out


def iterate_nested_array(array, index=()):
    """Doc string"""
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError:  # final level
        yield (*index, slice(len(array))), array


def get_dimensions(arr, level=0):
    """Doc string"""
    yield level, len(arr)
    try:
        for row in arr:
            yield from get_dimensions(row, level + 1)
    except TypeError:  # not an iterable
        pass


def pad_fill_array(array, fill_value):
    """Create an array of the desired shape, fill value and dtype.

    Use ``get_max_shape`` to provide the dimensions

    shape : list tuple, array
        The desired array shape, see below.
    fill_value : int, float, string
        The value to fill the array with
    dtype : data type
        An acceptable data type.  For example 'int32', 'float64', 'U10'::

    shp = (2,3,2)
    b = np.full(shp, 0, dtype='int32')
    array([[[0, 0],
            [0, 0],
            [0, 0]],

           [[0, 0],
            [0, 0],
            [0, 0]]])
    a
    array([array([[1, 1],
           [2, 2]]),
           array([[3, 3],
           [4, 4],
           [5, 5]])], dtype=object)
    """
    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result


def get_max_shape(arr):
    """Get maximum dimensions in a possibly nested list/array structure

    Parameters
    ----------
    arr : list, tuple, ndarray
        Potentially nested data structure.

    Returns
    -------
    A variable length list with the format::

        [3, 2, 4, 10, 2]

    Representing the maximum expected value in each column::

        [ID, parts, bits, points, 2D]

    References
    ----------
    Marten Fabre, Code Review, 2019-06-20
    `Pad a ragged multidimensional array to rectangular shape
    <https://codereview.stackexchange.com/questions/222623/pad-a-ragged-
    multidimensional-array-to-rectangular-shape>`_.
    """
    from collections import defaultdict
    dimensions = defaultdict(int)
    for level, length in get_dimensions(arr):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


# --------------------------------------------------------------------------
def nested_shape(arr_like):
    """Provide the *maximum* nested depth.

    This is the length of each list at each depth of the nested list.
    (Designed to take uneven nested lists)
    """
    if not isinstance(arr_like[0], (np.ndarray, list, tuple)):
        return tuple([len(arr_like)])
    return tuple([len(arr_like)]) + max([nested_shape(i) for i in arr_like])


# ----------------------------------------------------------------------------
# ----
#
def tile_(a):
    """Produce a line sweep"""
    uniq_x = np.unique(a[:, 0])
    uniq_y = np.unique(a[:, 1])
    xs = uniq[:, 0]
    ys = uniq[:, 1]
    LB = np.min(uniq, axis=0)
    RT = np.max(uniq, axis=0)
    LR = np.array([LB[0], RT[0]])


# =========================================================================
# ---- (1) test functions

frmt = """
Shape {} len {}
         sub ID {}, len sub {}, num bits {}
         split ids {}
         points per bit {}
"""


def _test_(in_fc=None, full=False):
    """Demo files listed in __main__ section.

    Usage
    -----
    in_fc, g = npg._tests_._test_()
    """
    if in_fc is None:
        in_fc = r"C:/Git_Dan/npgeom/Project_npg/npgeom.gdb/Polygons2"
    kind = 2
    info = None
    SR = get_SR(in_fc)
    shapes = get_shapes(in_fc)
    # ---- Do the work ----
    poly_arr = poly2array(shapes)
    tmp = fc_to_Geo(in_fc)
    IFT = tmp.IFT
    m = np.nanmin(tmp, axis=0)
#    m = [300000., 5000000.]
    arr = tmp - m
    # poly_arr = [(i - m) for p in poly_arr for i in p]
    # arr = arr.round(3)
    g = npg.Geo(arr, IFT, kind, info)
    d = fc_data(in_fc)
    a = FeatureClassToNumPyArray(
        in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'], explode_to_points=True)
    a_id = a['OID@']
    g0xy = a[['SHAPE@X', 'SHAPE@Y']]
    g0xy['SHAPE@X'] = np.round(a['SHAPE@X'] - m[0], 3)
    g0xy['SHAPE@Y'] = np.round(a['SHAPE@Y'] - m[1], 3)
    a = repack_fields(g0xy)
#    g0xy = stu(g0xy)
#    g0xy = g0xy - m
    # g0xy = g0xy.round(3)
    frmt = """
    Type :  {}
    IFT  :
    {}
    """
    k_dict = {0: 'Points', 1: 'Polylines/lines', 2: 'Polygons'}
    print(dedent(frmt).format(k_dict[kind], IFT))
#    arr_poly_fc(a, p_type='POLYGON', gdb=gdb, fname='a_test', sr=SR, ids=ids)
#    a = np.array([[0.,  0.], [0., 10.], [10., 10.], [10.,  0.], [0.,  0.]])
#    a = npg.arrays_to_Geo(a, Kind=2, Info=None)
    if full:
        print("variables : SR, shapes, poly_arr, arr, IFT, g, g0xy, g0id")
        return SR, shapes, poly_arr, arr, IFT, g, d, a, a_id
    return in_fc, g, d


def test2(g, kind=2):
    """Run some standard tests"""
    np.set_printoptions(
        edgeitems=3, linewidth=80, precision=2, suppress=True,
        threshold=10, formatter=ft
    )
    header = """    \n
    ---------
    Geo array creation
    __new__(cls, g=None, IFT=None, Kind=2, Info="Geo array")
    required: g, IFT, Kind
    ---------
    g      the ndarray of xy coordinates
    g.IFT  Id, From, To - shape ID, from-to points in the shape\n    {}\n
    g.IDs  IFT[:, 0]   shape identifier\n    {}\n
    g.Fr   IFT[:, 1]   shape from point\n    {}\n
    g.To   IFT[:, 2]         to point\n    {}\n
    g.CW   IFT[:, 3]   shape orientation (C)lock(W)ise boolean result\n    {}\n
    g.PID  IFT[:, 4]   part identifier within a shape\n    {}\n
    g.Bit  IFT[:, 5]   bit identifier with a shape\n    {}\n
    g.FT   IFT[:, 1:3] from-to point pairs\n    {}\n
    g.K ... {} ...  shape kind 1, 2, 3 for points, polylines, polygons
    g.Info ... {} ... extra information string
    """

    msg0 = """    \n
    -------
    Derived
    -------
    - sample size, unique shapes
    g.N ... {} ...  len(uni)   #  uni, idx = np.unique(arr.IDs, True)
    g.U ... {} ...  g.IDs[idx]\n
    - coordinate values
    g.X    g[:, 0]\n    {}\n
    g.Y    g[:, 1]\n    {}\n
    g.XY   g[:, :2]\n    {}\n
    g.g[    not implemented yet
    - identifiers by shape, part, bit\n
    g.shp_IFT\n    {}\n
    g.part_IFT\n    {}\n
    g.bit_IFT\n    {}\n
    g.shp_ids  :{}
    g.part_ids :{}
    g.bit_ids  :{}
    g.bit_seq  :{}
    g.pnt_ids  :{}
    g.shp_pnt_cnt\n    {}\n
    g.shp_part_cnt\n    {}\n
    g.bit_pnt_cnt\n    {}\n
    g.shapes\n    {} ... snip\n    {}\n
    g.parts\n    {} ... snip\n    {}\n
    g.bits\n    {} ... snip\n    {}
    """

    gshapes = g.shapes
    gparts = g.parts
    gbits = g.bits
    props0 = [g.IFT, g.IDs, g.Fr, g.To, g.CW, g.PID, g.Bit, g.FT, g.K, g.Info]
    props1 = [
        g.N, g.U, g.X, g.Y, g.XY, g.shp_IFT, g.part_IFT, g.bit_IFT,
        g.shp_ids, g.part_ids, g.bit_ids, g.bit_seq, g.pnt_ids,
        g.shp_pnt_cnt, g.shp_part_cnt, g.bit_pnt_cnt,
        gshapes[0], gshapes[-1], gparts[0][0], gparts[-1], gbits[0], gbits[-1]
    ]

    meths = [
        ['g.first_bit(True)', g.first_bit(True)],
        ['g.first_bit(True).IFT', g.first_bit(True).IFT],
        ['g.first_part(True)', g.first_part(True)],
        ['g.first_part(True).IFT', g.first_part(True).IFT],
        ['g.get_shape(ID=1, asGeo=True)', g.get_shape(ID=1, asGeo=True)],
        ['g.get_shape(ID=3, asGeo=True).IFT',
         g.get_shape(ID=3, asGeo=True).IFT],
        ['g.outer_rings(asGeo=True)', g.outer_rings(asGeo=True)],
        ['g.outer_rings(asGeo=True).IFT', g.outer_rings(asGeo=True).IFT],
        ['g.areas(True)', g.areas(True)],
        ['g.areas(False)', g.areas(False)],
        ['g.lengths(True)', g.lengths(True)],
        ['g.lengths(False)', g.lengths(False)],
        ['g.centroids()', g.centroids()],
        ['g.aoi_extent()', g.aoi_extent()],
        ['g.aoi_rectangle()', g.aoi_rectangle()],
        ['g.extents(splitter="part")', g.extents(splitter="part")],
        ['g.extents(splitter="shape")', g.extents(splitter="shape")],
        ['g.extent_centers(splitter="part")',
         g.extent_centers(splitter="part")],
        ['g.extent_centers(splitter="shape")',
         g.extent_centers(splitter="shape")],
        ['g.extent_rectangles(splitter="shape")',
         g.extent_rectangles(splitter="shape")[:3]],
        ['g.extent_rectangles(splitter="part")',
         g.extent_rectangles(splitter="part")[:3]],
        ['g.maxs(by_bit=False)', g.maxs(by_bit=False)],
        ['g.maxs(by_bit=True)', g.maxs(by_bit=True)],
        ['g.is_clockwise(is_closed_polyline=False)',
         g.is_clockwise(is_closed_polyline=False)],
        ['g.is_convex()', g.is_convex()],
        ['g.is_multipart(as_structured=False)',
         g.is_multipart(as_structured=False)],
        ['g.angles_polygon(inside=True, in_deg=True)',
         g.angles_polygon(inside=True, in_deg=True)],
        ['g.bounding_circles(angle=5, return_xyr=False)',
         g.bounding_circles(angle=5, return_xyr=False)],
        ['g.min_area_rect(as_structured=False)',
         g.min_area_rect(as_structured=False)],
        ['g.triangulate(by_bit=False, as_polygon=True)',
         g.triangulate(by_bit=False, as_polygon=True)],
        ['g.fill_holes()', g.fill_holes()],
        ['g.holes_to_shape()', g.holes_to_shape()],
        ['g.multipart_to_singlepart(info="")',
         g.multipart_to_singlepart(info="")],
        ['g.od_pairs()', g.od_pairs()[0]],
        ['g.polygons_to_polylines()', g.polygons_to_polylines()],
        ['g.boundary()', g.boundary()],
        ['g.polys_to_points(keep_order=True, as_structured=False)',
         g.polys_to_points(keep_order=True, as_structured=False)],
        ['g.densify_by_distance(spacing=1)n',
         g.densify_by_distance(spacing=1)],
        ['g.densify_by_percent(percent=50)',
         g.densify_by_percent(percent=50)]
    ]
    print(dedent(header).format(*props0))
    print(dedent(msg0).format(*props1))
    print("""\n---------\nMethods.....\n---------""")
    for i, m in enumerate(meths, 1):
        try:
            print("\n({}) {} ...\n{}".format(i, *m))
        except ValueError:
            print("\n{} failed".format(m[0]))
        finally:
            pass


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
