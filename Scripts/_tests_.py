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
    2019-12-31

Purpose
-------
Tests for the Geo class.

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
from npg_arc import get_shapes, fc_nparray_Geo

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
def _radsrt_(a, dup_first=False):
    """Worker for radial sort.

    Notes
    -----
    See ``radial_sort`` for parameter details.

    Angles relative to the x-axis.
    >>> rad = np.arange(-6, 7.)*np.pi/6
    >>> np.degrees(rad)
    ... array([-180., -150., -120., -90., -60., -30.,  0.,
    ...          30., 60., 90., 120., 150., 180.])
    """
    uniq = np.unique(a, axis=0)
    cent = np.mean(uniq, axis=0)
    dxdy = uniq - cent
    angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])
    idx = angles.argsort()
    srted = uniq[idx]
    if dup_first:
        return np.concatenate((srted, [srted[0]]), axis=0)[::-1]
    return srted


def radial_sort(a, as_Geo=True, dup_first=False):
    """Sort the coordinates of polygon/polyline features.

    The features will be sorted so that their first coordinate is in the lower
    left quadrant (SW) as best as possible.  Outer rings are sorted clockwise
    and interior rings, counterclockwise.  Existing duplicates are removed to
    clean features, hence, the dup_first to provide closure for polygons.

    Parameters
    ----------
    a : Geo array
        The Geo array to sort.
    dup_first : boolean
        If True, the first point is duplicated.  This is needed for polygons
        and should be set to False otherwise.

    Returns
    -------
    Geo array, with points radially sorted (about their center).

    Notes
    -----
    See ``radial_sort`` for parameter details.

    Angles relative to the x-axis.
    >>> rad = np.arange(-6, 7.)*np.pi/6
    >>> np.degrees(rad)
    ... array([-180., -150., -120., -90., -60., -30.,  0.,
    ...          30., 60., 90., 120., 150., 180.])

    References
    ----------
    `<https://stackoverflow.com/questions/35606712/numpy-way-to-sort-out-a
    -messy-array-for-plotting>`_.
    """
    def _radsrt_(a, dup_first=False):
        """Worker for radial sort."""
        uniq = np.unique(a, axis=0)
        cent = np.mean(uniq, axis=0)
        dxdy = uniq - cent
        angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])
        idx = angles.argsort()
        srted = uniq[idx]
        if dup_first:
            return np.concatenate((srted, [srted[0]]), axis=0)[::-1]
        return srted
    # ----
    arrs = a.bits
    ift = a.IFT
    cw = a.CW
    kind = a.K
    if kind == 2 or dup_first is True:
        dup_first = True
    tmp = []
    for i, a in enumerate(arrs):
        arr = _radsrt_(a, dup_first)
        if cw[i] == 0:
            arr = arr[::-1]
        tmp.append(arr)
    out = npg.Geo(np.vstack(tmp), IFT=ift)
    return out

    # arrays_to_Geo(in_arrays, kind=2, info=None)
    # Split at opening in line
    # dx = np.diff(np.append(x, x[-1]))
    # dy = np.diff(np.append(y, y[-1]))
    # max_gap = np.abs(np.hypot(dx, dy)).argmax() + 1

    # x = np.append(x[max_gap:], x[:max_gap])
    # y = np.append(y[max_gap:], y[:max_gap])
    # return x, y


# ===========================================================================

# ----
# https://stackoverflow.com/questions/2158395/flatten-an-irregular-
# list-of-lists/2158532#2158532  # alfasin

def flat1(l):
    """Just does the basic flattening but doesn't yield where things are."""
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
# ---- demo

def _test_(in_fc=None, full=False):
    """Demo files listed in __main__ section.

    Usage
    -----
    in_fc, g = npg._tests_._test_()
    """
    if in_fc is None:
        in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Polygons2"
    kind = 2
    info = None
    SR = npg.get_SR(in_fc)
    shapes = npg.fc_shapes(in_fc)
    # ---- Do the work ----
    poly_arr = npg.poly2array(shapes)
    tmp, IFT = npg.fc_geometry(in_fc)
    m = np.nanmin(tmp, axis=0)
#    m = [300000., 5000000.]
    arr = tmp - m
    poly_arr = [(i - m) for p in poly_arr for i in p]
    arr = arr.round(3)
    g = npg.Geo(arr, IFT, kind, info)
    d = npg.fc_data(in_fc)
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
    return (np.diff(idx2) == 1).all() & (seq[idx2] == this).all()


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


"""
clockwise polygons
p = np.array([[ 50, 150], [100, 200], [100, 250], [150, 350], [200, 250],
                 [250, 300], [350, 300], [350, 150], [200,  50], [ 50, 150]])

c = np.array([[100, 100], [100, 300], [300, 300],
                    [300, 100], [100, 100]])

out = slip_poly(p, c)
array([[125.  , 100.  ],
       [100.  , 116.67],
       [100.  , 200.  ],
       [100.  , 200.  ],
       [100.  , 250.  ],
       [125.  , 300.  ],
       [175.  , 300.  ],
       [200.  , 250.  ],
       [250.  , 300.  ],
       [300.  , 300.  ],
       [300.  , 116.67],
       [275.  , 100.  ],
       [125.  , 100.  ]])
"""


# ----
'''
def _p_(sub):
   """print the sub results"""

s1 = sub[1]
for i, s in enumerate(s1):
    args = [s[0], s[0][1], s[i][0], s[i][1],
            s[i][2], s[i][3], [s[i][4: len(s[i])]]]
        prt_cnt = len(s)
        p_in_prt = s[i][0]
        b_in_prt = s[i][1]
        num_splits = s[i][2]
        splits = s[i][3]
        pnt_per_bit = [s[i][4:]]
        args = [i, prt_cnt, p_in_prt, b_in_prt, num_splits,
                splits, pnt_per_bit]
    print(frmt.format(*args))
return
'''


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

# ---- callers


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


# =========================================================================
# ---- (5) fc searchcursor to Geo


frmt = """
Shape {} len {}
         sub ID {}, len sub {}, num bits {}
         split ids {}
         points per bit {}
"""


def test(in_fc):
    """Test data"""
    a = fc_nparray_Geo(in_fc, geom_kind=2, info="")
    polys = get_shapes(in_fc)
    # a = npg_arc.fc_nparray_Geo(in_fc, geom_kind=2, info="")
    # polys = npg_arc.get_shapes(in_fc)
    return a, polys


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
        ['g.get_shape(ID=3, asGeo=True)', g.get_shape(ID=3, asGeo=True)],
        ['g.get_shape(ID=3, asGeo=True).IFT',
         g.get_shape(ID=3, asGeo=True).IFT],
        ['g.outer_rings(asGeo=True)', g.outer_rings(asGeo=True)],
        ['g.outer_rings(asGeo=True).IFT', g.outer_rings(asGeo=True).IFT],
        ['g.areas(True)', g.areas(True)],
        ['g.areas(False)', g.areas(False)],
        ['g.lengths(True)', g.lengths(True)],
        ['g.lengths(False)', g.lengths(False)],
        ['g.cent_shapes()', g.cent_shapes()],
        ['g.cent_parts()', g.cent_parts()],
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
        ['g.polygon_angles(inside=True, in_deg=True)',
         g.polygon_angles(inside=True, in_deg=True)],
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


"""
a = np.array([[10., 5.], [10., 0.], [0., 0.],
 [0., 10.], [10., 10.], [10., 5.]])
a2 = np.array([[10., 10], [10,0], [0,0], [0,10], [10,10]])
b = np.array([[[10, 5], [15, 10], [20, 5], [15, 0], [10, 5]], [[10, 5],
               [7.5, 7.5], [5, 5], [7.5, 2.5], [10, 5]]])
jb = [a], [a2], [b], [a, b ], [a2, b]

"""

r"""
extents = g.extents(by_bit=True)
pnts = np.array([[1., 1], [2.5, 2.5], [4.5, 4.5], [5., 5],
                  [6, 6], [10, 10], [12., 12], [12, 16]])
comp = np.logical_and(extents[:, 0:2] <= pnts[:, None],
                      pnts[:, None] <= extents[:, 2:4])
idx = np.logical_and(comp[..., 0], comp[..., 1])
idx.astype('int')
p_inside = [pnts[idx[:, i]] for i in range(idx.shape[-1])]
w0 = np.asarray(np.where(idx.astype('int'))).T    # col 0: pnt,   col 1: extent
w1 = np.asarray(np.where(idx.T.astype('int'))).T  # col 0: extent col 1: pnt

See Point_in_polygon.txt in my c:\Book_materials\Blogs folder

"""

# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
#    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb/fishnet"
#    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Ontario_LCConic"
#    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polylines"
#    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Dissolved"  #

#    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Wards_mtm9"
    # pth = r"C:/Git_Dan/npgeom/data/Polygons.geojson"
    # pth = r"C:/Git_Dan/npgeom/data/Oddities.geojson"
    # pth = r"C:/Git_Dan/npgeom/data/Ontario_LCConic.geojson"
    # in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/sample_10k"
    # in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Oddities"
    pnts = np.array([[1., 1], [2.5, 2.5], [4.5, 4.5], [5., 5],
                     [6, 6], [10, 10], [12., 12], [12, 16]])
    g, polys = test(in_fc)
    p0, p1, p2 = polys
    g0 = g.get_shape(1)

"""
    ccw = np.array([[10., 5.], [7.5, 7.5], [5., 5.], [7.5, 2.5], [10., 5.]])
    a0 = np.array([[10., 5.], [10., 0.], [0., 0.],
                   [0., 10.], [10., 10.], [10., 5.]])
    a1 = np.array([[10., 5.], [15., 10.], [20., 5.], [15., 0.], [10., 5.]])
    a2 = np.asarray([a0, ccw])
    a3 = np.asarray([a1, ccw])
    ccw2 = ccw + [6, 0]
    a4 = np.asarray([a1, ccw2])
    a5 = np.asarray([ccw2, a1])
    arrs = [a0, a1, a2, a3, a4, a5, ccw, ccw2]
    z = npg.arrays_to_Geo(arrs, 2, "arr = [a1, a2, b1, b2, ccw]")
    # b = np.array([[[10., 5.], [15., 10.], [20., 5.], [15., 0.], [10., 5.]],
    #              [[10., 5.], [7.5, 7.5], [5., 5.], [7.5, 2.5], [10., 5.]]])
    # np.array([[10., 10.], [10., 0.], [0., 0.], [0., 10.], [10., 10.]])
    # arr = [a1, a2, b1, b2, ccw]  # [a, b], [a1, b]]
"""

"""
NOTE

To reorder array elements

z = [0, 2, 1, 3, 5, 4]
np.choose(z[:3], g0.bits)

To take chunks out of arrays

z = [1, 3]
np.compress(z, g.bits)
"""
"""
heatmap = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
bboxes = np.array([
    [0, 0, 4, 7],
    [3, 4, 3, 4],
    [7, 2, 3, 7]
])

bbr = np.array([
    [0, 0, 7, 4],
    [4, 3, 4, 3],
    [2, 7, 7, 3]
])
s = heatmap[:7, :4]
fr = bbr[:, :2]
too = bbr[:, 2:]
https://stackoverflow.com/questions/59086169/is-there-a-way-to-slice-out-
multiple-2d-numpy-arrays-from-one-2d-numpy-array-in
"""
ifts = g0.IFT

