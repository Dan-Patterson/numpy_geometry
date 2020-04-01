# -*- coding: utf-8 -*-
"""
=================
_npgeom_notes_.py
=================

Script :
    _npgeom_notes_.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-12-24

Purpose :
    Notes for npgeom to try to reduce the header and help information in the
    other scripts

Results from _test_ using the following in_fc

>>> in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_LCConic"

Final array shape
>>> tmp.shape    # ---- (1053725, 2), 1,053,725 points
>>> len(shapes)  # ---- 589

%timeit npg.getSR(in_fc)
201 ms ± 2.29 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit npg.fc_shapes(in_fc)
824 ms ± 11.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit npg.poly2array(shapes)
17.5 s ± 189 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit tmp - m
12.3 ms ± 82.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# ---- this is what is used ----
%timeit npg.fc_geometry(in_fc)
18.6 s ± 109 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit [(i - m) for p in poly_arr for i in p]
9.23 ms ± 135 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit npg.Geo(a, IFT, kind, info)
21.9 µs ± 231 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

References :
"""

# ---- pylint section
# pylint: disable=C0103, C0302, C0415, E1136, E1121,
# pylint: disable=R0904, R0914, W0105, W0212, W0221
# pylint: disable=R0902

# ----
"""
**pylint issues**

pylint: disable=C0103  # invalid-name
pylint: disable=E1136  # self.base is unsubscriptable
pylint: disable=R0902  # Too many instance attributes
pylint: disable=R0904  # pylint issue
pylint: disable=R0914  # Too many local variables
pylint: disable=R1710  # inconsistent-return-statements
pylint: disable=W0105  # string statement has no effect
pylint: disable=W0201  # attribute defined outside __init__...
pylint: disable=W0621  # redefining name
pylint: disable=W0212  # Access to a protected member...
pylint: disable=W0221  #


currently use:

pylint: disable=C0103, C0302, C0415, E1136, E1121,
pylint: disable=R0904, R0914, W0105, W0212, W0221
pylint: disable=R0902
"""

"""
npGeo, __geodoc__ contents

    Parameters
    ----------
    **Required**
    arr : array-like
        A 2D array sequence of points with shape (N, 2).
    IFT : array-like
        Defines, the I(d)F(rom)T(o) and other structural elements that are
        present in polyline or polygon geometry that ``arr`` represents .
        Shape (N, 6) required.
    Kind : integer
        Points (0), polylines/lines (1) and polygons (2).
    Info : string (optional)
        Optional information if needed.

    **Derived**

    IDs : IFT[:, 0]
        Shape ids, the id number will be repeated for each part and hole in
        the shape
    Fr : IFT[:, 1]
        The ``from`` point in the point sequence.
    To : IFT[:, 2]
        The ``to`` point in the point sequence.
    CW : IFT[:, 3]
        A value of ``1`` for exterior rings, ``0`` for interior/holes.
    PID : IFT[:, 4]
        Part ids sequence by shape.  A singlepart shape will have one (1) part.
        Subsequent parts are numbered incrementally.
    Bit : IFT[:, 5]
        The bit sequence in a singlepart feature with holes and/or multipart
        features with or without holes
    FT : IFT[:, 1:3]
        The from-to ids together (Fr, To).
    IP : IFT[:, [0, 4]]
        Shape and part ids together (IDs, PID)
    N : integer
        The number of unique shapes.
    U : integer(s)
        A sequence of integers indicating the feature ID value.  There is no
        requirement for these to be sequential.
    SR : text
        Spatial reference name.
    X, Y, XY, Z: Derived from columns in the point array.
        X = arr[:, 0], Y = arr[:, 0], XY = arr[:, :2],
        Z = arr[:, 2] if defined
    XT : array
        An array/list of points identifying the lower-left and upper-right,
        of full extent of all the geometry objects.
    LL, UR : array
        The extent points as defined in XT.
    A featureclass with 3 shapes. The first two are multipart with holes.::
        arr.IFT
        array([[ 1,  0,  5,  1,  1,  0],
               [ 1,  5, 10,  0,  1,  1],
               [ 1, 10, 14,  0,  1,  2],
               [ 1, 14, 18,  0,  1,  3],
               [ 1, 18, 23,  1,  2,  0],
               [ 1, 23, 27,  0,  2,  1],
               [ 2, 27, 36,  1,  1,  0],
               [ 2, 36, 46,  1,  2,  0],
               [ 2, 46, 50,  0,  2,  1],
               [ 2, 50, 54,  0,  2,  2],
               [ 2, 54, 58,  0,  2,  3],
               [ 3, 58, 62,  1,  1,  0]], dtype=int32)
"""

# ---- Geo class and ndarray similarities/differences
"""
**Working with np.ndarray and Geo class**

To check the difference between the np.ndarray and Geo class, use...

geo_info(g)

Geo methods and properties
    FT, IDs, IFT, Info, K, N, X, XY, Y, Z, __dict__, __module__,
    aoi_extent, aoi_rectangle, areas, bit_IFT, bit_cnt, bit_ids, bits,
    bounding_circles, centers, centroids, change_indices, close_polylines,
    common_segments, convex_hulls, densify_by_distance,
    densify_by_percent, extent_centers, extent_rectangles, extents,
    fill_holes, first_bit, first_part, get, holes_to_shape, info,
    is_clockwise, is_convex, is_multipart, lengths, maxs, means,
    min_area_rect, mins, moveto_origin, multipart_to_singlepart, od_pairs,
    outer_rings, part_IFT, part_cnt, part_ids, parts, pnt_ids,
    pnt_on_poly, point_indices, point_info, polygon_angles,
    polygons_to_polylines, polyline_angles, polylines_to_polygons,
    polys_to_points, polys_to_segments, pull, rotate, shapes, shift,
    shp_IFT, shp_cnt, shp_ids, sort_by_area, sort_by_extent,
    sort_by_length, sort_coords, split_by, translate, triangulate,
    unique_segments

Geo.__dict_keys()
    __module__, __doc__, __new__, __array_finalize__, __array_wrap__,
    shapes, parts, bits, shp_IFT, part_IFT, bit_IFT, shp_ids, part_ids,
    bit_ids, pnt_ids, shp_cnt, part_cnt, bit_cnt, first_bit, first_part,
    get, pull, split_by, outer_rings, areas, lengths, centers, centroids,
    aoi_extent, aoi_rectangle, extents, extent_centers, extent_rectangles,
    maxs, mins, means, is_clockwise, is_convex, is_multipart,
    polyline_angles, polygon_angles, moveto_origin, shift, translate,
    rotate, bounding_circles, convex_hulls, min_area_rect, triangulate,
    fill_holes, holes_to_shape, multipart_to_singlepart, od_pairs,
    polylines_to_polygons, polygons_to_polylines, polys_to_points,
    close_polylines, densify_by_distance, densify_by_percent, pnt_on_poly,
    polys_to_segments, common_segments, unique_segments, change_indices,
    point_indices, sort_by_area, sort_by_length, sort_by_extent,
    sort_coords, info, point_info, __dict__

"""

# ---- geo class tests
"""
arrays

rectangle
a = np.array([[0., 0.], [0., 10.], [10., 10.], [10., 0.], [0., 0.]])

rectangle with hole
a0 = np.array([[[0.,  0.], [0., 10.], [10., 10.], [10.,  0.], [0.,  0.]],
              [[2., 2.], [8., 2.], [8., 8.], [2., 8.], [2., 2.]]])

just the hole reversed
a1 = a0[1][::-1]

----
x0, y1 = (a.T)[:, 1:]
x1, y0 = (a.T)[:, :-1]
e0 = np.einsum('...i,...i->...i', x0, y0)
e1 = np.einsum('...i,...i->...i', x1, y1)
t = e1 - e0
area = np.nansum((e0 - e1)*0.5)
x_c = np.nansum((x1 + x0) * t, axis=0) / (area * 6.0)
y_c = np.nansum((y1 + y0) * t, axis=0) / (area * 6.0)

t
Out[194]: array([ 0.00, -100.00, -100.00,  0.00])

area, x_c, y_c
Out[195]: (100.0, -5.0, -5.0)

----
x0, y1 = (a0.T)[:, 1:]
x1, y0 = (a0.T)[:, :-1]
e0 = np.einsum('...i,...i->...i', x0, y0)
e1 = np.einsum('...i,...i->...i', x1, y1)
t = e1 - e0
area = np.nansum((e0 - e1)*0.5)
x_c = np.nansum((x1 + x0) * t, axis=0) / (area * 6.0)
y_c = np.nansum((y1 + y0) * t, axis=0) / (area * 6.0)

t
array([[ 0.00, -12.00],
       [-100.00,  48.00],
       [-100.00,  48.00],
       [ 0.00, -12.00]])

area, x_c, y_c
Out[198]: (64.0, array([-7.81,  2.81]), array([-7.81,  2.81]))

----
x0, y1 = (a1.T)[:, 1:]
x1, y0 = (a1.T)[:, :-1]
e0 = np.einsum('...i,...i->...i', x0, y0)
e1 = np.einsum('...i,...i->...i', x1, y1)
t = e1 - e0
area = np.nansum((e0 - e1)*0.5)
x_c = np.nansum((x1 + x0) * t, axis=0) / (area * 6.0)
y_c = np.nansum((y1 + y0) * t, axis=0) / (area * 6.0)

t
Out[200]: array([ 12.00, -48.00, -48.00,  12.00])

area, x_c, y_c
Out[201]: (36.0, -5.0, -5.0)

"""

# ---- recfunctions section
"""
>>> dir(rfn)
... ["MaskedArray", "MaskedRecords", "__all__", "__builtins__", "__cached__",
...  "__doc__", "__file__", "__loader__", "__name__", "__package__",
...  "__spec__", ..., "_check_fill_value", ..., "_fix_defaults", "_fix_output",
...  "_get_fields_and_offsets", "_get_fieldspec", "_is_string_like",
...  "_izip_fields", "_izip_fields_flat", "_izip_records", ..., "_keep_fields",
...  "_zip_descr", "_zip_dtype", "absolute_import", "append_fields",
...  "apply_along_fields", ..., "assign_fields_by_name", "basestring",
...  "division", "drop_fields", "find_duplicates", "flatten_descr",
...  "get_fieldstructure", "get_names", "get_names_flat", ...,
...  "join_by", ..., "merge_arrays", ...,
...  "rec_append_fields", "rec_drop_fields", "rec_join", "recarray",
...  "recursive_fill_fields", "rename_fields", "repack_fields",
...  "require_fields", "stack_arrays", "structured_to_unstructured",
...  "suppress_warnings", ..., "unstructured_to_structured"]

"""

# ---- Timing tests on large data
"""
import npgeom as npg
import arcpy
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_LCConic"
SR = npg.getSR(in_fc)
c = arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR)

prts = [row[1].partCount for row in c]
c.reset()
pnts = [row[1].pointCount for row in c]
c.reset()
np.sum(pnts)   # 1,053,261

%timeit [row[1].partCount for row in c]
The slowest run took 9.50 times longer than the fastest.
This could mean that an intermediate result is being cached.
471 ns ± 585 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit [row[1].pointCount for row in c]
The slowest run took 8.50 times longer than the fastest.
This could mean that an intermediate result is being cached.
429 ns ± 520 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)


c0 = arcpy.da.SearchCursor(in_fc, ('SHAPE@XY'), None, SR)
c0 = arcpy.da.SearchCursor(in_fc, ('SHAPE@XY'), None, SR,
                           explode_to_points=True)
%timeit arcpy.da.SearchCursor(in_fc, ('SHAPE@XY'), None, SR)
5.06 ms ± 316 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

c1 = arcpy.da.SearchCursor(in_fc, ('SHAPE@X', 'SHAPE@Y'), None, SR)
%timeit arcpy.da.SearchCursor(in_fc, ('SHAPE@X', 'SHAPE@Y'), None, SR)
5.93 ms ± 1.12 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

------------------------
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
SR = npg.getSR(in_fc)
c1 = arcpy.da.SearchCursor(in_fc, ('SHAPE@X', 'SHAPE@Y'), None, SR)
g1 = c1._as_narray()
gxy = stu(g1)

----------------------------------
---- small set 12 shapes with 2 multipart shapes
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
SR = npg.getSR(in_fc)

%timeit _ptypes_(in_fc, SR)
8.89 ms ± 298 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
----
---- big set
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_LCConic"
SR = npg.getSR(in_fc)

%timeit _ptypes_(in_fc, SR)
20.9 s ± 198 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

589 shapes
 82 multipart shapes
507 singlepart shapes
1,053,261 points
1,197 parts

---- subset of singlepart shapes from Ontario_LCConic
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_singlepart"
SR = npg.getSR(in_fc)

%timeit _ptypes_(in_fc, SR)
15 s ± 247 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

507 singlepart shapes
748,874 points
"""

# ---- line magics
"""
line magics
-----------
%automagic  run this and you don't need the %
%cd      change directory
%conda
%lsmagic  list the available lint magics
%pprint    turn on/off pretty printing
%timeit
%%timeit
%pdoc      prints objects docstring
%quickref  provides a list of the line magics with more detail
%whos    somewhat detailed of variables

**looking at code**
np.hstack??    brings up the code
npg??   lists all of npgeom.__init__.py
"""

# ---- Time tests for Geo properties
"""
Time tests
----------
Properties

%timeit g.shapes
58.5 µs ± 3.26 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.parts
18.6 µs ± 664 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

%timeit g.bit_IFT
29.9 µs ± 669 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.bit_ids
30.5 µs ± 1.1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.bits
43 µs ± 3.86 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.part_cnt
16.4 µs ± 119 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

%timeit g.pnt_cnt
139 µs ± 9.83 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.areas(by_shape=True)
255 µs ± 5.01 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.areas(by_shape=False)
245 µs ± 6.81 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.centers
1.14 ms ± 32.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.centroids
870 µs ± 86.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.lengths
247 µs ± 37.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

Methods

%timeit g.aoi_extent()
17.8 µs ± 1.43 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.aoi_rectangle()
24.5 µs ± 2.6 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.extents(by_part=False)
226 µs ± 1.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.extents(by_part=True)
273 µs ± 31.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.extent_rectangles(by_part=False)
324 µs ± 32.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.extent_rectangles(by_part=True)
327 µs ± 49.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.maxs(by_part=False)
210 µs ± 24.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.maxs(by_part=True)
150 µs ± 16.1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.mins(by_part=False)
172 µs ± 1.91 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.mins(by_part=True)
158 µs ± 17.2 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.means(by_part=False, remove_dups=True)
662 µs ± 14.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.means(by_part=True, remove_dups=True)
892 µs ± 86.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.means(by_part=True, remove_dups=True)
702 µs ± 29.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.first_bit(asGeo=False)
724 µs ± 104 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.first_bit(asGeo=False)
520 µs ± 10.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.first_part(asGeo=False)
228 µs ± 9.12 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.first_part(asGeo=True)
269 µs ± 6.77 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.get(1, asGeo=False)
53.3 µs ± 4.45 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.get(1, asGeo=True)
67.4 µs ± 5.86 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.pull([1], asGeo=False)
22.6 µs ± 498 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.pull([1], asGeo=True)
80.4 µs ± 9.64 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.outer_rings(asGeo=False)
527 µs ± 14.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.outer_rings(asGeo=True)
567 µs ± 5.45 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.is_clockwise()
521 µs ± 50.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.is_convex()
1.4 ms ± 32.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.is_multipart(as_structured=False)
46.3 µs ± 6.13 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.is_multipart(as_structured=True)
70.1 µs ± 5.14 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.polygon_angles(inside=True, in_deg=True)
2.55 ms ± 517 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.moveto_origin()
36.2 µs ± 5.33 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.shift(dx=1, dy=1)
21.4 µs ± 2.78 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.translate(dx=1, dy=1)
22.7 µs ± 3.35 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit g.rotate(about_center=True, angle=22.5, clockwise=False)
1.19 ms ± 137 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit g.bounding_circles(angle=5, return_xyr=False)
7.38 ms ± 660 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.bounding_circles(angle=5, return_xyr=True)
8.05 ms ± 1.23 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.convex_hulls(by_part=False)
2.06 ms ± 29.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.convex_hulls(by_part=True)
2.08 ms ± 173 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.min_area_rect(as_structured=False)
8.5 ms ± 1.12 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.min_area_rect(as_structured=True)
9.22 ms ± 1.31 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.triangulate(as_polygon=False)
15.6 ms ± 353 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit g.triangulate(as_polygon=True)
15.3 ms ± 2.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

"""

""" ---- clipping example
from npgeom.npg_plots import plot_polygons
clipper = np.array([[0,0], [10,8], [20,0], [0,0]])  # triangle
outer = array([[10.,  10.], [10.,  0.], [ 1.5,  1.5], [ 0.,  10.], [10.,  10.])
inner = array([[ 3., 9.], [ 3., 3.], [ 9., 3.], [ 9., 9.], [ 3., 9.]])
poly = array([outer, inner], dtype=object)
cs = [clip_(p, clipper) for p in poly]
cs = [i for i in cs if i.size > 0]
c0 = npg.arrays_to_Geo(cs, 2, "")
plot_polygons(c0)
npg._svg(c0, True)
c0      x       y       dy/dx
Geo([[ 8.193,  8.000],
     [ 8.193,  0.000],  
     [ 0.000,  1.446],  # final connecting
     [ 8.193,  8.000],  # line.
     [ 1.943,  3.000],  # second segment
     [ 7.193,  3.000],
     [ 7.193,  7.200],  # overlapping
     [ 1.943,  3.000]]) # segment.
c0.IFT
array([[0, 0, 4, 1, 1, 0],
       [1, 4, 8, 0, 0, 0]], dtype=int64)
The inner ring and outer ring are super-imposed and the segment
"""


"""
np_arc.py notes
---------------

>>> dir(arcgisscripting.da)
['ContingentFieldValue', 'ContingentValue', 'DatabaseSequence', 'Describe',
 'Domain', 'Editor', 'ExtendTable', 'FeatureClassToNumPyArray', 'InsertCursor',
 'ListContingentValues', 'ListDatabaseSequences', 'ListDomains',
 'ListFieldConflictFilters', 'ListReplicas', 'ListSubtypes', 'ListVersions',
 'NumPyArrayToFeatureClass', 'NumPyArrayToTable', 'Replica', 'SearchCursor',
 'TableToNumPyArray', 'UpdateCursor', 'Version', 'Walk',
 '__doc__', '__loader__', '__name__', '__package__', '__spec__',
 '_internal_eq', '_internal_sd', '_internal_vb']


Time tests
----------

>>> a0 = fc_geo_Geo(in_fc)
>>> a1 = fc_sc_Geo(in_fc)
>>> a2 = fc_nparray_Geo(in_fc, kind=2, info="")

>>> %timeit fc_geo_Geo(in_fc)  # a0
3.28 s ± 194 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit fc_sc_Geo(in_fc)   # a1
3.7 s ± 349 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit fc_nparray_Geo(in_fc, kind=2, info="")  # a2
386 ms ± 93.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

"""

# ----
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
