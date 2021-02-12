# Geo array helpers

There are several levels of information that can be acquired for Geo arrays.

The most basic is attached as an info tag to the array itself.  This property is proper case **Info**.
```python

g.Info
'rolled'
```

A fuller description describing the extent, the number of shapes, parts and points and a textual presentation of the Geo array's IFT information can be derived using
the lowercase **info** property.

```python

g.info
--------------
Extents :
  LL [ 300000.00  5000000.00]
  UR [ 300012.00  5000015.00]
Shapes :     7
Parts  :    10
Points :    55
Sp Ref : NAD 1983 CSRS MTM  9

  OID_    Fr_pnt    To_pnt    CW_CCW    Part_ID    Bit_ID  
----------------------------------------------------------------
 000      1         0         7         1          1         0
 001      1         7        11         0          1         1
 002      2        11        19         1          1         0
 003      3        19        25         1          1         0
 004      4        25        30         1          1         0
 005      5        30        36         1          1         0
 006      5        36        40         0          1         1
 007      6        40        46         1          1         0
 008      6        46        50         0          1         1
 009      9        50        55         1          1         0
```

The similarities and differences between the Geo array and the base ndarray is ascribed to a method.

All common properties can be determined using `npg.dirr` while the output from `npg.geo_info` is subdivided into the properties and methods specific to a geo array, their base and special properties.

```python

g.geo_info()

geo_info(geo_array)
Geo methods and properties.
    Bit, CW, FT, Fr, H, IDs, IFT, IFT_str, IP, Info, K, LL, N, PID, SR, SVG, To, U,
    UR, X, XT, XY, Y, Z, __author__, __dict__, __module__, __name__, aoi_extent,
    aoi_rectangle, areas, as_arrays, as_lists, bit_IFT, bit_ids, bit_pnt_cnt,
    bit_seq, bits, boundary, bounding_circles, centers, centroids, change_indices,
    close_polylines, common_segments, convex_hulls, densify_by_distance,
    densify_by_factor, densify_by_percent, dupl_pnts, extent_centers,
    extent_corner, extent_pnts, extent_rectangles, extents, fill_holes, first_bit,
    first_part, fr_to_pnts, geo_info, geom_check, get_shapes, holes_to_shape, info,
    inner_IFT, inner_rings, is_clockwise, is_convex, is_in, is_multipart,
    is_multipart_report, lengths, maxs, means, min_area_rect, mins, moveto,
    multipart_to_singlepart, od_pairs, outer_IFT, outer_rings, part_IFT, part_ids,
    parts, pnt_counts, pnt_ids, pnt_indices, pnt_on_poly, polygon_angles,
    polygons_to_polylines, polyline_angles, polylines_to_polygons, polys_to_points,
    prn, prn_obj, radial_sort, roll_shapes, rotate, segment_angles,
    segment_pnt_ids, segment_polys, shapes, shift, shp_IFT, shp_ids, shp_part_cnt,
    shp_pnt_cnt, shp_pnt_ids, sort_by_area, sort_by_extent, sort_by_length,
    sort_coords, split_by, structure, svg, to_segments, translate, triangulate,
    uniq_pnts, unique_segments, xy_id

Geo base properties.
    Bit, CW, FT, Fr, IDs, IFT, IP, Info, K, LL, N, PID, SR, SVG, To, U, UR, X, XT,
    XY, Y, Z

Geo special.
    __array_finalize__, __array_wrap__, __doc__, __new__
```
