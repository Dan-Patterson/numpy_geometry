# Geo array helpers

The differences in properties and methods between Geo arrays and ndarrays.

All common properties can be determined using `npg.dirr` while the output from `npg.geo_info` is subdivided into the properties and methods specific to a geo array, their base and special properties.

**npg.geo_info(g)**  # ---- g is a geo array

```
geo_info(geo_array)
Geo methods and properties.
    Bit, CW, FT, Fr, H, IDs, IFT, IFT_str, IP, Info, K, LL, N, PID, SR,
    SVG, To, U, UR, X, XT, XY, Y, Z, __author__, __dict__, __module__,
    __name__, angles_polygon, angles_polyline, aoi_extent, aoi_rectangle,
    areas, bit_IFT, bit_ids, bit_pnt_cnt, bit_seq, bits, boundary,
    bounding_circles, centers, centroids, change_indices, close_polylines,
    common_segments, convex_hulls, densify_by_distance,
    densify_by_percent, dupl_pnts, extent_centers, extent_corner,
    extent_pnts, extent_rectangles, extents, fill_holes, first_bit,
    first_part, geom_check, get_shape, hlp, holes_to_shape, info,
    inner_rings, is_clockwise, is_convex, is_in, is_multipart, lengths,
    maxs, means, min_area_rect, mins, moveto, multipart_to_singlepart,
    od_pairs, original_arrays, outer_rings, part_IFT, part_ids, parts,
    pnt_counts, pnt_ids, pnt_indices, pnt_on_poly, polygons_to_polylines,
    polylines_to_polygons, polys_to_points, polys_to_segments,
    pull_shapes, radial_sort, rotate, segment_pnt_ids, shapes, shift,
    shp_IFT, shp_ids, shp_part_cnt, shp_pnt_cnt, shp_pnt_ids,
    sort_by_area, sort_by_extent, sort_by_length, sort_coords, split_by,
    structure, translate, triangulate, uniq_pnts, unique_segments, xy_id

Geo base properties.
    Bit, CW, FT, Fr, IDs, IFT, IP, Info, K, LL, N, PID, SR, SVG, To, U,
    UR, X, XT, XY, Y, Z, hlp

Geo special.
    __array_finalize__, __array_wrap__, __doc__, __new__
```
