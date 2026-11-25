# Scripts Folder

This is the repository for the NumPy geometry class and methods (aka *npg*).    

I use my own **npg.dirr** function to produce a formatted list of objects, their properties, methods and pretty well what python`s **dir** does.

Here is a partial listing of the code to the left with their **dir** contents.

**npgGeo**

```python
----------------------------------------------------------------------
| npg.dirr(npg.npGeo) ...
|    npg.dirr...
-------
  (001)  FLOATS                 Geo                    Geo_hlp                
  (002)  Geo_to_arrays          Geo_to_lists           INTS                   
  (003)  NUMS                   TwoPI                  __all__                
  (004)  __builtins__           __cached__             __doc__                
  (005)  __file__               __loader__             __name__               
  (006)  __package__            __spec__               _angles_3pnt_          
  (007)  _area_centroid_        _bit_area_             _bit_crossproduct_     
  (008)  _bit_length_           _bit_min_max_          _rotate_               
  (009)  array_IFT              array_IFT_doc          array_to_rings         
  (010)  arrays_to_Geo          bounding_circles_doc   check_geometry         
  (011)  clean_polygons         clean_segments         convex_hulls_doc       
  (012)  dedent                 dirr                   dirr_doc               
  (013)  extent_rectangles_doc  fill_float_array       fmt_                   
  (014)  geom                   geom_angles            get_shapes_doc         
  (015)  indent                 inner_rings_doc        inspect                
  (016)  is_Geo                 is_in_doc              np                     
  (017)  npg                    npg_bool_hlp           npg_bool_ops           
  (018)  npg_io                 npg_prn                od_pairs_doc           
  (019)  outer_rings_doc        parts_doc              pnt_on_poly_doc        
  (020)  radial_sort_doc        reindex_shapes         remove_seq_dupl        
  (021)  repack_fields          roll_arrays            roll_coords            
  (022)  sc                     script                 shapes_doc             
  (023)  sort_by_area_doc       sort_by_extent_doc     sys                    
  (024)  uniq_1d                uts                    wrap                      

```

**npgDocs**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npgDocs) ...
|    npg.dirr...
-------
  (001)  Geo_hlp                __all__                __builtins__           
  (002)  __cached__             __doc__                __file__               
  (003)  __loader__             __name__               __package__            
  (004)  __spec__               _update_docstring      add_intersections_doc  
  (005)  array_IFT_doc          author_date            bounding_circles_doc   
  (006)  convex_hulls_doc       dirr_doc               extent_rectangles_doc  
  (007)  fmt_                   get_shapes_doc         inner_rings_doc        
  (008)  is_in_doc              np                     npGeo_doc              
  (009)  od_pairs_doc           outer_rings_doc        parts_doc              
  (010)  pnt_on_poly_doc        prep_overlay_doc       radial_sort_doc        
  (011)  script                 shapes_doc             sort_by_area_doc       
  (012)  sort_by_extent_doc     sys     
```

**npg.npg_analysis**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_analysis) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __loader__             __name__               __package__            
  (004)  __spec__               _demo                  _dist_arr_             
  (005)  _e_dist_               _w_                    _x_sect_2              
  (006)  closest_n              concave                connect                
  (007)  dedent                 distances              fmt_                   
  (008)  intersection_pnt       knn                    knn0                   
  (009)  mst                    n_check                n_near                 
  (010)  n_spaced               nearest_polygon        not_closer             
  (011)  np
```      

**npg.npg_arc_npg**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_arc_npg) ...
|    npg.dirr...
-------
  (001)  Geo                    Geo_to_arc_shapes      Geo_to_fc              
  (002)  __all__                __builtins__           __cached__             
  (003)  __doc__                __file__               __loader__             
  (004)  __name__               __package__            __spec__               
  (005)  _array_to_poly_        _fc_as_narray_         _fc_geo_interface_     
  (006)  _fc_shapes_            _json_geom_            _poly_arr_             
  (007)  _poly_to_array_        _to_ndarray            array_poly             
  (008)  attr_to_npz            copy                   e                      
  (009)  fc2na                  fc_data                fc_dissolve            
  (010)  fc_to_Geo              fc_union               geometry_fc            
  (011)  get_SR                 get_shape_K            id_fr_to               
  (012)  make_nulls             np                     npGeo                  
  (013)  poly2array             repack_fields          script                 
  (014)  shp_dissolve           stu                    sys                    
  (015)  tbl_data               toolbox_info           view_poly 
```

**npg.npg_bool_hlp**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_bool_hlp) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __imports__            __loader__             __name__               
  (004)  __package__            __spec__               _add_intersections_    
  (005)  _add_pnts_             _del_seq_dupl_pnts_    _node_type_            
  (006)  _p_ints_p_             _roll_                 _seg_prep_             
  (007)  _w_                    _wn_clip_              add_intersections      
  (008)  add_intersections_doc  fmt_                   np                     
  (009)  np_wn                  npg                    plot_2d                
  (010)  plot_polygons          prep_overlay           prep_overlay_doc       
  (011)  script                 segment_intersections  self_intersection_check 
  (012)  sort_segment_pairs     sys 
```

**npg.npg_bool_ops**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_bool_ops) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __imports__            __loader__             __name__               
  (004)  __package__            __spec__               _adjacent_             
  (005)  _bit_area_             _bit_check_            _cut_across_           
  (006)  _cut_pairs_            _union_op_             add_intersections      
  (007)  adjacency_array        adjacency_matrix       append_                
  (008)  clip_                  closed_seg_check       concat_pairs           
  (009)  del_seq_dupl_pnts      dissolve_geo           drop_seq_dupl          
  (010)  erase_                 fmt_                   intersect_check        
  (011)  merge_                 np                     npGeo                  
  (012)  npg                    nx                     nx_solve               
  (013)  orient_clockwise       pair_                  pnt_connections        
  (014)  polygon_overlay        prepare                prn_                   
  (015)  prn_as_obj             renumber_pnts          reorder_x_pnts         
  (016)  rolling_match          rolling_pairer         script                 
  (017)  seg_connections        segment_classify       split_at_intersections 
  (018)  sweep_srt              symm_diff_             sys                    
  (019)  turns                  union_adj              union_over             
  (020)  uts                    winding_num            wrap_  
```

**npg.npg_buffer**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_buffer) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __imports__            __loader__             __name__               
  (004)  __package__            __spec__               _angles_3pnt_          
  (005)  _area_centroid_        _bit_crossproduct_     _in_extent_            
  (006)  _offset_np_            area_buffer            del_seq_dupl_pnts      
  (007)  fmt_                   node_buffer            np                     
  (008)  npGeo                  offset_buffer          plot_polygons          
  (009)  plot_polylines         plot_segments          rounded_buffer         
  (010)  script                 sys 
```


**npg.npg_create**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_create) ...
|    npg.dirr...
-------
  (001)  FLOATS                 INTS                   NUMS                   
  (002)  __all__                __builtins__           __cached__             
  (003)  __doc__                __file__               __helpers__            
  (004)  __loader__             __name__               __package__            
  (005)  __spec__               _arc_                  _test_data             
  (006)  _to_lists_             arc_                   arc_sector             
  (007)  arrays_to_Geo          base_spiral            buffer_rings           
  (008)  circ_3p                circ_3pa               circle                 
  (009)  circle_mini            circle_ring            circle_sectors         
  (010)  code_grid              ellipse                fmt_                   
  (011)  from_spiral            hex_flat               hex_pointy             
  (012)  merge_                 mesh_xy                mini_weave             
  (013)  np                     npg_plots              plot_mixed             
  (014)  plot_polygons          pnt_from_dist_bearing  pyramid                
  (015)  rectangle              repeat                 rot_matrix             
  (016)  script                 spiral_archim          spiral_ccw             
  (017)  spiral_cw              spiral_sqr             sys                    
  (018)  to_spiral              transect_lines         triangle               
  (019)  xy_grid
```

**npg.npg_geom_hlp**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_geom_hlp) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __imports__            __loader__             __name__               
  (004)  __package__            __spec__               _adjacent_within_      
  (005)  _angle_between_        _angles_from_north_    _angles_from_xaxis_    
  (006)  _area_centroid_        _area_scaler_          _base_                 
  (007)  _bit_area_             _bit_check_            _bit_crossproduct_     
  (008)  _bit_length_           _bit_min_max_          _bit_segment_angles_   
  (009)  _e_2d_                 _from_to_pnts_         _hole_area_            
  (010)  _in_LBRT_              _in_extent_            _is_ccw_               
  (011)  _is_clockwise_         _is_convex_            _is_turn               
  (012)  _od_angles_dist_       _orient_clockwise_     _perp_                 
  (013)  _pnts_in_extent_       _rotate_               _scale_                
  (014)  _trans_rot_            _translate_            a_eq_b                 
  (015)  classify_pnts          close_pnts             common_pnts            
  (016)  compare_geom           compare_segments       del_seq_dups           
  (017)  dist_angle_sort        fmt_                   geom_angles            
  (018)  keep_geom              np                     npg                    
  (019)  nums                   pnt_segment_info       radial_sort            
  (020)  reclass_ids            remove_geom            script                 
  (021)  segment_angles         sort_segment_pairs     sort_xy                
  (022)  swap_segment_pnts      sys                    uts
```

**npg.npg_geom_ops**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_geom_ops) ...
|    npg.dirr...
-------
  (001)  CH                     Delaunay               __all__                
  (002)  __builtins__           __cached__             __doc__                
  (003)  __file__               __helpers__            __imports__            
  (004)  __loader__             __name__               __package__            
  (005)  __spec__               _add_pnts_on_line_     _angles_3pnt_          
  (006)  _base_                 _bit_area_             _bit_min_max_          
  (007)  _ch_                   _ch_scipy_             _ch_simple_            
  (008)  _closest_pnt_on_poly_  _dist_along_           _e_2d_                 
  (009)  _is_pnt_on_line_       _percent_along_        _pnt_on_segment_       
  (010)  _view_as_struct_       bin_pnts               bins_to_poly           
  (011)  common_extent          densify_by_distance    densify_by_factor      
  (012)  dist_array             eucl_dist              extent_to_poly         
  (013)  find_closest           fmt_                   in_hole_check          
  (014)  mabr                   near_analysis          np                     
  (015)  npGeo                  np_wn                  npg_geom_hlp           
  (016)  npg_pip                pnts_in_pnts           pnts_on_poly           
  (017)  pnts_to_extent         polys_to_segments      polys_to_unique_pnts   
  (018)  prn_q                  prn_tbl                repack_fields          
  (019)  script                 segments_to_polys      simplify               
  (020)  simplify_lines         spider_diagram         stu                    
  (021)  sys                    triangulate_pnts       uts             

```

**npg.npg_helpers**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_helpers) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __loader__             __name__               __package__            
  (004)  __spec__               _base_                 _isin_2d_              
  (005)  _iterate_              _to_lists_             _view_as_struct_       
  (006)  as_strided             cartesian_product      coerce2array           
  (007)  drop_seq_dupl          find_sequence          flat                   
  (008)  flatten_list           interweave             np                     
  (009)  script                 separate_string_number sequences              
  (010)  stride_2d              swv                    sys                    
  (011)  uniq_1d                uniq_2d                unpack                 
  (012)  which_quad                 
```

**npg.npg_io**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_io) ...
|    npg.dirr...
-------
  (001)  FLOATS                 INTS                   NUMS                   
  (002)  __all__                __builtins__           __cached__             
  (003)  __doc__                __file__               __loader__             
  (004)  __name__               __package__            __spec__               
  (005)  dtype_info             geo_to_geojson         geojson_to_geo         
  (006)  get_keys               json                   len_check              
  (007)  lists_to_arrays        load_geo               load_geo_attr          
  (008)  load_geojson           load_txt               nested_len             
  (009)  np                     npGeo                  npg                    
  (010)  prn_keys               save_geo               save_txt               
  (011)  script                 sys 
```

**npg.npg_maths**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_maths) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __imports__            __loader__             __name__               
  (004)  __package__            __spec__               _angle_between_        
  (005)  _angles_3pnt_          _arc_mini_             _area_centroid_2       
  (006)  _offset_segment_       _pnt_on_segment_       _point_along_a_line    
  (007)  _resize_segment_       _trans_rot_2           circ_circ_intersection 
  (008)  cross_product_2d       dot_product_2d         flip_left_right        
  (009)  flip_up_down           line_circ_intersection n_largest              
  (010)  n_smallest             norm_2d                np                     
  (011)  on_line_chk            pnt_to_array_distances project_pnt_to_line    
  (012)  rot180                 rot270                 rot90                  
  (013)  running_count          script                 segment_crossing       
  (014)  sys
```

**npg.npg_min_circ**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_min_circ) ...
|    npg.dirr...
-------
  (001)  __builtins__           __cached__             __doc__                
  (002)  __file__               __loader__             __name__               
  (003)  __package__            __spec__               center                 
  (004)  circle_mini            distance               farthest               
  (005)  fmt_                   np                     small_circ             
  (006)  sub_1                  sub_2  

```

**npg.npg_overlay**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_overlay) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __loader__             __name__               __package__            
  (004)  __spec__               _in_LBRT_              _intersect_            
  (005)  _line_crossing_        _p_ints_p_             _seg_prep_             
  (006)  _to_lists_             crossings              fmt_                   
  (007)  in_out_crosses         intersections          intersects             
  (008)  left_right_pnts        line_crosses           line_side              
  (009)  np                     script                 segment_intersections  
  (010)  sort_segment_pairs     sys                    union_adj 
```

**npg.npg_pip**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_pip) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __helpers__            
  (003)  __loader__             __name__               __package__            
  (004)  __spec__               _is_right_side         _side_                 
  (005)  crossing_num           fmt_                   np                     
  (006)  np_wn                  partition              pip                    
  (007)  pip_test               pnts_in_Geo            pnts_on_segments       
  (008)  script                 sys                    winding_num
```

**npg.npg_plots**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_plots) ...
|    npg.dirr...
-------
  (001)  LineCollection         __all__                __builtins__           
  (002)  __cached__             __doc__                __file__               
  (003)  __helpers__            __loader__             __name__               
  (004)  __package__            __spec__               _demo                  
  (005)  _get_cmap_             axis_mins_maxs         fmt_                   
  (006)  matplotlib             np                     plot_2d                
  (007)  plot_3d                plot_buffs             plot_mesh              
  (008)  plot_mixed             plot_mst               plot_polygons          
  (009)  plot_polylines         plot_segments          plt                    
  (010)  scatter_params         script                 subplts                
  (011)  sys
```

**npg.npg_prn**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_prn) ...
|    npg.dirr...
-------
  (001)  FLOATS                 INTS                   NUMS                   
  (002)  __all__                __builtins__           __cached__             
  (003)  __doc__                __file__               __helpers__            
  (004)  __loader__             __name__               __package__            
  (005)  __spec__               _ckw_                  _col_format            
  (006)  _svg                   col_hdr                dedent                 
  (007)  fmt_                   gms                    indent                 
  (008)  make_row_format        np                     prn_                   
  (009)  prn_Geo_shapes         prn_arrays             prn_as_obj             
  (010)  prn_geo                prn_lists              prn_q                  
  (011)  prn_tbl                script                 shape_finder           
  (012)  stu                    sys                    uts 
```

**npg.npg_rand_data**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_rand_data) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __loader__             
  (003)  __name__               __package__            __spec__               
  (004)  _rand_case             _rand_float            _rand_int              
  (005)  _rand_str              _rand_text             colrow_txt             
  (006)  concat_flds            fmt_                   np                     
  (007)  npg                    pnts_IdShape           prn_                   
  (008)  prn_q                  prn_tbl                rfn                    
  (009)  rowcol_txt             script                 str_opt                
  (010)  strip_concatenate      sys
```

**npg.npg_table**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_table) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __loader__             
  (003)  __name__               __package__            __spec__               
  (004)  _append_fields         _as_pivot              _field_specs           
  (005)  _fill_fields           _get_numeric_fields    _prn                   
  (006)  calc_stats             col_stats              crosstab_array         
  (007)  crosstab_rc            crosstab_tbl           dedent                 
  (008)  find_a_in_b            find_in                flotsam                
  (009)  fmt_                   group_sort             group_stats            
  (010)  id_duplicates          keep_fields_by_kind    keep_fields_by_name    
  (011)  l_case                 merge_arrays           n_largest_vals         
  (012)  n_smallest_vals        nd2struct              np                     
  (013)  npg_prn                rfn                    script                 
  (014)  split_sort_slice       struct2nd              sys                    
  (015)  u_case
```

**npg.npg_utils**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_utils) ...
|    npg.dirr...
-------
  (001)  Path                   __all__                __builtins__           
  (002)  __cached__             __doc__                __file__               
  (003)  __helpers__            __loader__             __name__               
  (004)  __package__            __spec__               _utils_help_           
  (005)  _wrapper               dedent                 dir_py                 
  (006)  doc_deco               doc_func               env_list               
  (007)  find_def               fmt_                   folders                
  (008)  get_dirs               get_func               get_module_info        
  (009)  indent                 np                     os                     
  (010)  run_deco               script                 sub_folders            
  (011)  sys                    testing                time_deco              
  (012)  wrap 
```
