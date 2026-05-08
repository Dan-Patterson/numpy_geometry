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
  (008)  _bit_length_           _bit_min_max_          _clean_segments_       
  (009)  _rotate_               array_IFT              array_IFT_doc          
  (010)  arrays_to_Geo          bounding_circles_doc   check_geometry         
  (011)  clean_polygons         convex_hulls_doc       dedent                 
  (012)  dirr                   dirr_doc               extent_rectangles_doc  
  (013)  fill_float_array       geom                   geom_angles            
  (014)  get_shapes_doc         indent                 inner_rings_doc        
  (015)  inspect                is_Geo                 is_in_doc              
  (016)  np                     npg                    npg_geom_hlp           
  (017)  npg_io                 npg_prn                od_pairs_doc           
  (018)  outer_rings_doc        parts_doc              pnt_on_poly_doc        
  (019)  radial_sort_doc        reindex_shapes         remove_seq_dupl        
  (020)  repack_fields          roll_arrays            roll_coords            
  (021)  sc                     script                 shapes_doc             
  (022)  sort_by_area_doc       sort_by_extent_doc     sys                    
  (023)  uniq_1d                uts                    wrap                   

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
  (004)  __spec__               _update_docstring      array_IFT_doc          
  (005)  author_date            bounding_circles_doc   convex_hulls_doc       
  (006)  dirr_doc               extent_rectangles_doc  ft                     
  (007)  get_shapes_doc         inner_rings_doc        is_in_doc              
  (008)  np                     npGeo_doc              od_pairs_doc           
  (009)  outer_rings_doc        parts_doc              pnt_on_poly_doc        
  (010)  radial_sort_doc        script                 shapes_doc             
  (011)  sort_by_area_doc       sort_by_extent_doc     sys     
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
  (015)  tbl_data               view_poly
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
  (006)  _cut_pairs_            _del_seq_dupl_pnts_    _union_op_             
  (007)  add_intersections      adjacency_array        adjacency_matrix       
  (008)  append_                bail                   clip_                  
  (009)  concat_pairs           dissolve_geo           drop_seq_dupl          
  (010)  erase_                 flatten_list           fmt_                   
  (011)  intersect_check        merge_                 no_overlay_            
  (012)  np                     npGeo                  npg                    
  (013)  nx                     nx_solve               orient_clockwise       
  (014)  pair_                  pnt_connections        polygon_overlay        
  (015)  prepare                prn_                   prn_as_obj             
  (016)  renumber_pnts          reorder_x_pnts         rolling_match          
  (017)  rolling_pairer         script                 seg_connections        
  (018)  segment_classify       split_at_intersections sweep_srt              
  (019)  symm_diff_             sys                    turns                  
  (020)  union_adj              union_over             uts                    
  (021)  winding_num            wrap_
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
  (005)  _area_centroid_        _del_seq_dupl_pnts_    _in_extent_            
  (006)  _is_pnt_on_line_       _offset_np_            _side_                 
  (007)  _x_ings_               area_buffer            fmt_                   
  (008)  node_buffer            np                     npGeo                  
  (009)  np_wn                  offset_buffer          on_line_chk            
  (010)  plot_polygons          plot_polylines         plot_segments          
  (011)  rounded_buffer         script                 sys 
```

**npg.npg_clip_split**
```python
----------------------------------------------------------------------
| npg.dirr(npg.npg_clip_split) ...
|    npg.dirr...
-------
  (001)  __all__                __builtins__           __cached__             
  (002)  __doc__                __file__               __loader__             
  (003)  __name__               __package__            __spec__               
  (004)  _add_pnts_             _del_seq_dupl_pnts_    _wn_clip_              
  (005)  a_eq_b                 clip_poly              find_overlap_segments  
  (006)  fmt_                   np                     npg                    
  (007)  plot_polygons          prep_overlay           script                 
  (008)  split_poly             sys
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
  (004)  __package__            __spec__               _adj_within_           
  (005)  _angle_between_        _angles_from_north_    _angles_from_xaxis_    
  (006)  _area_centroid_        _bit_area_             _bit_check_            
  (007)  _bit_crossproduct_     _bit_length_           _bit_min_max_          
  (008)  _bit_segment_angles_   _clean_segments_       _from_to_pnts_         
  (009)  _get_base_             _in_LBRT_              _in_extent_            
  (010)  _is_ccw_               _is_clockwise_         _is_convex_            
  (011)  _is_turn               _od_angles_dist_       _perp_                 
  (012)  _pnts_in_extent_       _rotate_               _scale_                
  (013)  _trans_rot_            _translate_            a_eq_b                 
  (014)  classify_pnts          close_pnts             coerce2array           
  (015)  common_pnts            compare_geom           compare_segments       
  (016)  del_seq_dups           dist_angle_sort        flat                   
  (017)  geom_angles            interweave             keep_geom              
  (018)  multi_check            np                     nums                   
  (019)  pnt_segment_info       prn_tbl                radial_sort            
  (020)  reclass_ids            remove_geom            script                 
  (021)  segment_angles         shape_finder           sort_segment_pairs     
  (022)  sort_xy                swap_segment_pnts      sys                    
  (023)  uts
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
  (006)  _bit_min_max_          _ch_                   _ch_scipy_             
  (007)  _ch_simple_            _closest_pnt_on_poly_  _dist_along_           
  (008)  _e_2d_                 _get_base_             _is_pnt_on_line_       
  (009)  _percent_along_        _pnt_on_segment_       _view_as_struct_       
  (010)  bin_pnts               common_extent          densify_by_distance    
  (011)  densify_by_factor      dist_array             eucl_dist              
  (012)  extent_to_poly         find_closest           in_hole_check          
  (013)  mabr                   near_analysis          np                     
  (014)  npGeo                  np_wn                  npg_geom_hlp           
  (015)  npg_pip                on_line_chk            pnts_in_pnts           
  (016)  pnts_on_poly           pnts_to_extent         polys_to_segments      
  (017)  polys_to_unique_pnts   prn_q                  prn_tbl                
  (018)  repack_fields          script                 segments_to_polys      
  (019)  simplify               simplify_lines         spider_diagram         
  (020)  stu                    sys                    triangulate_pnts       
  (021)  uts                    which_quad             

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
  (006)  cartesian_product      flatten                np                     
  (007)  npg                    remove_seq_dupl        script                 
  (008)  separate_string_number sequences              stride_2d              
  (009)  sys                    uniq_1d                uniq_2d                
  (010)  unpack                 
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
  (011)  npg                    pnt_to_array_distances project_pnt_to_line    
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
  (005)  _line_crossing_        _p_ints_p_             _to_lists_             
  (006)  crossings              fmt_                   in_out_crosses         
  (007)  intersections          intersects             left_right_pnts        
  (008)  line_crosses           line_side              np                     
  (009)  script                 sys                    union_adj 
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
  (004)  __spec__               _is_right_side         _partition_            
  (005)  _side_                 crossing_num           fmt_                   
  (006)  np                     np_wn                  pnts_in_Geo            
  (007)  script                 sys                    winding_num
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
  (008)  make_row_format        n_h                    np                     
  (009)  prn_                   prn_Geo_shapes         prn_arrays             
  (010)  prn_as_obj             prn_geo                prn_lists              
  (011)  prn_q                  prn_tbl                script                 
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
  (013)  npg_arc_npg            npg_prn                rfn                    
  (014)  script                 split_sort_slice       struct2nd              
  (015)  sys                    u_case
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
  (012)  toolbox_info           wrap
```
