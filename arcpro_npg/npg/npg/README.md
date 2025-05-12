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
  (006)  _cut_pairs_            _del_seq_dupl_pnts_    _tri_chk_              
  (007)  _union_op_             add_intersections      adjacency_array        
  (008)  adjacency_matrix       append_                bail                   
  (009)  clp_                   erase_                 merge_                 
  (010)  no_overlay_            np                     npGeo                  
  (011)  npg                    nx                     nx_solve               
  (012)  one_overlay_           orient_clockwise       overlay_to_geo         
  (013)  pnt_connections        polygon_overlay        prepare                
  (014)  prn_                   prn_as_obj             renumber_pnts          
  (015)  reorder_x_pnts         roll_arrays            rolling_match          
  (016)  script                 split_at_intersections sweep                  
  (017)  symm_diff_             sys                    tri_array              
  (018)  triangle_check         turns                  union_adj              
  (019)  union_over             uts                    wrap_                  

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
  (004)  __package__            __spec__               _add_pnts_             
  (005)  _del_seq_dupl_pnts_    _node_type_            _roll_                 
  (006)  _seg_prep_             _w_                    _wn_clip_              
  (007)  add_intersections      find_sequence          np                     
  (008)  np_wn                  npg                    p_ints_p               
  (009)  plot_2d                plot_polygons          prep_overlay           
  (010)  roll_arrays            script                 segment_intersections  
  (011)  self_intersection_check sequences              sort_segment_pairs     
  (012)  swv                    sys                    

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
