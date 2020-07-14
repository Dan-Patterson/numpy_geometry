# Scripts
----

**This is way out of date**
I will update the documentation as soon as I can.

The following scripts are listed in this folder and the documentation guide

1. \_\_init__.py
2. npg_io.py
3. npg_arc_npg.py
4. npGeo.py
5. npg_helpers.py
6. smallest_circle.py

Links to other documentation will be provided as appropriate


----
## npg_io.py

Some useful functions to access and document featureclass information

**dtype_info(a, as_string=False)**

Return dtype information as lists or a string.
```
a = np.array([(0,  1,  2), (3,  4,  5), (6,  7,  8), (9, 10, 11)],
              dtype=[('f0', '<i4'), ('f1', '<i4'), ('f2', '<i4')])

npg.dtype_info(a, False)
(['f0', 'f1', 'f2'], ['<i4', '<i4', '<i4'])

npg.dtype_info(a, True)
Out[4]: ('f0, f1, f2', '<i4, <i4, <i4')
```

**load_geo(f_name, suppress_extras=True)**

Load saved arrays and supplemental information.

```
 geo = npg.load_geo(f_name, True)

Loading...C:/Git_Dan/npgeom/data/g_arr.npz
Arrays include...['g', 'ift', 'kind', 'extents', 'spatial_ref']
(0) name : g
  shape : (62, 2)
  descr. : [('', '<f8')]
(1) name : ift
  shape : (12, 6)
  descr. : [('', '<i4')]
(2) name : kind
  shape : ()
  descr. : [('', '<i4')]
(3) name : extents
  shape : (2, 2)
  descr. : [('', '<f8')]
(4) name : spatial_ref
  shape : ()
  descr. : [('', '<U19')]

geo, arrs, names = npg.load_geo(f_name, False)

Loading...C:/Git_Dan/npgeom/data/g_arr.npz
Arrays include...['g', 'ift', 'kind', 'extents', 'spatial_ref']
(0) name : g
  shape : (62, 2)
  descr. : [('', '<f8')]
(1) name : ift
  shape : (12, 6)
  descr. : [('', '<i4')]
(2) name : kind
  shape : ()
  descr. : [('', '<i4')]
(3) name : extents
  shape : (2, 2)
  descr. : [('', '<f8')]
(4) name : spatial_ref
  shape : ()
  descr. : [('', '<U19')]
```

**save_geo(g, f_name, folder)**

Save an array as an npz file.

**save_txt(a, name="arr.txt", sep=", ", dt_hdr=True)**

Save a NumPy structured/recarray to text.

**load_txt(name="arr.txt", data_type=None)**

Read a structured/recarray created by save_txt.  Many options are specified in save_txt.  If you wish to modify this, modify save_txt as well.

**Others**

The remainder of the functions deal with formatting attribute data and printing.

```
npg.npg_io
'_ckw_', '_col_format', 'col_hdr', 'dtype_info', 'geojson_Geo', 'gms', 'load_geo', 'load_geojson',
 'load_txt', 'make_row_format', 'prn_', 'prn_geo', 'prn_q', 'prn_tbl', 'save_geo', 'save_txt
```
----
## npg_arc_npg

<a href="url"><img src="https://github.com/Dan-Patterson/numpy_geometry/blob/master/images/npg_arc_npg.png" align="right" height="auto" width="150" ></a>

```
in_fc = 'C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2'
```
**get_SR(in_fc, verbose=False)**

```
getSR(in_fc, verbose=False)
<SpatialReference object at 0x2168428a1d0[0x216842398b0]>

getSR(in_fc, verbose=True)
SR name: NAD_1983_CSRS_MTM_9  factory code: 2951
```

**get_shape_K(in_fc)**

Returns the shape type for a featureclass as (kind, k), where kind is polygon, polyline, multipoint, point and variants.  k is 2, 1 or 0.

**fc_to_Geo(in_fc, geom_kind=2, minX=0, minY=0, info="")**

Convert FeatureClassToNumPyArray to a Geo array.  `in_fc` is the path to the featureclass.  `geom_kind` is either 1 or 2 representing polylines or polygons.

**id_fr_to(a, oids)**

Produce the `id`, `from` and `to points` used to delineate poly* bit geometry.

**Geo_to_shapes(geo, as_singlepart=True)**

Convert a geo array back to esri geometry objects.

**Geo_to_fc(geo, gdb=None, name=None, kind=None, SR=None)**

Produce a geodatabase featureclass from a geo array.

**Other functions**

There are a variety of other functions that deal with converting between geometries.
----
## npGeo.py

This is where the Geo class is housed along with methods an properties applicable to it.  The Geo class inherits from the numpy ndarray and methods applied to Geo arrays generally returns arrays of that class.

Geo arrays can be constructed from other ndarrays using **arrays_Geo**.  Three sample arrays are shown below.  They have been arranged in column format to save space.  

```
array(                  array([[[12., 18.], array([[14., 20.],
    [array([[10., 20.],         [12., 12.],        [10., 20.],
            [10., 10.],         [20., 12.],        [15., 28.],
            [ 0., 10.],         [20., 10.],        [14., 20.]])
            [ 0., 20.],         [10., 10.],
            [10., 20.],         [10., 20.],
            [ 3., 19.],         [20., 20.],
            [ 3., 13.],         [20., 18.],
            [ 9., 13.],         [12., 18.]],
            [ 9., 19.],        [[25., 24.],
            [ 3., 19.]]),       [25., 14.], 
     array([[ 8., 18.],         [15., 14.],
            [ 8., 14.],         [15., 16.],
            [ 4., 14.],         [23., 16.],
            [ 4., 18.],         [23., 22.],
            [ 8., 18.],         [15., 22.],
            [ 6., 17.],         [15., 24.],
            [ 5., 15.],         [25., 24.]]])
            [ 7., 15.],
            [ 6., 17.]])],
            dtype=object),
```

Both the array of arrays and the geo array are saved in the Scripts folder.
To load the Geo array, save the files to disk.  You can save and load arrays using the follow syntax.  This was used to create the files saved here.
```
# ---- For single arrays
np.save("c:/path_to_file/three_arrays.npy", z, allow_pickle=True, fix_imports=False)   # ---- save to disk

arr = np.load(c:/path_to_file/three_arrays.npy", allow_pickle=True, fix_imports=False) # ---- load above arrays

# ---- For multiple arrays
np.savez("c:/temp/geo_array.npz", s2=s2, IFT=s2.IFT)  # ---- save arrays, s2 and s2.IFT with names (s2, IFT)

npzfiles = np.load("c:/temp/geo_array.npz")  # ---- the Geo array and the array of I(ds)F(rom)T(o) values
npzfiles.files                               # ---- will show ==> ['s2', 'IFT']
s2 = npzfiles['s2']                          # ---- slice the arrays by name from the npz file to get each array
IFT = npzfiles['IFT']
```

----
OLD
----

**fc_info(in_fc, prn=True)**

```fc_info(in_fc, prn=True)

FeatureClass:
   C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2
shapeFieldName  OIDFieldName  shapeType spatialReference
Shape           OBJECTID      Polygon   NAD_1983_CSRS_MTM_9
```

**fc_fld_info(in_fc, prn=True)**
```
fc_fld_info(in_fc, prn=True)

FeatureClass:
   C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2
Name          Type         Length Nullable  Required  
OBJECTID      OID               4 False     True      
Shape         Geometry          0 True      True      
Shape_Length  Double            8 True      True      
Shape_Area    Double            8 True      True      
CENTROID_X    Double            8 True      False     
CENTROID_Y    Double            8 True      False     
INSIDE_X      Double            8 True      False     
INSIDE_Y      Double            8 True      False     
```

**fc_geom_info(in_fc, SR=None, prn=True, start=0, num=50)**
```
fc_geom_info(in_fc, SR=None, prn=True, start=0, num=50)

Featureclass:
    C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2
   Shape    Parts   Points From_pnt   To_pnt 
       1        2       21        0       21 
       2        2       18       21       39 
       3        1        4       39       43 
```

**fc_composition(in_fc, SR=None, prn=True, start=0, end=50)**
```fc_composition(in_fc, SR=None, prn=True, start=0, end=50)

C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2
Shapes :   3
Parts  :   5
  max  :   2
Points :   43
  min  :   4
  median : 9
  max  :   11
     IDs     Part   Points From_pnt   To_pnt 
       1        0       11        0       11 
       1        1       10       11       21 
       2        0        9       21       30 
       2        1        9       30       39 
       3        0        4       39       43 
```

**arr = tbl_arr(in_fc)**

```
arr = tbl_arr(in_fc)

array([(1,  86.47,  78., 300004.72, 5000014.73, 300004.72, 5000014.73),
       (2, 112.  , 104., 300017.5 , 5000017.  , 300015.  , 5000011.  ),
       (3,  21.5 ,  16., 300013.  , 5000022.67, 300013.  , 5000022.67)],
      dtype=[('OID_', '<i4'), ('Shape_Length', '<f8'), ('Shape_Area', '<f8'),
      ('CENTROID_X', '<f8'), ('CENTROID_Y', '<f8'),
      ('INSIDE_X', '<f8'), ('INSIDE_Y', '<f8')])
```


**prn_tbl(arr)**

```
prn_tbl(arr)

OID_    Shape_Length    Shape_Area    CENTROID_X    CENTROID_Y    INSIDE_X     INSIDE_Y    
------------------------------------------------------------------------------------------------
 000      1           86.47         78.00     300004.72    5000014.73    300004.72    5000014.73
 001      2          112.00        104.00     300017.50    5000017.00    300015.00    5000011.00
 002      3           21.50         16.00     300013.00    5000022.67    300013.00    5000022.67
 ```

The methods used to convert esri geometries to an appropriate array format.
This script will require you have a valid license for ArcGIS Pro since esri FeatureClasses in a File Geodatabase need to be read and their arcpy geometries extracted and converted to numpy arrays.

Numpy is part of the distribution for ArcGIS Pro, so no other dependencies need to be made to their conda packages.
If you wish to clone their distribution or modify the existing one, some guidance is provided here.

[Clone .... ArcGISPro 2.5](https://community.esri.com/blogs/dan_patterson/2020/02/09/clone-arcgis-pro-25)

[ArcGISPro 2.4 ... Installation, package updates and installs](https://community.esri.com/blogs/dan_patterson/2019/06/28/arcgis-pro-24-installation-package-updates-and-installs)

[Clone... ArcGIS Pro ... for non administrators](https://community.esri.com/blogs/dan_patterson/2018/12/28/clone)

<a href="url"><img src="https://github.com/Dan-Patterson/npGeo/blob/master/images/clones2.png" align="center" height="200" width="auto" ></a>
