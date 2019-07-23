# NumPy and Geometry

A numpy geometry class and functions that work with arcpy and ESRI featureclasses is provided here to facilitate working with vector geometry using the ndarray as its base.

This is a work in progress, so bear with me.  The intent of the Geo class is to treat the geometry of featureclasses as one entity.  See **npGeo.py** in the Scripts section for details.  Converting esri's arcpy geometry objects to array representations is contained in **npGeo_io.py**. Most approaches I have see so far tend to construct the geometric representations of geometries using some variant of arcpy cursors.

When trying to work with numpy and the geometries, this creates problems since geometry is rarely a collection of simple shapes (eg. rectangles, circles, triangles).  Object arrays containing the coordinates are the norm.  An object array is created when the number of points per feature and/or feature part are not uniform.  For example, a square with a triangular hole in it, will have an outer ring, oriented clockwise consisting of a list of 5 points with the first and last point being the same.  The triangular hole will be represented by 4 points oriented counterclockwise.  Now that arrangement of points can be used to represent a polygon, a closed-loop polyline or a multipoint.  The same points can be used to represent 3 distinctly different geometric objects.

What I set out to do was create a uniform 2D array of coordinates with np.nan values separating the parts of a particular shape and a companion array which denotes the feature ID and the from-to point pairs.  This is similar to the FeatureClassToNumPy array approach, but that particular function and its inverse, are only useful for simple singlepart geometries.

I will document and build on these tools set with examples.  I am only working with featureclasses stored in a file geodatabase.

# Some links

[Geometry in NumPy](https://community.esri.com/blogs/dan_patterson/2019/03/17/geometry-in-numpy-1)

[Geometry, Arcpy and NumPy](https://community.esri.com/blogs/dan_patterson/2019/04/10/geometry-arcpy-and-numpy-2)

[Deconstructing poly* features](https://community.esri.com/blogs/dan_patterson/2019/04/10/geometry-deconstructing-poly-features-3)

[Reconstructing poly* features](https://community.esri.com/blogs/dan_patterson/2019/04/17/geometry-reconstructing-poly-features-4)

[Attributes.. the other bits](https://community.esri.com/blogs/dan_patterson/2019/04/17/geometry-attributes-actually-the-other-bits-5)

[Don't believe what you see](https://community.esri.com/blogs/dan_patterson/2019/05/09/geometry-dont-believe-what-you-see-6)

[Geometry: forms of the same feature](https://community.esri.com/blogs/dan_patterson/2019/05/13/geometry-forms-of-the-same-feature-7)

[Links to the script documentation](https://github.com/Dan-Patterson/npGeo/blob/master/Scripts/README.md)


----
# Multipart shapes

<a href="url"><img src="images/Shape2.png" align="right" height="300" width="auto" ></a>

Consider the following multipart shapes.  The first shape has is second part slightly offset and it also contains a hole.  The second shape is a flip/mirror/translate of its first part.

The centroids of each part are shown on the image.  These locations have been confirmed using arcpy and npGeo methods.

The point coordinates with (300,000 m, 5,000,000 m, MTM 9) subtracted from their values.  So the data are in a projected coordinate system and all further measures will be in planar/metric units.



```
pnt shape  part  X       Y     
--------------------------------
 000     0         10.00   20.00 
 001     0         10.00   10.00
 002     0          0.00   10.00
 003     0          0.00   20.00
 004     0         10.00   20.00
 005     0   x       nan     nan  ---- the null point separating the inner and outer rings of the first shape
 006     0          3.00   19.00
 007     0          3.00   13.00
 008     0          9.00   13.00
 009     0          9.00   19.00
 010     0          3.00   19.00
 011     0   o      8.00   18.00  ---- Start of the 2nd part of the first shape
 012     0          8.00   14.00
 013     0          4.00   14.00
 014     0          4.00   18.00
 015     0          8.00   18.00
 016     0   x       nan     nan  ---- the null point, separating the inner and outer rings of the 2nd part
 017     0          6.00   17.00        of the first shape
 018     0          5.00   15.00
 019     0          7.00   15.00
 020     0  ___     6.00   17.00
 021     1   o     12.00   18.00  ---- the 2nd shape begins, its first part
 022     1         12.00   12.00
 023     1         20.00   12.00
 024     1         20.00   10.00
 025     1         10.00   10.00
 026     1         10.00   20.00
 027     1         20.00   20.00
 028     1         20.00   18.00
 029     1         12.00   18.00
 030     1   o     25.00   24.00  ---- the 2nd part of the 2nd shape
 031     1         25.00   14.00
 032     1         15.00   14.00
 033     1         15.00   16.00
 034     1         23.00   16.00
 035     1         23.00   22.00
 036     1         15.00   22.00
 037     1         15.00   24.00
 038     1         25.00   24.00
``` 
 
This shape (s2) is simply represented by the last 2 columns, the first 2 columns are solely for printing purposes.
The sequence of points is identified by their Id and From and To points (IFT)

```
s2.IFT 
array([[ 0,  0, 11],    1st shape, 1st part, points 0 to 11 (but not including 11,
       [ 0, 11, 21],    1st shape, 2nd part                  following array slicing format)
       [ 1, 21, 30],    2nd shape, 1st part
       [ 1, 30, 39]])   2nd shape, 2nd part
```
I added another method to the pack to expand upon the IFT information. 
```
prn_tbl(s2.info(False))

....  IDs    Part    Points    From_ID    To_ID  
------------------------------------------------
 000     0       0        11          0       11
 001     0       1        10         11       21
 002     1       0         9         21       30
 003     1       1         9         30       39
 004     2       0         4         39       43
```
As is shown **prn_tbl** produces a nicely labelled output from the structured array that can be returned from the **Geo.info** method.
A quick survey shows repetition of the shape ID in the *Part* column.  The *Points* for each part are given, from which the *From_ID* and *To_ID* values are derived.


----
The methods and functions that will be shown use this information in their processing.  In this fashion, it is possible to try and optimize the derivation of properties and application of functions by using the whole point sequence of their subgroupings.

This will obviously not be possible in all situations, but every bit helps.


----
# ndarray values from esri geometry


**(1) The arcpy geometry**

This is what the geometry looks like for the first shape (multipart with holes).

```
p0
<Polygon object at 0x2a193f76128[0x2a18ea2d8a0]>

p0[:2]  # ---- two parts, so slice
[<Array [<Point (300010.0, 5000020.0, #, #)>, <Point (300010.0, 5000010.0, #, #)>, <Point (300000.0, 5000010.0, #, #)>,
         <Point (300000.0, 5000020.0, #, #)>, <Point (300010.0, 5000020.0, #, #)>,
         None,
         <Point (300003.0, 5000019.0, #, #)>, <Point (300003.0, 5000013.0, #, #)>, <Point (300009.0, 5000013.0, #, #)>,
         <Point (300009.0, 5000019.0, #, #)>, <Point (300003.0, 5000019.0, #, #)>]>,
 <Array [<Point (300008.0, 5000018.0, #, #)>, <Point (300008.0, 5000014.0, #, #)>, <Point (300004.0, 5000014.0, #, #)>,
         <Point (300004.0, 5000018.0, #, #)>, <Point (300008.0, 5000018.0, #, #)>,
         None,
         <Point (300006.0, 5000017.0, #, #)>, <Point (300005.0, 5000015.0, #, #)>, <Point (300007.0, 5000015.0, #, #)>,
         <Point (300006.0, 5000017.0, #, #)>]>]
```
The polygon consists of two parts, represented as the arcpy.Array.  This in turn consists of sequences of arcpy.Point values with outer rings ordered clockwise and inner rings/holes, order counter-clockwise.  Inner and outer rings are separated by None, rather than a null point since a null point, unfortunately, has X and Y values of 0.
```
(arcpy.Point()
<Point (0.0, 0.0, #, #)>
```


**(2)  FeatureClassToNumPyArray**

The standby, great for singlepart simple shapes.  You have to read the X, and Y coordinates separately or as a ``SHAPE@XY`` since reading the ``SHAPE@`` to retrieve the object directly is not permitted.

In the examples below, extra effort would have to be made to subtract the extent minimum from each point to obtain their values relative to it.

```
a0 = arcpy.da.FeatureClassToNumPyArray(in_fc3, ['SHAPE@X', 'SHAPE@Y'], explode_to_points=True, spatial_reference=SR)
a0
array([(300010., 5000020.), (300010., 5000010.), (300000., 5000010.), (300000., 5000020.),
       (300010., 5000020.), (300003., 5000019.), (300003., 5000013.), (300009., 5000013.),
       (300009., 5000019.), (300003., 5000019.), (300008., 5000018.), (300008., 5000014.),
       (300004., 5000014.), (300004., 5000018.), (300008., 5000018.), (300006., 5000017.),
       (300005., 5000015.), (300007., 5000015.), (300006., 5000017.), (300012., 5000018.),
       (300012., 5000012.), (300020., 5000012.), (300020., 5000010.), (300010., 5000010.),
       (300010., 5000020.), (300020., 5000020.), (300020., 5000018.), (300012., 5000018.),
       (300025., 5000024.), (300025., 5000014.), (300015., 5000014.), (300015., 5000016.),
       (300023., 5000016.), (300023., 5000022.), (300015., 5000022.), (300015., 5000024.),
       (300025., 5000024.)], dtype=[('SHAPE@X', '<f8'), ('SHAPE@Y', '<f8')])
```


**(3)  SearchCursors and the ``__geo_interface__`` method**

Works, and useful if you intend to work with the arcgis module.  There are variants on this as well depending on whether you want arrays or arrays or just an array of objects.

```
with arcpy.da.SearchCursor(in_fc3, 'SHAPE@', None, SR) as cursor:
    a1 = [row[0].__geo_interface__['coordinates'] for row in cursor] 
```
```
a1  # ---- as a list of values ----
[[[[(300010.0, 5000020.0),
    (300010.0, 5000010.0),
    (300000.0, 5000010.0),
    (300000.0, 5000020.0),
    (300010.0, 5000020.0)],
   [(300003.0, 5000019.0),
    (300003.0, 5000013.0),
    (300009.0, 5000013.0),
    (300009.0, 5000019.0),
    (300003.0, 5000019.0)]]],
 [[[(300008.0, 5000018.0),
    (300008.0, 5000014.0),
    (300004.0, 5000014.0),
    (300004.0, 5000018.0),
    (300008.0, 5000018.0)],
   [(300006.0, 5000017.0),
    (300005.0, 5000015.0),
    (300007.0, 5000015.0),
    (300006.0, 5000017.0)]]],
 [[[(300012.0, 5000018.0),
    (300012.0, 5000012.0),
    (300020.0, 5000012.0),
    (300020.0, 5000010.0),
    (300010.0, 5000010.0),
    (300010.0, 5000020.0),
    (300020.0, 5000020.0),
    (300020.0, 5000018.0),
    (300012.0, 5000018.0)]]],
 [[[(300025.0, 5000024.0),
    (300025.0, 5000014.0),
    (300015.0, 5000014.0),
    (300015.0, 5000016.0),
    (300023.0, 5000016.0),
    (300023.0, 5000022.0),
    (300015.0, 5000022.0),
    (300015.0, 5000024.0),
    (300025.0, 5000024.0)]]]]

np.asarray(a1)  # ---- as an object array containing lists of coordinates

array([[list([[(300010.0, 5000020.0), (300010.0, 5000010.0), (300000.0, 5000010.0), (300000.0, 5000020.0),
               (300010.0, 5000020.0)], [(300003.0, 5000019.0), (300003.0, 5000013.0), (300009.0, 5000013.0),
               (300009.0, 5000019.0), (300003.0, 5000019.0)]])],
       [list([[(300008.0, 5000018.0), (300008.0, 5000014.0), (300004.0, 5000014.0), (300004.0, 5000018.0),
               (300008.0, 5000018.0)], [(300006.0, 5000017.0), (300005.0, 5000015.0), (300007.0, 5000015.0),
               (300006.0, 5000017.0)]])],
       [list([[(300012.0, 5000018.0), (300012.0, 5000012.0), (300020.0, 5000012.0), (300020.0, 5000010.0),
               (300010.0, 5000010.0), (300010.0, 5000020.0), (300020.0, 5000020.0), (300020.0, 5000018.0),
               (300012.0, 5000018.0)]])],
       [list([[(300025.0042000003, 5000024.0), (300024.9957999997, 5000014.0), (300014.9957999997, 5000014.0),
               (300014.9974999996, 5000016.0), (300022.9974999996, 5000016.0), (300023.0025000004, 5000022.0),
               (300015.0025000004, 5000022.0), (300015.0042000003, 5000024.0), (300025.0042000003, 5000024.0)]])]],
      dtype=object)
```

**(4) Searchcursors and _as_narray**

A related ditty, however, you have to specify the fields directly and you essentially get the equivalent of FeatureClassToNumPyArray.

```
cur = arcpy.da.SearchCursor(in_fc2, ['OID@', 'SHAPE@X', 'SHAPE@Y'], spatial_reference=SR, explode_to_points=True)

cur._as_narray()  # ---- The worker
 
array([(1, 300010., 5000020.), (1, 300010., 5000010.), (1, 300000., 5000010.), (1, 300000., 5000020.),
       (1, 300010., 5000020.), (1, 300003., 5000019.), (1, 300003., 5000013.), (1, 300009., 5000013.),
       (1, 300009., 5000019.), (1, 300003., 5000019.), (1, 300008., 5000018.), (1, 300008., 5000014.),
       (1, 300004., 5000014.), (1, 300004., 5000018.), (1, 300008., 5000018.), (1, 300006., 5000017.),
       (1, 300005., 5000015.), (1, 300007., 5000015.), (1, 300006., 5000017.), (2, 300012., 5000018.),
       (2, 300012., 5000012.), (2, 300020., 5000012.), (2, 300020., 5000010.), (2, 300010., 5000010.),
       (2, 300010., 5000020.), (2, 300020., 5000020.), (2, 300020., 5000018.), (2, 300012., 5000018.),
       (2, 300025., 5000024.), (2, 300025., 5000014.), (2, 300015., 5000014.), (2, 300015., 5000016.),
       (2, 300023., 5000016.), (2, 300023., 5000022.), (2, 300015., 5000022.), (2, 300015., 5000024.),
       (2, 300025., 5000024.)], dtype=[('OID@', '<i4'), ('SHAPE@X', '<f8'), ('SHAPE@Y', '<f8')])

# related

cur._dtype   # ==> dtype([('OID@', '<i4'), ('SHAPE@X', '<f8'), ('SHAPE@Y', '<f8')])

cur.fields   # ==> ('OID@', 'SHAPE@X', 'SHAPE@Y')

```


<a href="url"><img src="images/bad_shape.png" align="right" height="300" width="auto" ></a>

The parts and the geometry are not identified within the sequences.  Constructing points from the above is no big deal, but polylines and polygons would fail miserably... as shown in this example.

The need to identify parts and holes in polygons prompted this study to see whether arcpy geometries could be represented in a different manner in numpy array format.  Currently, there are operations that cannot be done simply on arcpy geometries that are so simple in numpy.  Want to shift some polygons a couple of meters?  Just shuffle through a search cursor disassemble the point to an arcpy.Array, cycle through each point (checking for None), then doing the math on each point.  Finally, just reassemble the points array and reconstitute the polygon.  In numpy, if the whole dataset can be represented as an Nx2 array... you just add/subtract from the whole array.  Other functions, like convex hulls will require you to operate on 'chunks' of the array, rather than on the whole dataset at once.  At least nothing needs to be devolved to its smallest part first.  More on this in subsequent sections.


