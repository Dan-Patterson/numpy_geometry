# npGeo
A numpy geometry class and functions that work with arcpy and ESRI featureclasses

This is a work in progress, so bear with me.  The intent of the Geo class is to treat the geometry of featureclasses as one entity.  Most approaches I have see so far tend to construct the geometric representations of geometries using some variant of arcpy cursors.

When trying to work with numpy and the geometries, this creates problems since geometry is rarely nice uniform same shapes.  Object arrays containing the coordinates are the norm.  What I set out to do was create a uniform 2D array of coordinates with np.nan values separating the parts of a particular shape and a companion array which denote the feature ID and the from-to point pairs.  This is similar to the FeatureClassToNumPy array approach, but that particular function and its inverse are only useful for simple singlepart geometries.

I will document and build on this tools set with examples.  I am only working with featureclasses stored in file geodatabase featureclasses.

**Some links**

[Geometry in NumPy](https://community.esri.com/blogs/dan_patterson/2019/03/17/geometry-in-numpy-1)

[Geometry, Arcpy and NumPy](https://community.esri.com/blogs/dan_patterson/2019/04/10/geometry-arcpy-and-numpy-2)

[Deconstructing poly* features](https://community.esri.com/blogs/dan_patterson/2019/04/10/geometry-deconstructing-poly-features-3)

[Reconstructing poly* features](https://community.esri.com/blogs/dan_patterson/2019/04/17/geometry-reconstructing-poly-features-4)

[Attributes.. the other bits](https://community.esri.com/blogs/dan_patterson/2019/04/17/geometry-attributes-actually-the-other-bits-5)

[Don't believ what you see](https://community.esri.com/blogs/dan_patterson/2019/05/09/geometry-dont-believe-what-you-see-6)

[Geometry: forms of the same feature](https://community.esri.com/blogs/dan_patterson/2019/05/13/geometry-forms-of-the-same-feature-7)

Consider the following multipart shapes.  The first shape has is second part slightly offset and it also contains a hole.  The second shape is a flip/mirror/translate of its first part.

<a href="url"><img src="https://github.com/Dan-Patterson/npGeo/blob/master/Shape2.png" align="right" height="229" width="400" ></a>

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
 016     0   x       nan     nan  ---- the null point, separating the inner and outer rings of the 2nd part of the first shape
 017     0          6.00   17.00
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
array([[ 0,  0, 11],    1st shape, 1st part, points 0 to but not including 11
       [ 0, 11, 21],    1st shape, 2nd part
       [ 1, 21, 30],    2nd shape, 1st part
       [ 1, 30, 39]])   2nd shape, 2nd part
```       
The methods and functions that will be shown use this information in their processing.  In this fashion, it is possible to try and optimize the derivation of properties and application of functions by using the whole point sequence of their subgroupings.



