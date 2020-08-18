# npGeomTools

----

<a href="url"><img src="../images/npGeomTools.png" align="right" height="auto" width="200" ></a>


The following tools are implemented in the npGeom.tbx toolbox for use in ArcGIS Pro.

The *Geo* array, based on a numpy array, is used along with *arcpy* functions to implement the tools.

The tools here are the most recent version of those provided in *FreeTools*








(1) Attribute tools

The tools are self-evident and don't include multiple options.

(2) Containers

The containers toolset offers the options shown in the image.

`Bounding circles` is probalby the most uncommon for GIS tools.

The `extent polygon` is the axis aligned extent of the feature geometry.

`Convex hull` is included for convenience.  It uses scipy's Qhull implementation.

(3) Conversion

The options for `conversion` are as follows:

`Feature to point` will return the geometry centroid for polygons.

`Split at vertices` is for polygon geometry. From-to point segmentation of the geometry will be returned.

`Vertices to points` applies to polyline/polygon geometry.

`Polygons to polylines` is also a convenience function since it only requires a `Kind (K)` conversion in the Geo class.  The reciprocal function was not implemented because I didn't want to provide a whole load of geometry checks.  If you have polyline geometry that you know will create well-formed polygons, simply change `K`.


<a href="url"><img src="../images/containers.png" align="left" height="auto" width="200" ></a>
<a href="url"><img src="../images/npGeo_conversion_tools.png" align="center" height="auto" width="200" ></a>








*Source image*

../numpy_geometry/images/npGeomTools.png
