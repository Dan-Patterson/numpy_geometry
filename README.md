# npGeo
A numpy geometry class and functions that work with arcpy and ESRI featureclasses

This is a work in progress, so bear with me.  The intent of the Geo class is to treat the geometry of featureclasses as one entity.  Most approaches I have see so far tend to construct the geometric representations of geometries using some variant of arcpy cursors.

When trying to work with numpy and the geometries, this creates problems since geometry is rarely nice uniform same shapes.  Object arrays containing the coordinates are the norm.  What I set out to do was create a uniform 2D array of coordinates with np.nan values separating the parts of a particular shape and a companion array which denote the feature ID and the from-to point pairs.  This is similar to the FeatureClassToNumPy array approach, but that particular function and its inverse are only useful for simple singlepart geometries.

I will document and build on this tools set with examples.  I am only working with featureclasses stored in file geodatabase featureclasses.

**Some links**

https://community.esri.com/blogs/dan_patterson/2019/03/17/geometry-in-numpy-1
https://community.esri.com/blogs/dan_patterson/2019/04/10/geometry-arcpy-and-numpy-2
https://community.esri.com/blogs/dan_patterson/2019/04/10/geometry-deconstructing-poly-features-3
https://community.esri.com/blogs/dan_patterson/2019/04/17/geometry-reconstructing-poly-features-4
https://community.esri.com/blogs/dan_patterson/2019/04/17/geometry-attributes-actually-the-other-bits-5
https://community.esri.com/blogs/dan_patterson/2019/05/09/geometry-dont-believe-what-you-see-6
https://community.esri.com/blogs/dan_patterson/2019/05/13/geometry-forms-of-the-same-feature-7

