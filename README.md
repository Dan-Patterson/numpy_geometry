# npGeo
A numpy geometry class and functions that work with arcpy and ESRI featureclasses

This is a work in progress, so bear with me.  The intent of the Geo class is to treat the geometry of featureclasses as one entity.  Most approaches I have see so far tend to construct the geometric representations of geometries using some variant of arcpy cursors.

When trying to work with numpy and the geometries, this creates problems since geometry is rarely nice uniform same shapes.  Object arrays containing the coordinates are the norm.  What I set out to do was create a uniform 2D array of coordinates with np.nan values separating the parts of a particular shape and a companion array which denote the feature ID and the from-to point pairs.  This is similar to the FeatureClassToNumPy array approach, but that particular function and its inverse are only useful for simple singlepart geometries.

I will document and build on this tools set with examples.  I am only working with featureclasses stored in file geodatabase featureclasses.
