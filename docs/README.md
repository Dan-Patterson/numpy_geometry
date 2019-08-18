## Docs ##

Import **npgeom** and take a subsample of some featureclass (in_fc) geometry objects (g).

IFT refers to the feature `id` value, the `from` and `to` points.

To reduce numeric problems, the value of the lower left corner is subtracted from all coordinates moving coordinate space into quadrant I.  You could also subtract the mean value of the points which would center the shapes about the x-y axis origin.

```
Usage...
  import npgeom as npg

in_fc, g = _test_()

Type :  Polygons
IFT  :
[[  1   0  11]
 [  1  11  21]
 [  2  21  31]
 [  2  31  40]
 [  3  40  44]
 [  4  44  49]
 [  5  49  56]
 [  6  56  86]
 [  7  86 110]
 [  8 110 117]
 [  9 117 124]
 [ 10 124 128]
 [ 11 128 134]
 [ 12 134 147]]
```

The subsample is converted to a geoarray, so the point numbering will become different, but the feature ids remain the same.
```
a = g.pull([2, 8, 9, 10], asGeo=True)

a.IFT
Out[4]: 
array([[ 2,  0, 10],
       [ 2, 10, 19],
       [ 8, 19, 26],
       [ 9, 26, 33],
       [10, 33, 37]])

```

Now we will step back and explore the geoarray properties.

In this example we are working with polygons 2, 8, 9, 10.  Polygon 2 is a multipart polygon.  The first part is constructed from points 0 to 10 (but not including 10).  The second part, continues from points 10 to 19.

```
# ---- From the array construction properties

a.IFT  # ---- I(d), F(rom), T(o) values, we are working with polygons 2, 8, 9, 10
array([[ 2,  0, 10],
       [ 2, 10, 19],
       [ 8, 19, 26],
       [ 9, 26, 33],
       [10, 33, 37]])

a.IDs  # ---- Just the id values
array([ 2,  2,  8,  9, 10])

a.FT   # ---- Just the from-to values
array([[ 0, 10],
       [10, 19],
       [19, 26],
       [26, 33],
       [33, 37]])

a.K    # ---- The feature type.  Points (0), Polylines (1) and Polygons (2)
2

a.Info  # ---- No special information
''

a.N    # ---- Number of features.
4

```
Some properties

```
a.part_cnt    # ---- number of parts for each shape formatted as [id, part count]
array([[ 2,  2],
       [ 8,  1],
       [ 9,  1],
       [10,  1]], dtype=int64)

a.pnt_cnt     # ---- number of points
array([[ 2, 10],
       [ 2,  9],
       [ 8,  7],
       [ 9,  7],
       [10,  4]])
 
a.areas    # ---- planar/euclidean areas
array([104.  , 104.  ,  15.35,  19.47,   5.9 ])

a.centers  # ---- x, y coordinates for the center 
array([[15.33,  5.56],
       [19.5 ,  9.  ],
       [21.46, 20.71],
       [23.72, 19.02],
       [31.32, 17.85]])

a.centroids  # ---- centroids... area weighted centers
array([[17.5 ,  7.  ],
       [17.5 ,  7.  ],
       [20.96, 20.96],
       [24.4 , 18.93],
       [31.32, 17.85]])

a.lengths  # ---- lengths/perimeter depending on feature type.
array([112.  , 112.  ,  16.05,  19.97,  12.65])
```
