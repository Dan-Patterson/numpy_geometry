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
**Some properties**

*part, counts, areas, centroids*
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

*extents*

```
a.aoi_extent()  # ---- the extent of the whole file... aka, the area of interest
array([ 0.  ,  0.  , 36.71, 33.  ])

a.aoi_rectangle()  # ---- the aoi as a rectangle of x, y points. 
array([[ 0.  ,  0.  ],
       [ 0.  , 33.  ],
       [36.71, 33.  ],
       [36.71,  0.  ],
       [ 0.  ,  0.  ]])

a.extents(by_part=False)  # ---- the extents of each shape (Left, Bottom, Right, Top)
Out[15]: 
array([[ 0.  ,  0.  , 10.  , 10.  ],
       [10.  ,  0.  , 25.  , 14.  ],
       [10.  , 10.  , 15.  , 18.  ],
       [12.  , 27.  , 18.  , 33.  ],
       [ 0.  , 24.5 ,  6.  , 31.5 ],
       [19.14, 25.33, 24.86, 28.67],
       [12.22, 22.22, 15.77, 25.78],
       [18.16, 18.73, 23.76, 23.18],
       [21.29, 16.76, 27.61, 21.32],
       [29.98, 15.41, 32.32, 20.96],
       [26.82,  9.46, 33.18, 14.54],
       [29.29, 22.17, 36.71, 29.83]])

a.extents(by_part=True)  # ---- same as above, but the extent for multiparts is also returned
array([[ 0.  ,  0.  , 10.  , 10.  ],
       [ 4.  ,  4.  ,  8.  ,  8.  ],
       [10.  ,  0.  , 20.  , 10.  ],
       [15.  ,  4.  , 25.  , 14.  ],
       [10.  , 10.  , 15.  , 18.  ],
       [12.  , 27.  , 18.  , 33.  ],
       [ 0.  , 24.5 ,  6.  , 31.5 ],
       [19.14, 25.33, 24.86, 28.67],
       [12.22, 22.22, 15.77, 25.78],
       [18.16, 18.73, 23.76, 23.18],
       [21.29, 16.76, 27.61, 21.32],
       [29.98, 15.41, 32.32, 20.96],
       [26.82,  9.46, 33.18, 14.54],
       [29.29, 22.17, 36.71, 29.83]])

a.extent_rectangles(False)  # ---- like the aoi.rectangles, but polygon rectangles are by 
array([[[ 0.  ,  0.  ],     # shape (False) or by part (True)
        [ 0.  , 10.  ],
        [10.  , 10.  ],
        [10.  ,  0.  ],
        [ 0.  ,  0.  ]],
... snip
       [[29.29, 22.17],
        [29.29, 29.83],
        [36.71, 29.83],
        [36.71, 22.17],
        [29.29, 22.17]]])
```

*stats... mins, maxs, means*

```
a.mins(False)  # ---- maxs, maxs and means by shape (False) or by part (True)
array([[ 0.  ,  0.  ],
       [10.  ,  0.  ],
       [10.  , 10.  ],
       [12.  , 27.  ],
       [ 0.  , 24.5 ],
       [19.14, 25.33],
       [12.22, 22.22],
       [18.16, 18.73],
       [21.29, 16.76],
       [29.98, 15.41],
       [26.82,  9.46],
       [29.29, 22.17]])
 
 ```
 
 *retrieving shapes*
 
 ```
a.get(ID=3, asGeo=True)  # ---- a single shape as a Geo array
Geo([[14., 10.],
     [10., 10.],
     [15., 18.],
     [14., 10.]])

a.get(ID=3, asGeo=False)  # ---- as an ndarray
array([[15., 33.],
       [18., 30.],
       [15., 27.],
       [12., 30.],
       [15., 33.]])

a.pull([3,4], True)  # ---- multiple shapes in the order given as a Geo array
Geo([[14., 10.],
     [10., 10.],
     [15., 18.],
     [14., 10.],
     [15., 33.],
     [18., 30.],
     [15., 27.],
     [12., 30.],
     [15., 33.]])

a.pull([3, 4], False)  # ---- and as an ndarray
array([array([[14., 10.],
       [10., 10.],
       [15., 18.],
       [14., 10.]]),
       array([[15., 33.],
       [18., 30.],
       [15., 27.],
       [12., 30.],
       [15., 33.]])], dtype=object)
  
a.outer_rings(asGeo=True)  # ---- outer rings as Geo array
Geo([[10.  , 10.  ],
     [10.  ,  0.  ],
     [ 0.  ,  0.  ],
     [ 0.  , 10.  ],
     [10.  , 10.  ],
     ...,
     [29.31, 24.65],
     [29.29, 27.24],
     [31.96, 27.25],
     [31.96, 29.83],
     [34.33, 29.83]])

a.outer_rings(asGeo=True).IFT  # ---- note IFT changes
array([[  1,   0,   5],
       [  1,   5,  10],
       [  2,  10,  20],
       [  2,  20,  29],
       [  3,  29,  33],
       [  4,  33,  38],
       [  5,  38,  45],
       [  6,  45,  75],
       [  7,  75,  99],
       [  8,  99, 106],
       [  9, 106, 113],
       [ 10, 113, 117],
       [ 11, 117, 123],
       [ 12, 123, 136]])

a.outer_rings(asGeo=False)  # --- as a list of arrays
[array([[10., 10.],
        [10.,  0.],
        [ 0.,  0.],
        [ 0., 10.],
        [10., 10.]]), array([[8., 8.],
        [8., 4.],
        [4., 4.],
        [4., 8.],
        [8., 8.]]), ... snip ...
        ]
 ```
