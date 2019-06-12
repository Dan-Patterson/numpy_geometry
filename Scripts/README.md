# Scripts

## fc_npGeo.py

The methods used to convert esri geometries to an appropriate array format.
This script will require you have a valid license for ArcGIS Pro since esri FeatureClasses in a File Geodatabase need to be read and their arcpy geometries extracted and converted to numpy arrays.

Numpy is part of the distribution for ArcGIS Pro, so no other dependencies need to be made to their conda packages.
If you wish to clone their distribution or modify the existing one, some guidance is provided here.

[Clone... ArcGIS Pro ... for non administrators](https://community.esri.com/blogs/dan_patterson/2018/12/28/clone)

<a href="url"><img src="https://github.com/Dan-Patterson/npGeo/blob/master/images/clones2.png" align="center" height="200" width="auto" ></a>

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
            [nan, nan],         [20., 20.],
            [ 3., 19.],         [20., 18.],
            [ 3., 13.],         [12., 18.]],
            [ 9., 13.],
            [ 9., 19.],        [[25., 24.],
            [ 3., 19.]]),       [25., 14.],
     array([[ 8., 18.],         [15., 14.],
            [ 8., 14.],         [15., 16.],
            [ 4., 14.],         [23., 16.],
            [ 4., 18.],         [23., 22.],
            [ 8., 18.],         [15., 22.],
            [nan, nan],         [15., 24.],
            [ 6., 17.],         [25., 24.]]])
            [ 5., 15.],
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
