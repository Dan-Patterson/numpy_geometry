# -*- coding: utf-8 -*-
r"""
-----------
npgDocs.py
-----------

Documentation for Geo array class methods and functions.

----

Script :
    npgDocs.py

Author :
    Dan_Patterson

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2020-10-07

Purpose
-------
Documentation strings.  These docstrings are assigned in the __init__.py file.

"""
# pycodestyle D205 gets rid of that one blank line thing
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0402,E0611,E1136,E1121,R0904,R0914,
# pylint: disable=W0201,W0212,W0221,W0612,W0621,W0105
# pylint: disable=R0902


import sys
# from textwrap import dedent
import numpy as np

# noqa: E501
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['npGeo_doc', 'Geo_hlp',
           'shapes_doc', 'parts_doc',
           'outer_rings_doc', 'inner_rings_doc',
           'get_shapes_doc', 'radial_sort_doc',
           'sort_by_extent_doc',
           'array_IFT_doc', 'dirr_doc'
           ]

author_date = r"""
Author :
    Dan Patterson
- Dan_Patterson@carleton.ca
- https://github.com/Dan-Patterson
- Initial creation period 2019-05 onward.

"""


def _update_docstring(obj, doc):
    """Update/expand a docstring.
    Based on _add_docstring in function_base.py
    """
    try:
        obj.__doc__ = doc
    except Exception:
        pass


# ----------------------------------------------------------------------------
# ---- (1) ...npGeo
#
npGeo_doc = author_date + r"""

Purpose
-------
A numpy based geometry class, its properties and methods.

Notes
-----
**Class instantiation**

Quote from Subclassing ndarrays::

    As you can see, the object can be initialized in the __new__ method or the
    __init__ method, or both, and in fact ndarray does not have an __init__
    method, because all the initialization is done in the __new__ method.

----

Geo class notes
---------------
Create a Geo array based on the numpy ndarray.  The class focus is on
geometry properties and methods.  Construction of geometries can be made
using numpy arrays, File Geodatabase Featureclasses (Esri) or GeoJSON data
as the source of the base geometries.

The IDs can either be 0-based or in the case of some data-types, 1-based.
No assumption is made about IDs being sequential.  In the case of featureclass
geometry, the OID@ property is read.  For other geometries, provide ID values
as appropriate.

Point, polyline, polygon features represented as numpy ndarrays.
The required inputs are created using `fc_geometry(in_fc)` or
`Arrays_to_Geo`.

**Attributes**

Normal ndarray parameters including shape, ndim, dtype.

shapes :
    The points for polyline, polygons.
parts :
    Multipart shapes and/or outer and inner rings for holes.
bits :
    The final divisions to individual bits constituting the shape.
is_multipart :
    Array of booleans
part_pnt_cnt :
    ndarray of ids and counts.
pnt_cnt :
    ndarray of ids and counts.
geometry properties :
    Areas, centers, centroids and lengths are properties and not methods.

**Comments**

You can use `arrays_to_Geo` to produce the required 2D array from lists
of array_like objects of the same dimension, or a single array.
The IFT will be derived from breaks in the sequence resulting from nesting of
lists and/or arrays.

>>> import npgeom as npg
>>> g = npg.Geo(a, IFT)
>>> g.__dict__.keys()
... dict_keys(['IFT', 'K', 'Info', 'IDs', 'Fr', 'To', 'CW', 'PID', 'Bit', 'FT',
...  'IP', 'N', 'U', 'SR', 'X', 'Y', 'XY', 'LL', 'UR', 'Z', 'hlp'])
>>> sorted(g.__dict__.keys())
... ['Bit', 'CW', 'FT', 'Fr', 'IDs', 'IFT', 'IP', 'Info', 'K', 'LL', 'N',
...  'PID','SR', 'To', 'U', 'UR', 'X', 'XY', 'Y', 'Z', 'hlp']

----

**See Also**

__init__.py :
    General comments about the package.
npg_io.py :
    Import and conversion routines for the Geo class.
npg_geom :
    Methods/functions for working with the Geo class or used by it.
npg_table :
    Methods/functions associated with tabular data.

----

**General notes**

The Geo class returns a 2D array of points which may consist of single or
multipart shapes with or without inner rings (holes).

The methods defined  in the Geo class allow one to operate on the parts of the
shapes separately or in combination.  Since the coordinate data are represented
as an Nx2 array, it is sometimes easier to perform calculations on the dataset
all at once using numpy functions.  For example, to determine the
minimum for the whole dataset:

>>> np.min(Geo, axis=0)

**Useage of methods**

`g` is a Geo instance with 2 shapes.  Both approaches yield the same results.

>>> Geo.centers(g)
array([[ 5.  , 14.93],
       [15.5 , 15.  ]])
>>> g.centers()
array([[ 5.  , 14.93],
       [15.5 , 15.  ]])

References
----------
`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.

`The N-dimensional array (ndarray)
<https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_.

`Scalable vector graphics... SVG
<https://www.w3.org/TR/SVG/>`_.

----

**Sample file**

Save to folder::

    fname = 'C:/Git_Dan/npgeom/data/geo_array.npz'
    ift = g.IFT  # g is a Geo array
    xt = g.XT
    sr = np.array([g.SR])
    np.savez(fname, g, ift, xt, sr)

Load file::

    fname = 'C:/Git_Dan/npgeom/data/g.npz'
    npzfiles = np.load(fname)     # ---- the Geo, I(ds)F(rom)T(o) arrays
    npzfiles.files                # ---- will show ==> ['g', 'IFT']
    arr = npzfiles['g']            # ---- slice by name from the npz file
    IFT = npzfiles['IFT']         #       to get each array
    extents = npzfiles['xt']
    sr = npzfiles['sr']
    #         arr, IFT, Kind, Extent, Info, SR (spatial reference)
    g = npg.Geo(g, ift, 2, extents, "Geo array", sr_array)

**how to test for Geo array**

>>> if ('Geo' in str(type(obj))) & (issubclass(obj.__class__, np.ndarray)):
        print(`do stuff`)
... # end of check

"""

# ----------------------------------------------------------------------------
# ---- (2) ... Geo class
# ---- ... Geo_hlp

Geo_hlp = r"""

**Geo class**
-------------

Construction from an ndarray, IFT, Kind and optional Info.

Parameters
----------

Required

arr : array_like
    A 2D array sequence of points with shape (N, 2).
IFT : array_like
    Defines, the I(d)F(rom)T(o) and other structural elements that are
    present in polyline or polygon geometry that `arr` represents.
    Shape (N, 6) required.
Kind : integer
    Points (0), polylines/lines (1) and polygons (2).
Info : text (optional)
    Optional information if needed.

**Derived**

IDs : IFT[:, 0]
    Shape ids, the id number will be repeated for each part and hole in
    the shape
Fr : IFT[:, 1]
    The ``from`` point in the point sequence.
To : IFT[:, 2]
    The ``to`` point in the point sequence.
CW : IFT[:, 3]
    A value of ``1`` for exterior rings, ``0`` for interior/holes.
PID : IFT[:, 4]
    Part ids sequence by shape.  A singlepart shape will have one (1) part.
    Subsequent parts are numbered incrementally.
Bit : IFT[:, 5]
    The bit sequence in a singlepart feature with holes and/or multipart
    features with or without holes
FT : IFT[:, 1:3]
    The from-to ids together (Fr, To).
IP : IFT[:, [0, 4]]
    Shape and part ids together (IDs, PID)
N : integer
    The number of unique shapes.
U : integer(s)
    A sequence of integers denoting the unique feature ID values.  There is no
    requirement for these to be sequential.
SR : text
    Spatial reference name.
X, Y, XY, Z: Derived from columns in the point array.
    X = arr[:, 0], Y = arr[:, 0], XY = arr[:, :2],
    Z = arr[:, 2] if defined
XT : array
    An array/list of points identifying the lower-left and upper-right,
    of full extent of all the geometry objects.
LL, UR : array
    The extent points as defined in XT.
hlp : text
    self.H or self.__doc__ help information for the a Geo array `self`.

Example
-------
A featureclass with 3 shapes. The first two are multipart shapes with holes.

>>> arr.dtype, arr.shape
(dtype('float64'), (62, 2))

>>> arr.IFT  # ---- annotated ----
#      IDs, Fr, To, CW, PID, Bit
array([[ 1,  0,  5,  1,   1,  0],  # first shape, first part, outer ring
       [ 1,  5, 10,  0,   1,  1],  # hole 1
       [ 1, 10, 14,  0,   1,  2],  # hole 2
       [ 1, 14, 18,  0,   1,  3],  # hole 3
       [ 1, 18, 23,  1,   2,  0],  # first shape, second part, outer ring
       [ 1, 23, 27,  0,   2,  1],  # hole 1
       [ 2, 27, 36,  1,   1,  0],  # second shape, first part, outer ring
       [ 2, 36, 46,  1,   2,  0],  # second shape, second part, outer ring
       [ 2, 46, 50,  0,   2,  1],  # hole 1
       [ 2, 50, 54,  0,   2,  2],  # hole 2
       [ 2, 54, 58,  0,   2,  3],  # hole 3
       [ 3, 58, 62,  1,   1,  0]], # third shape, first part, outer ring
                     dtype=int64)
"""

shapes_doc = r"""

Returns
-------
Either an object array or ndarray depending on the shape of the parts.
The overall array should be an object array.

Notes
-----
Pull out and ravel the from-to point IDs.  Slice the first and last
locations. Finally, slice the actual coordinates for each range.

"""

parts_doc = r"""

Returns
-------
Either an array of ndarrays and/or object arrays.  Slice by IP, ravel
to get the first and last from-to points, then slice the XY
coordinates.
"""

# ---- ... outer_rings
outer_rings_doc = r"""

Returns
-------
A new Geo array with holes discarded.  The IFT is altered to adjust for the
removed points. If there are not holes, `self.bits` is returned.
"""

# ---- ... inner_rings
inner_rings_doc = r"""

Returns
-------
Return a list of ndarrays or optionally a new Geo array.
Use True for `to_clockwise` to convert holes to outer rings.
"""

# ---- ... get_shapes
get_shapes_doc = r"""

The original IDs are added to the `Info` property of the output array.
The point sequence is altered to reflect the new order.

Parameters
----------
ID_list : array_like
    A list, tuple or ndarray of ID values identifying which features
    to pull from the input.
asGeo : Boolean
    True, returns an updated Geo array.  False returns an ndarray or
    object array.

Notes
-----
>>> a.pull_shapes(np.arange(3:8))  # get shapes over a range of values
>>> a.pull_shapes([1, 3, 5])  # get selected shapes
"""

# ---- ... radial_sort
radial_sort_doc = r"""

The features will be sorted so that their first coordinate is in the
lower left quadrant (SW) as best as possible.  Outer rings are sorted
clockwise and interior rings, counterclockwise.  Existing duplicates
are removed to clean features, hence, the dup_first to provide closure
for polygons.

Returns
-------
Geo array, with points radially sorted (about their center).

Notes
-----
Angles relative to the x-axis.
>>> rad = np.arange(-6, 7.)*np.pi/6
>>> np.degrees(rad)
... array([-180., -150., -120., -90., -60., -30.,  0.,
...          30., 60., 90., 120., 150., 180.])

References
----------
`<https://stackoverflow.com/questions/35606712/numpy-way-to-sort-out-a
-messy-array-for-plotting>`_.
"""

# ---- ... sort_by_extent
sort_by_extent_doc = r"""

Parameters
----------
extent_pnt : text
    Use `LB` for left-bottom, `LT` for left-top or `C`  for center.
key : int
    The key is the directional order used to sort the features from their
    extent points as described in `Notes`.
just_indices : boolean
    True returns the geometry ID values only.

Notes
-----
The key/azimuth information is::

              S-N SW-NE W-E NW-SE N-S NE-SW E-W SE-NW
    azimuth [   0,  45,  90, 135, 180, 225, 270, 315]
    key     [   0,   1,   2,   3,   4,   5,   6,   7]

    where a key of ``0`` would mean sort from south to north.

**Sorting... key - direction vector - azimuth**

::

      key     vector  azimuth
    |  0   | S to N  |    0  |
    |  1   | SW - NE |   45  |
    |  2   | W to E  |   90  |
    |  3   | NW - SE |  135  |
    |  4   | N to S  |  180  |
    |  5   | NE - SW |  225  |
    |  6   | E to W  |  270  |
    |  7   | SE - NW |  315  |


The key values are used to select the dominant direction vector.

>>> z = sin(a) * [X] + cos(a) * [Y]  # - sort by vector

References
----------
`vector sorting
<https://gis.stackexchange.com/questions/60134/sort-a-feature-table-by
-geographic-location>`_.
"""


# ----------------------------------------------------------------------------
# ---- (3) ... array_IFT
array_IFT_doc = r"""

Parameters
----------
in_arrays : list, array
    The input data can include list of lists, list of arrays or arrays
    including multidimensional and object arrays.
shift_to_origin : boolean
    True, shifts coordinates by subtracting the minimum x,y from all
    values.  Only use `True` if working with arrays for the first time.

Returns
-------
a_stack : ndarray
    Nx2 array (float64) of x,y values representing the geometry.
    There are no breaks in the sequence.
IFT : ndarray
    Nx6 array (int32) which contains the geometry IDs, the F(rom) and T(o)
    point sequences used to construct the parts of the geometry.
    Ring orientation for `polygon` geometry is also provided (CW or CCW).
    A geometry  with many parts will contain a `bit` and `part` sequence
    identifier.
extent : ndarray
    The bounding rectangle of the input geometry.

Notes
-----
- Called by `arrays_to_Geo`.
- Use `fc_geometry` to produce `Geo` objects directly from FeatureClasses.
"""


# ----------------------------------------------------------------------------
# ---- (4) ... dirr function
dirr_doc = r"""

Source, ``arraytools.py_tools`` has a pure python equivalent.

Parameters
----------
colwise : boolean
    `True` or `1`, otherwise, `False` or `0`.
cols : number
    Pick a size to suit.
prn : boolean
  `True` for print or `False` to return output as string.

Returns
-------
A directory listing of an object or module's namespace.

Notes
-----
See the `inspect` module for possible additions like `isfunction`,
`ismethod`, `ismodule`.

Example
-------
>>> npg.dirr(g)
----------------------------------------------------------------------
| dir(npgeom) ...
|    <class 'npgeom.Geo'>
-------
  (001)  ... Geo class ...       Bit                     CW
  (002)  FT                      Fr                      H
  (003)  IDs                     IFT                     IFT_str
  (004)  IP                      Info                    K
  (005)  LL                      N                       PID
  (006)  SR                      SVG                     To
  (007)  U                       UR                      X
  (008)  XT                      XY                      Y
  (009)  Z                       __author__              __dict__
  (010)  __module__              __name__                angles_polygon
  (011)  angles_polyline         aoi_extent              aoi_rectangle
  (012)  areas                   bit_IFT                 bit_ids
... snip
"""


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polylines2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygon2pnts"
