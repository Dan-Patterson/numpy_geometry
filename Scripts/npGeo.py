# -*- coding: utf-8 -*-
# noqa
"""
=====
npGeo
=====

Script : npGeo.py
    A geometry class and methods based on numpy.

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-12-04
    Initial creation period 2019-05.

Purpose : geometry tools
    A numpy geometry class, its properties and methods.

Notes
-----
**Class instantiation**

Quote from Subclassing ndarrays::

    As you can see, the object can be initialized in the __new__ method or the
    __init__ method, or both, and in fact ndarray does not have an __init__
    method, because all the initialization is done in the __new__ method.

----

**Geo class notes**


Create a Geo array based on the numpy ndarray.  The class focus is on
geometry properties and methods.  Construction of geometries can be made
using File Geodatabase Featureclasses (Esri) or GeoJSON data as the source
of the base geometries.

The IDs can either be 0-based or in the case of some data-types, 1-based.
No assumption is made about IDs being sequential.  In the case of featureclass
geometry, the OID@ property is read.  For other geometries, provide ID values
as appropriate.

Point, polyline, polygon features represented as numpy ndarrays.
The required inputs are created using ``fc_geometry(in_fc)`` or
``Arrays_to_Geo``.

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

You can use ``arrays_to_Geo`` to produce the required 2D array from lists
of array-like objects of the same dimension, or a single array.
The IFT will be derived from breaks in the sequence and/or the presence of
null points within a sequence.

>>> import npgeom as npg
>>> g = npg.Geo(a, IFT)
>>> g.__dict__.keys()
... dict_keys(['IFT', 'K', 'Info', 'IDs', 'Fr', 'To', 'CW', 'PID', 'Bit', 'FT',
...  'IP', 'N', 'U', 'SR', 'X', 'Y', 'XY', 'LL', 'UR', 'Z'])
>>> sorted(g.__dict__.keys())
... ['Bit', 'CW', 'FT', 'Fr', 'IDs', 'IFT', 'IP', 'Info', 'K', 'LL', 'N',
...  'PID','SR', 'To', 'U', 'UR', 'X', 'XY', 'Y', 'Z']

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

----

**Sample file**

Saved in the arraytools folder and on GitHub::

    fname = 'C:/Git_Dan/arraytools/Data/geo_array.npz'
    npzfiles = np.load(fname)   # ---- the Geo, I(ds)F(rom)T(o) arrays
    npzfiles.files              # ---- will show ==> ['s2', 'IFT']
    s2 = npzfiles['s2']         # ---- slice by name from the npz file to get
    IFT = npzfiles['IFT']       #      each array

"""
# pylint: disable=C0103, C0302, C0415, E1136, E1121, R0904, R0914,
# pylint: disable=W0201, W0212, W0221
# pylint: disable=R0902


import sys
import copy
from textwrap import dedent
import numpy as np

from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import repack_fields

# import arcgisscripting as ags
# from arcpy.da import FeatureClassToNumPyArray

import npgeom as npg
import npgeom.npg_geom as geom
from npg_io import prn_tbl
import smallest_circle as sc

# import importlib
# importlib.reload(npg)

# noqa: E501
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.4f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft)

script = sys.argv[0]  # print this should you need to locate the script

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS
TwoPI = np.pi*2.

__all__ = ['Geo', 'arrays_to_Geo', '_arr_ift_', 'Update_Geo',
           'Geo_to_arrays', '_fill_float_array', 'dirr',
           ]

hlp = r"""
**Geo class**

Construction from an ndarray, IFT, Kind and optional Info.

Parameters
----------
**Required**
arr : array-like
    A 2D array sequence of points with shape (N, 2).
IFT : array-like
    Defines, the I(d)F(rom)T(o) and other structural elements that are
    present in polyline or polygon geometry that ``arr`` represents .
    Shape (N, 6) required.
Kind : integer
    Points (0), polylines/lines (1) and polygons (2).
Info : string (optional)
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
    A sequence of integers indicating the feature ID value.  There is no
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
hlp : this
    self.H or self.docs where self is a Geo array will recall this information.

A featureclass with 3 shapes. The first two are multipart with holes.::
    arr.IFT
    array([[ 1,  0,  5,  1,  1,  0],
           [ 1,  5, 10,  0,  1,  1],
           [ 1, 10, 14,  0,  1,  2],
           [ 1, 14, 18,  0,  1,  3],
           [ 1, 18, 23,  1,  2,  0],
           [ 1, 23, 27,  0,  2,  1],
           [ 2, 27, 36,  1,  1,  0],
           [ 2, 36, 46,  1,  2,  0],
           [ 2, 46, 50,  0,  2,  1],
           [ 2, 50, 54,  0,  2,  2],
           [ 2, 54, 58,  0,  2,  3],
           [ 3, 58, 62,  1,  1,  0]], dtype=int64)
"""


# ----------------------------------------------------------------------------
# ---- (1) ... Geo class, properties and methods ... -------------------------
#
class Geo(np.ndarray):
    """
    **Geo class**

    Construction from an ndarray, IFT, Kind and optional Info.
    """
    # ----
    __name__ = "npGeo"
    __module__ = "npgeom"
    __author__ = "Dan Patterson 2019"
    __hlp__ = hlp

    def __new__(cls, arr=None,
                IFT=None,
                Kind=2,
                Extent=None,
                Info="Geo array",
                SR=None
                ):
        """
        See script header for construction notes.
        """
        arr = np.ascontiguousarray(arr)
        IFT = np.ascontiguousarray(IFT)
        if (arr.ndim != 2) or (IFT.ndim != 2):
            m = "ndim != 2 : {} or IFT.dim != 2 : {}"
            print(dedent(m).format(arr.ndim, IFT.ndim))
            return None
        if (IFT.shape[-1] < 4) or (Kind not in (0, 1, 2)):
            print(dedent(hlp))
            return None
        # ----
        self = arr.view(cls)      # view as Geo class
        self.IFT = IFT            # array id, fr-to, cw, part id
        self.K = Kind             # Points (0), Polylines (1), Polygons (2)
        self.Info = Info          # any useful information
        self.IDs = IFT[:, 0]      # shape id
        self.Fr = IFT[:, 1]       # from point id
        self.To = IFT[:, 2]       # to point id
        self.CW = IFT[:, 3]       # clockwise and outer/inner ring identifier
        self.PID = IFT[:, 4]      # part identifier per shape
        self.Bit = IFT[:, 5]      # bit sequence for each shape... arr.bit_seq
        self.FT = IFT[:, 1:3]     # from-to sequence
        self.IP = IFT[:, [0, 4]]  # shape and part id together
        # --- other properties
        uni, idx = np.unique(self.IDs, True)
        self.N = len(uni)       # sample size, unique shapes
        self.U = self.IDs[idx]  # unique ids
        self.SR = SR
        self.XT = Extent
        if self.XT is not None:
            self.LL, self.UR = self.XT  # extent of featureclass
        else:
            self.LL = np.array([0., 0.])
            self.UR = None
        if self.shape[1] >= 2:  # X,Y and XY initialize
            self.X = arr[:, 0]
            self.Y = arr[:, 1]
            self.XY = arr[:, :2]
        if self.shape[1] >= 3:  # add Z, although not implemented
            self.Z = arr[:, 2]  # directly, but kept for future additions
        else:
            self.Z = None
        self.hlp = hlp
        return self

    def __array_finalize__(self, src_arr):
        """The new object... this is where housecleaning takes place for
        explicit, view casting or new from template...
        ``src_arr`` is either None, any subclass of ndarray including our own
        (words from documentation) OR another instance of our own array.
        You can use the following with a dictionary instead of None:

        >>> self.info = getattr(obj,'info',{})
        """
        if src_arr is None:
            return
        self.IFT = getattr(src_arr, 'IFT', None)
        self.K = getattr(src_arr, 'K', None)
        self.Info = getattr(src_arr, 'Info', None)
        self.IDs = getattr(src_arr, 'IDs', None)
        self.Fr = getattr(src_arr, 'Fr', None)
        self.To = getattr(src_arr, 'To', None)
        self.CW = getattr(src_arr, 'CW', None)
        self.PID = getattr(src_arr, 'PID', None)
        self.Bit = getattr(src_arr, 'Bit', None)
        self.FT = getattr(src_arr, 'FT', None)
        self.IP = getattr(src_arr, 'IP', None)
        self.N = getattr(src_arr, 'N', None)
        self.U = getattr(src_arr, 'U', None)
        self.SR = getattr(src_arr, 'SR', None)
        self.X = getattr(src_arr, 'X', None)
        self.Y = getattr(src_arr, 'Y', None)
        self.XY = getattr(src_arr, 'XY', None)
        self.XT = getattr(src_arr, 'XT', None)
        self.LL = getattr(src_arr, 'LL', None)
        self.UR = getattr(src_arr, 'UR', None)
        self.Z = getattr(src_arr, 'Z', None)
        self.hlp = getattr(src_arr, 'hlp', None)

    def __array_wrap__(self, out_arr, context=None):
        """Wrap it up"""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # ------------------------------------------------------------------------
    # ---- End of class definition -------------------------------------------
    #
    # ---- help information
    @property
    def H(self):
        """Prints the parameter documentation for an instance of the Geo class.
        If you want to save this as a string, use::
            >>> doc_string = g.__docs__  # note the ``s`` in doc``s``
        """
        print(self.hlp)
        return

    @property
    def docs(self):
        """Prints the parameter documentation for an instance of the Geo class.
        If you want to save this as a string, use::
            >>> doc_string = g.__docs__  # note the ``s`` in doc``s``
        """
        print(hlp)
        return

    @property
    def info(self, prn=True):
        """Convert an IFT array to full information. Only the first 20 records
        maximum will be printed. To see the data structure, and/or more records
        use the ``prn_geo`` method.
        """
        info_ = self.IFT_str[:50]
        frmt = "-"*14 + "\nExtents :\n  LL {}\n  UR {}" + \
            "\nShapes :{:>6.0f}\nParts  :{:>6.0f}" + "\nPoints :{:>6.0f}\n"
        if prn:
            args = [str(self.LL), str(self.UR), len(self.U), info_.shape[0],
                    info_['To_pnt'][-1]]
            print(dedent(frmt).format(*args))
            prn_tbl(info_)
            return None
        return info_

    # ---- IFT : shape, part, bit
    # see also self.U, self.PID
    @property
    def shp_IFT(self):
        """Shape IFT values. Stack by column.  If there are no multipart shapes
        in the geometry ``shp_IFT == part_IFT``
        """
        if np.sum(self.is_multipart()[:, 1]) == 0:  # no multiparts check
            return self.part_IFT
        df = self.To - self.Fr
        cnt = np.bincount(self.IDs, df)
        too = np.cumsum(cnt, axis=0, dtype=np.int64)[1:]
        fr = np.concatenate(([0], too[:-1]), axis=0)
        ift = np.full((len(fr), 6), -1, dtype=np.int64)
        ift[:, 0] = self.U
        ift[:, 1] = fr
        ift[:, 2] = too
        return ift

    @property
    def part_IFT(self):
        """Part IFT values. Stack by column."""
        uni, idx = np.unique(self.IP, True, axis=0)
        idx = np.concatenate((idx, [len(self.IP)]))
        fr_to = list(zip(idx[:-1], idx[1:]))         # np.array_split or
        subs = [self.IFT[i[0]:i[1]] for i in fr_to]  # np.split equivalent
        # ---- fix ift sequence
        ifts = []
        for i, sub in enumerate(subs):
            ift = sub[0].tolist()
            ift[2] = sub[-1][2]
            ifts.append(ift)
        return np.array(ifts)

    @property
    def bit_IFT(self):
        """IFT for shape bits."""
        return self.IFT

    @property
    def IFT_str(self):
        """self.IFT as a structured array.

        OID_ : shape id
        From_pnt : from point id
        To_pnt : to point id for a shape
        Ring_type : outer (1) inner/hole (0)
        Part_ID : part id for each shape
        Bit_seq : sequence order of each part in a shape

        >>> npg.prn_tbl(z.IFT_str)  # to print
        """
        nmes = ["OID_", "From_pnt", "To_pnt", "Ring_type",
                "Part_ID", "Bit_seq"]
        return uts(self.IFT, names=nmes, align=False)

    #
    # ---- ids : shape, part, bit
    @property
    def shp_ids(self):
        """Shape ID values. Note, they may not be sequential or continuous."""
        return self.U  # self.shp_IFT[:, 0]

    @property
    def part_ids(self):
        """Return the ID values of the shape parts.  See, shp_ids warning."""
        return self.part_IFT[:, 0]  # [0, -1]]

    @property
    def bit_ids(self):
        """Return the ID values for each bit in a shape.  Ids are repeated for
        each part or ring in a shape.  See, shp_ids warning.
        """
        return self.IDs

    @property
    def bit_seq(self):
        """Return the bit sequence for each bit in a shape.  The sequence is
        numbered from zero to ``n``.  A shape can consist of a single bit or
        multiple bits consisting of outer rings and holes
        """
        return self.Bit

    @property
    def pnt_ids(self):
        """Feature id that each point belongs to.  Useful for slicing the
        points of poly features.  See, shp_ids warning.
        """
        reps = [np.repeat(i[0], i[2] - i[1]) for i in self.IFT]
        return np.concatenate(reps)

    def xy_id(self):
        """Return a structured array of numbered points shifted back to their
        original bounding box.
        """
        N = self.shape[0]
        dt = [('SeqID_', '<i4'), ('ID_', '<i4'), ('X_', '<f8'), ('Y_', '<f8')]
        z = np.empty((N,), dtype=dt)
        z['SeqID_'] = np.arange(N)
        z['ID_'] = self.pnt_ids
        z['X_'] = self.X + self.LL[0]
        z['Y_'] = self.Y + self.LL[1]
        z = repack_fields(z)
        return z

    #
    # ---- counts : shape, part, bit
    @property
    def shp_pnt_cnt(self):
        """Points in each shape.  Includes all parts and null points."""
        df = self.To - self.Fr
        cnt = np.bincount(self.IDs, df)[1:]
        out = np.zeros((self.U.size, 2), dtype=np.int32)
        out[:, 0] = self.U
        out[:, 1] = cnt
        return out

    @property
    def shp_part_cnt(self):
        """Part count for shapes. Returns IDs and count array"""
        uni, cnts = np.unique(self.part_ids, return_counts=True)
        return np.concatenate((uni[:, None], cnts[:, None]), axis=1)

    @property
    def bit_pnt_cnt(self):
        """Point count for shape bits, by ID. The array columns are shape_id,
        bit sequence and point count for bit.
        """
        b_ids = self.bit_ids
        b_seq = self.bit_seq  # self.IFT[5]
        return np.array([(b_ids[i], b_seq[i], len(p))
                         for i, p in enumerate(self.bits)])

    #
    # ---- coordinates: shape, parts, bits
    @property
    def shapes(self):
        """Shapes consist of points.  They can be singlepart or multipart.

        Returns
        -------
        Either an object array or ndarray depending on the shape of the parts.
        The overall array should be an object array.

        Notes
        -----
        Pull out and ravel the from-to point IDs.  Slice the first and last
        locations. Finally, slice the actual coordinates for each range.
        """
        c = [self.FT[self.IDs == i].ravel() for i in self.U]
        c1 = [(i[0], i[-1]) for i in c]
        return np.array([np.array(self[f:t]) for f, t in c1])

    @property
    def parts(self):
        """Deconstruct the 2D array into its parts.  Returns an array of
        ndarrays and/or object arrays.
        """
        uni = np.unique(self.IP, axis=0)
        ift = self.IFT
        out = []
        for u in uni:
            c = ift[(self.IP == u).all(axis=1)]
            c1 = [(i[1], i[2]) for i in c]
            s = np.vstack([self[f:t] for f, t in c1])
            out.append(s)
        return np.array(out)

    @property
    def bits(self):
        """Deconstruct the 2D array then parts of a piece if a piece contains
        multiple parts.  Keeps all rings.
        """
        fr_to = self.FT.tolist()
        out = np.asarray([self[f:t] for f, t in fr_to])
        return out

    def properties(self):
        """Return a list of all the @property methods
        """
        print(self.H)

    #
    # ---- methods and derived properties section ----------------------------
    # ---- (1) slicing, sampling equivalents
    #
    def first_bit(self, asGeo=True):
        """Get the first bit of multipart shapes and/or shapes with holes,
        discarding the holes.  Fixes the IFT to account for the removed points.
        """
        info = "{} first part".format(str(self.Info))
        ift_s = self.IFT[self.Bit == 0]
        fr_to = ift_s[:, 1:3]
        a_2d = [self.XY[f:t] for f, t in fr_to]
        if asGeo:
            a_2d = np.concatenate(a_2d, axis=0)
            ft = np.concatenate(([0], ift_s[:, 2] - ift_s[:, 1]))
            c = np.cumsum(ft)
            ift_s[:, 1] = c[:-1]
            ift_s[:, 2] = c[1:]
            return npg.Geo(a_2d, ift_s, self.K, self.XT, info)
        return a_2d

    def first_part(self, asGeo=True):
        """Return the first part of a multipart shape and/or a shape with
        holes, keeping the holes.
        Fix the IFT to account for the removed points.
        """
        info = "{} first part".format(str(self.Info))
        ift_s = self.IFT[self.PID == 1]
        f_t = ift_s[:, 1:3]
        a_2d = [self[f:t] for f, t in f_t]
        if asGeo:
            a_2d = np.concatenate(a_2d, axis=0)
            ft = np.concatenate(([0], ift_s[:, 2] - ift_s[:, 1]))
            c = np.cumsum(ft)
            ift_s[:, 1] = c[:-1]
            ift_s[:, 2] = c[1:]
            return npg.Geo(a_2d, ift_s, self.K, self.XT, info)
        return a_2d

    def get_shape(self, ID=None, asGeo=True):
        """Return a Geo or ndarray associated with the feature ID.  The ID must
        exist, otherwise None is returned and a warning is issued.

        Parameters
        ----------
        ID : integer
            A single integer value.
        asGeo : Boolean
            True, returns an updated Geo array.  False returns an ndarray or
            object array.
        """
        if not isinstance(ID, (int)):
            print("An integer ID is required. See ``pull`` for multiple IDs.")
            return None
        if ID not in self.U:
            print("The ID specified is not in present in the list of IDs.")
            return None
        shp = self[self.pnt_ids == ID]
        if asGeo:
            f_t = self.IFT[self.IDs == ID]
            st = f_t[:, 1][0]
            f_t[:, 1] = f_t[:, 1] - st
            f_t[:, 2] = f_t[:, 2] - st
            return Geo(shp, f_t, self.K, self.XT)
        return np.asarray(shp)

    def pull_shapes(self, ID_list=None, asGeo=True):
        """Pull multiple shapes, in the order provided.  The original IDs are
        kept but the point sequence is altered to reflect the new order.

        Parameters
        ----------
        ID_list : array-like
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
        ID_list = np.asarray(ID_list)
        if (ID_list.ndim and ID_list.size) == 0:
            print("An array/tuple/list of IDs are required, see ``get``.")
            return None
        if not np.all([a in self.IDs for a in ID_list]):
            print("Not all required IDs are in the list provided")
            return None
        w = np.isin(self.IDs, ID_list)
        parts = self.XY[np.isin(self.pnt_ids, w)]
        if asGeo:
            ifts = self.IFT[w]
            return Geo(parts, ifts, self.K, self.XT)
        return parts

    def split_by(self, splitter="bit"):
        """Split points by shape or by parts for each shape. **keep for now**
        Use self.parts or self.shapes directly.

        Parameters
        ----------
        splitter : b, p, s
            split by (b)it, (p)art (s)hape
        """
        case = splitter[0].lower()
        if case == "b":
            return self.bits
        elif case == "p":
            return self.parts
        elif case == "s":
            return self.shapes
        return

    def outer_rings(self, asGeo=False):
        """Collect the outer ring of a polygon shape.  Returns a list of
        ndarrays or optionally a new Geo array.
        """
        if self.K != 2:
            print("Polygons required...")
            return None
        if asGeo:
            return self.first_bit(True)
        return self.first_bit(False)

    def original_arrays(self):
        """Convert the Geo arrays back to the original input arrays
        """
        g0 = self.shift(self.LL[0], self.LL[1])
        b = g0.bits
        w = np.where(self.CW == 1)[0]
        arrs = np.split(b, w)[1:]
        return arrs

    # ---- (2) areas, centrality, lengths/perimeter for polylines/polygons
    #
    def areas(self, by_shape=True):
        """Area for the sub arrays using _e_area for the calculations.  Uses
        ``_area_part_`` to calculate the area.
        ``by_shape=True`` reduces the redundancy in the areas which are
        repeated for each part.
        """
        def _area_bit_(ar):
            """Mini e_area, used by areas and centroids"""
            x0, y1 = (ar.T)[:, 1:]
            x1, y0 = (ar.T)[:, :-1]
            e0 = np.einsum('...i,...i->...i', x0, y0)
            e1 = np.einsum('...i,...i->...i', x1, y1)
            return np.sum((e0 - e1)*0.5)
        # ----
        if self.K != 2:
            print("Polygons required")
            return None
        bit_totals = [_area_bit_(i) for i in self.bits]  # by bit
        if by_shape:
            b_ids = self.bit_ids
            return np.bincount(b_ids, weights=bit_totals)[self.U]
        return bit_totals

    def lengths(self, by_shape=True):
        """Polyline lengths or polygon perimeter."""
        def _cal(a):
            """Perform the calculation, mini-e_leng."""
            diff = a[:-1] - a[1:]
            return np.sum(np.sqrt(np.einsum('ij,ij->i', diff, diff)))
        # ----
        if self.K not in (1, 2):
            print("Polyline/polygon representation is required.")
            return None
        bit_lengs = [_cal(i) for i in self.bits]
        if by_shape:
            b_ids = self.bit_ids
            return np.bincount(b_ids, weights=bit_lengs)[self.U]
        return bit_lengs

    def cent_shapes(self):
        """Return the center of all a shapes point, whether multipart or shapes
        with holes.
        """
        shps = self.shapes
        if self.K == 2:
            return np.stack([np.mean(s[:-1], axis=0) for s in shps])
        return np.stack([np.mean(s, axis=0) for s in shps])

    def cent_parts(self):
        """ """
        o_rings = self.first_bit(False)
        if self.K == 2:
            return np.stack([np.mean(r[:-1], axis=0) for r in o_rings])
        return np.stack([np.mean(r, axis=0) for r in o_rings])

    def centroids(self):
        """Centroid of the polygons.  Uses ``_area_centroid_`` to calculate
        values for each shape part.  The centroid is weighted by area for
        multipart features.
        """
        # ----
        def weighted(x_y, Ids, areas):
            """Weighted coordinate by area, x_y is either the x or y."""
            w = x_y * areas                # area weighted x or y
            w1 = np.bincount(Ids, w)[1:]  # [Ids] weight divided by bin size
            ar = np.bincount(Ids, areas)[1:]  # [I]  # areas per bin
            return w1/ar

        # ----
        def _area_centroid_(a):
            """Calculate area and centroid for a singlepart polygon, `a`.  Also
            as geom._area_centroid_
            """
            x0, y1 = (a.T)[:, 1:]
            x1, y0 = (a.T)[:, :-1]
            e0 = np.einsum('...i,...i->...i', x0, y0)
            e1 = np.einsum('...i,...i->...i', x1, y1)
            t = e1 - e0
            area = np.sum((e0 - e1)*0.5)
            x_c = np.sum((x1 + x0) * t, axis=0) / (area * 6.0)
            y_c = np.sum((y1 + y0) * t, axis=0) / (area * 6.0)
            return area, np.asarray([-x_c, -y_c])
        # ----
        if self.K != 2:
            print("Polygons required.")
            return None
        centr = []
        areas = []
        ids = self.part_ids  # unique shape ID values
        uni = np.unique(ids)
        for ID in uni:
            parts_ = self.part_IFT[self.part_ids == ID]
            out = np.asarray([np.asarray(self.XY[p[1]:p[2]]) for p in parts_])
            for prt in out:
                area, cen = _area_centroid_(prt)  # ---- determine both
                centr.append(cen)
                areas.append(area)
        centr = np.asarray(centr)
        areas = np.asarray(areas)
        xs = weighted(centr[:, 0], ids, areas)
        ys = weighted(centr[:, 1], ids, areas)
        return np.array(list(zip(xs, ys)))

    # ---- (3) extents and extent shapes
    #
    def aoi_extent(self):
        """The full geographic extent of the dataset `aoi` (area of interest).
        """
        return np.concatenate((np.min(self.XY, axis=0),
                               np.max(self.XY, axis=0)))

    def aoi_rectangle(self):
        """The polygon bounds derived from the aoi_extent.
        """
        L, B, R, T = self.aoi_extent()
        return np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])

    def extents(self, splitter="part"):
        """Extents are returned as L(eft), B(ottom), R(ight), T(op).

        Parameters
        ----------
        splitter : b, p, s
            split by (b)it, (p)art (s)hape
        """
        def _extent_(i):
            """Extent of a sub-array in an object array"""
            i = np.atleast_2d(i)
            return np.concatenate((np.min(i, axis=0), np.max(i, axis=0)))
        # ----
        if self.N == 1:
            splitter = "bit"
        parts = self.split_by(splitter)
        return np.asarray([_extent_(p) for p in parts])

    def extent_centers(self, splitter="shape"):
        """Return extent centers.

        Parameters
        ----------
        splitter : b, p, s
            split by (b)it, (p)art (s)hape
        """
        ext = self.extents(splitter)
        xs = (ext[:, 0] + ext[:, 2])/2.
        ys = (ext[:, 1] + ext[:, 3])/2.
        return np.asarray(list(zip(xs, ys)))

    def extent_pnts(self, splitter="shape", as_Geo_array=False):
        """Derives the LB and RT point for a shape geometry.

        Parameters
        ----------
        splitter : b, p, s
            split by (b)it, (p)art (s)hape
        """
        ext_polys = []
        for ext in self.extents(splitter):
            L, B, R, T = ext
            poly = np.array([[L, B], [R, T]])
            ext_polys.append(poly + self.LL)
        if as_Geo_array:
            ext_polys = arrays_to_Geo(ext_polys, kind=2, info="extent pnts")
        return ext_polys

    def extent_rectangles(self, splitter='shape', as_Geo_array=False):
        """Return extent polygons for for the whole shape of the shape by bit.
        Points are ordered clockwise from the bottom left, with the first and
        last points the same.  Requires an Advanced license in Pro for
        equivalent functionality.

        See Also
        --------
        ``aoi_extent`` and ``aoi_rectangles``
        """
        ext_polys = []
        for ext in self.extents(splitter):
            L, B, R, T = ext
            poly = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
            ext_polys.append(poly + self.LL)
        if as_Geo_array:
            ext_polys = arrays_to_Geo(ext_polys, kind=2, info="extent polys")
        return ext_polys

    # ---- (4) maxs, mins, means for all features
    # useful # b_id = self.IDs[self.Bit == 0]
    def maxs(self, by_bit=False):
        """Maximums per feature or part."""
        if len(self.shp_part_cnt) == 1:
            return np.asarray(np.max(self, axis=0))
        if by_bit:
            b = self.bits[self.Bit == 0]
            return np.asarray([np.max(i, axis=0) for i in b])
        return np.asarray([np.max(i, axis=0) for i in self.shapes])

    def mins(self, by_bit=False):
        """Minimums per feature or part."""
        if len(self.shp_part_cnt) == 1:
            return np.asarray(np.min(self, axis=0))
        if by_bit:
            b = self.bits[self.Bit == 0]
            return [np.min(i, axis=0) for i in b]
        return np.asarray([np.min(i, axis=0) for i in self.shapes])

    def means(self, by_bit=False, remove_dups=True):
        """Mean per feature or part, optionally keep duplicates."""
        if len(self.shp_part_cnt) == 1:
            chunks = [self]
        else:
            chunks = self.shapes
        if remove_dups:
            chunks = [np.unique(i, axis=0) for i in chunks]
        return np.asarray([np.mean(i, axis=0) for i in chunks])

    # ---- methods/properties that use functions from npg_geom
    #
    # ---- (1) **is** section, condition/case checking, kept to a minimum
    def is_clockwise(self, is_closed_polyline=False):
        """Utilize `shoelace` area calculation to determine whether polygon
        rings are clockwise or not.  If the geometry represent a closed-loop
        polyline, then set the `is_closed_polyline` to True.  Validity of the
        geometry is not checked.
        """
        msg = "Polygons or closed-loop polylines are required."
        if self.K not in (1, 2):
            print(msg)
            return None
        if (self.K == 1) and (not is_closed_polyline):
            print(msg)
            return None
        ids = self.bit_ids
        cw = self.CW
        return uts(np.asarray(list(zip(ids, cw))), names=['IDs', 'Clockwise'])

    def is_convex(self):
        """Return True for convex, False for concave.  Holes are excluded
        The first part of multipart shapes are used. Duplicate start-end
        points removed prior to cross product
        """
        def _x_(a):
            """Cross product.  Concatenate version is slightly faster"""
            ba = a - np.concatenate((a[-1][None, :], a[:-1]), axis=0)
            bc = a - np.concatenate((a[1:], a[0][None, :]), axis=0)
            return np.cross(ba, bc)
        # ----
        if self.K != 2:
            print("Polygons are required.")
            return None
        f_bits = self.first_bit(False)
        check = [_x_(p[:-1]) for p in f_bits]  # cross-product
        return np.array([np.all(np.sign(i) >= 0) for i in check])

    def is_multipart(self, as_structured=False):
        """For each shape, returns whether it has multiple parts.  An ndarray
        is returned with the first column being the shape number and the second
        is coded as 1 for True and 0 for False.
        """
        partcnt = self.shp_part_cnt
        w = np.where(partcnt[:, 1] > 1, 1, 0)
        ids = self.U
        arr = np.array(list(zip(ids, w)))
        if as_structured:
            dt = np.dtype([('IDs', '<i4'), ('Parts', '<i4')])
            return uts(arr, dtype=dt)
        return arr

    # ---- (2) angles, convexity and other properties that
    #
    def polyline_angles(self, fromNorth=False):
        """Polyline/segment angles.  *** needs work ***
        """
        s = self.polys_to_segments()
        dxy = s[:, -2:] - s[:, :2]
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        if fromNorth:
            ang = np.mod((450.0 - ang), 360.)
        return ang

    def polygon_angles(self, inside=True, in_deg=True):
        """Sequential 3 point angles from a poly* shape.  The outer ring for
        each part is used.  see ``_angles2_`` and ``first_bit``.
        """
        f_bits = self.first_bit(False)
        return [geom._angles2_(p, inside, in_deg) for p in f_bits]

    # ---- (3) return altered geometry
    #
    def moveto(self, x=0, y=0):
        """Shift the dataset so that the origin is the lower-left corner.
        see also ``shift`` and ``translate``.
        """
        dx, dy = x, y
        if dx == 0 and dy == 0:
            dx, dy = np.min(self.XY, axis=0)
        return Geo(self.XY + [-dx, -dy], self.IFT, self.K, self.XT)

    def shift(self, dx=0, dy=0):
        """see ``translate``"""
        return Geo(self.XY + [dx, dy], self.IFT, self.K, self.XT)

    def translate(self, dx=0, dy=0):
        """Move/shift/translate by dx, dy to a new location."""
        return Geo(self.XY + [dx, dy], self.IFT, self.K, self.XT)

    def rotate(self, as_group=True, angle=0.0, clockwise=False):
        """Rotate shapes about the group center or individually.

        Rotation is done by npg_geom._rotate_ and a new geo array is returned.
        """
        if clockwise:
            angle = -angle
        angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, s), (-s, c)))
        out = geom._rotate_(self, R, as_group)
        info = "{} rotated".format(self.Info)
        out = np.vstack(out)
        return Geo(out, self.IFT, self.K, self.XT, info)

    # ---- (4) bounding circle, convex_hulls, minimum area rectangle, Delaunay
    # **see also** extent properties above
    #
    def bounding_circles(self, angle=5, return_xyr=False):
        """Bounding circles for features.

        Parameters
        ----------
        angle : number
            Angles to form n-gon.  A value of 10 will yield 36 point circle.
        return_xyr : boolean {optional}
            Return circle center and radius.

        Returns
        -------
        Circle points and optionally, the circle center and radius.
        """
        xyr = [sc.small_circ(s) for s in self.shapes]
        circs = []
        for vals in xyr:
            x, y, r = vals
            circs.append(sc.circle_mini(r, angle, x, y))
        circs = [circ + self.LL for circ in circs]
        circs = arrays_to_Geo(circs, kind=2, info="circs")
        if return_xyr:
            return xyr, circs
        return circs

    def convex_hulls(self, by_bit=False, threshold=50):
        """Convex hull for shapes.  Calls ``_ch_`` to control method used.

        Parameters
        ----------
        by_part : boolean
            False for whole shape.  True for shape parts if present.
        threshold : integer
            Points... less than threshold uses simple CH method, greater than,
            uses scipy.
        """
        # ----
        shps = self.first_bit(False) if by_bit else self.shapes
        # ---- run convex hull, _ch_, on point groups
        ch_out = [geom._ch_(s, threshold) for s in shps]
        for i, c in enumerate(ch_out):  # check for closed
            if np.all(c[0] != c[-1]):
                ch_out[i] = np.vstack((c, c[0]))
        out = [i + self.LL for i in ch_out]
        out = arrays_to_Geo(out, kind=2, info="convex hulls")
        return out

    def min_area_rect(self, as_structured=False):
        """Determines the minimum area rectangle for a shape represented
        by a list of points.  If the shape is a polygon, then only the outer
        ring is used.  This is the MABR... minimum area bounding rectangle.
       """
        def _extent_area_(a):
            """Area of an extent polygon."""
            LBRT = np.concatenate((np.min(a, axis=0), np.max(a, axis=0)))
            dx, dy = np.diff(LBRT.reshape(2, 2), axis=0).squeeze()
            return dx * dy, LBRT

        def _extents_(a):
            """Extents are returned as L(eft), B(ottom), R(ight), T(op)."""
            def _sub_(i):
                """Extent of a sub-array in an object array"""
                return np.concatenate((np.min(i, axis=0), np.max(i, axis=0)))
            p_ext = [_sub_(i) for i in a]
            return np.asarray(p_ext)
        # ----
        # chs = self.convex_hulls(False, 50)
        chs = [geom._ch_(i) for i in self.outer_rings()]
        ang_ = [geom._angles2_(i) for i in chs]
        xt = _extents_(chs)
        cent_ = np.c_[np.mean(xt[:, 0::2], axis=1),
                      np.mean(xt[:, 1::2], axis=1)]
        rects = []
        for i, p in enumerate(chs):
            # ---- np.radians(np.unique(np.round(ang_[i], 2))) # --- round
            uni_ = np.radians(np.unique(ang_[i]))
            area_old, LBRT = _extent_area_(p)
            for angle in uni_:
                c, s = np.cos(angle), np.sin(angle)
                R = np.array(((c, s), (-s, c)))
                ch = np.einsum('ij,jk->ik', p - cent_[i], R) + cent_[i]
                area_, LBRT = _extent_area_(ch)
                Xmin, Ymin, Xmax, Ymax = LBRT
                vals = [area_, Xmin, Ymin, Xmax, Ymax]
                if area_ < area_old:
                    area_old = area_
                    Xmin, Ymin, Xmax, Ymax = LBRT
                    vals = [area_, Xmin, Ymin, Xmax, Ymax]   # min_area,
            rects.append(vals)
        rects = np.asarray(rects)
        if as_structured:
            dt = np.dtype([('Rect_area', '<f8'), ('Xmin', '<f8'),
                           ('Ymin', '<f8'), ('Xmax', '<f8'), ('Ymax', '<f8')])
            return uts(rects, dtype=dt)
        return rects

    def triangulate(self, by_bit=False, as_polygon=True):
        """Delaunay triangulation for point groupings."""
        if by_bit:
            shps = self.bits
        else:
            shps = self.shapes
        out = [geom._tri_pnts_(s) for s in shps]
        kind = 2 if as_polygon else 1
        g, ift, extent = _arr_ift_(out)
        return Geo(g, ift, Kind=kind, Extent=self.XT, Info="triangulation")

    #
    # ---- (5) conversions ---------------------------------------------
    #
    def fill_holes(self):
        """Fill holes in polygon shapes.  Returns a Geo class."""
        if self.K < 2:
            print("Polygon geometry required.")
            return None
        a_2d = np.vstack(self.first_bit(False))
        tmp_ift = self.IFT[self.Bit == 0]
        tmp_ft = [(j - i) for i, j in tmp_ift[:, 1:3]]
        id_too = np.zeros((len(tmp_ft), 2), dtype=np.int64)
        cs = np.cumsum(tmp_ft)
        id_too[1:, 0] = cs[:-1]
        id_too[:, 1] = cs
        info = "{} fill_holes".format(self.Info)
        tmp_ift[:, 1:3] = id_too
        return npg.Geo(a_2d, IFT=tmp_ift, Kind=2, Extent=self.XT, Info=info)

    def holes_to_shape(self):
        """Return holes in polygon as shapes.  Returns a Geo class or None."""
        id_too = []
        # get the counter-clockwise shapes
        cw = self.CW
        if not np.any(cw):
            print("No holes... bailing")
            return None
        ccw = np.zeros(self.CW.shape, np.bool_)
        ccw[:] = cw
        ccw = ~ccw
        if self.K < 2:
            print("Polygon geometry required.")
            return None
        # ---- pull out all the ccw bits... aka, holes and reorder the rings
        ccw_bits = self.bits[ccw]
        a_2d = np.vstack([i[::-1] for i in ccw_bits])
        # ---- fix the IFT
        tmp_ift = self.IFT
        tmp_ift = tmp_ift[ccw]
        tmp_ft = [(j - i) for i, j in tmp_ift[:, 1:3]]
        id_too = np.zeros((len(tmp_ft), 2), dtype=np.int64)
        cs = np.cumsum(tmp_ft)
        id_too[1:, 0] = cs[:-1]
        id_too[:, 1] = cs
        tmp_ift[:, 1:3] = id_too
        if len(a_2d) < 1:  # ---- if empty
            return None
        return Geo(a_2d, IFT=tmp_ift, Kind=2,
                   Extent=self.XT, Info="filled holes")

    def multipart_to_singlepart(self, info=""):
        """Convert multipart shapes to singleparts and return a new Geo array.
        """
        ift = np.copy(self.IFT)                  # copy! the IFT don't reuse
        data = self.XY                           # reuse the data
        w = np.where(ift[:, -1] == 0)[0]
        dif = np.diff(w)
        seq = np.arange(len(w) - 1)
        ids = np.repeat(seq, dif)
        ids = np.concatenate((ids, [ids[-1] + 1]))
        ift[:, 0] = ids                          # reset the ids
        ift[:, -2] = np.ones(len(ift))           # reset the part ids to 1
        return Geo(data, IFT=ift, Kind=self.K, Extent=self.XT, Info=info)

    def od_pairs(self):
        """Construct origin-destination pairs for traversing around the
        perimeter of polygons, along polylines or between point sequences.

        Returns
        -------
        An object array of origin-destination pairs is returned.

        See Also
        --------
        polys_to_segments
        """
        od = [np.concatenate((p[:-1], p[1:]), axis=1) for p in self.bits]
        return np.asarray(od)

    def polylines_to_polygons(self):
        """Return a polygon Geo type from a polyline Geo.  It is assumed that
        the polylines form closed-loops, otherwise use ``close_polylines``.
        """
        if self.K == 2:
            print("Already classed as a polygon.")
            return self
        polygons = self.copy()
        polygons.K = 2
        return polygons

    def polygons_to_polylines(self):
        """Return a polyline Geo type from a polygon Geo.
        """
        if self.K == 1:
            print("Already classed as a polyline.")
            return self
        polylines = Geo(self.XY, self.IFT, Kind=1, Extent=self.XT,
                        SR=self.SR)
        return polylines

    def boundary(self):
        """alias for polygons_to_polylines"""
        return self.polygons_to_polylines()

    def polys_to_points(self, keep_order=True, as_structured=False):
        """Convert all feature vertices to an ndarray of unique points.
        Optionally, retain point order."""
        if as_structured:
            arr = self + self.LL
            return geom._polys_to_unique_pnts_(arr, as_structured=True)
        uni, idx = np.unique(self, True, axis=0)
        if keep_order:
            uni = self[np.sort(idx)]
        return uni

    def close_polylines(self, out_kind=1):
        """Attempt to produce closed-loop polylines (1) or polygons (2)
        from polylines.  Multipart features are converted to single part.
        """
        polys = []
        for s in self.bits:  # shape as bits
            if len(s) > 2:
                if np.all(s[0] == s[-1]):
                    polys.append(s)
                else:
                    polys.append(np.concatenate((s, s[..., :1, :]), axis=0))
        g, ift, extent = _arr_ift_(polys)
        return Geo(g, ift, self.K, self.XT, "Closed polylines.")

    def densify_by_distance(self, spacing=1):
        """Densify poly features by a specified distance.  Converts multipart
        to singlepart features during the process.
        Calls ``_pnts_on_line_`` for Geo bits.
        """
        polys = [geom._pnts_on_line_(a, spacing) for a in self.bits]
        g, ift, extent = _arr_ift_(polys)
        return Geo(g, ift, self.K, self.XT, "Densify by distance")

    def densify_by_percent(self, percent=50):
        """Densify poly features by a percentage for each segment.  Converts
        multipart to singlepart features during the process.
        Calls ``_percent_along``
        """
        bits = self.bits
        polys = [geom._pnts_on_line_(a, spacing=percent, is_percent=True)
                 for a in bits]
        polys = [a + self.LL for a in polys]
        g0, ift, extent = _arr_ift_(polys)
        return Geo(g0, ift, self.K, self.XT, "Densify by percent")

    def pnt_on_poly(self, by_dist=True, val=1, as_structured=True):
        """Point on polyline/polygon by distance or percent.

        Parameters
        ----------
        by_dist : boolean
            Enter the actual planar distance if True.  If False, then enter the
            percentage between 0 and 100.
        Emulates
        `arcpy Polyline class, positionAlongLine (value, {use_percentage})
        <https://pro.arcgis.com/en/pro-app/arcpy/classes/polyline.htm>`_.
        """
        dt = [('OID_', '<i4'), ('X_', '<f8'), ('Y_', '<f8')]
        if by_dist:
            r = np.asarray(
                [geom._dist_along_(a, dist=val) for a in self.bits]
                )
        else:
            val = min(abs(val), 100.)
            r = np.asarray(
                [geom._percent_along_(a, percent=val) for a in self.bits]
                )
        if as_structured:
            z = np.empty((r.shape[0], ), dtype=dt)
            z['OID_'] = np.arange(r.shape[0])
            z['X_'] = r[:, 0]
            z['Y_'] = r[:, 1]
            return z
        else:
            return r

    # ---- segments for poly* boundaries
    #
    def polys_to_segments(self, as_basic=True, as_3d=False):
        """Segment poly* structures into o-d pairs from start to finish.

        Parameters
        ----------
        as_basic : boolean
            True returns the basic od pairs as an Nx5 array in the form
            [X_orig', Y_orig', 'X_orig', 'Y_orig'] as an ndarray.
            If False, the content is returned as a structured array with the
            same content and ids and length.
        as_3d : boolean
            True, the point pairs are returned as a 3D array in the form
            [[X_orig', Y_orig'], ['X_orig', 'Y_orig']], without the distances.

        Notes
        -----
        Use ``prn_tbl`` if you want to see a well formatted output.
        """
        if self.K not in (1, 2):
            print("Poly* features required.")
            return None
        b_vals = [b + self.LL for b in self.bits]  # shift to original extent
        fr_to = np.concatenate([np.concatenate((b[:-1], b[1:]), axis=1)
                                for b in b_vals], axis=0)
        # ---- shortcut to 3d from-to representation
        if as_3d:
            fr_to = fr_to[:, :4]
            s0, s1 = fr_to.shape
            return fr_to.reshape(s0, s1//2, s1//2)
        # ---- basic return as ndarray used by common_segments
        if as_basic:
            return fr_to
        # ----structured array section
        # add bit ids and lengths to the output array
        b_ids = self.bit_ids
        segs = np.asarray([[b_ids[i], len(b) - 1]
                           for i, b in enumerate(b_vals)])
        s_ids = np.concatenate([np.repeat(i[0], i[1]) for i in segs])
        dist = (np.sqrt(np.sum((fr_to[:, :2] - fr_to[:, 2:4])**2, axis=1)))
        fr_to = np.hstack((fr_to, s_ids.reshape(-1, 1), dist.reshape(-1, 1)))
        dt = np.dtype([('X_orig', 'f8'), ('Y_orig', 'f8'),
                       ('X_dest', 'f8'), ('Y_dest', 'f8'),
                       ('Orig_id', 'i4'), ('Length', 'f8')])
        fr_to = uts(fr_to, dtype=dt)
        return repack_fields(fr_to)

    def common_segments(self):
        """Return the common segments in poly features.  Result is an array of
        from-to pairs of points.  ft, tf pairs are evaluated to denote common
        and duplicates.
        """
        h = self.polys_to_segments(as_basic=True)  # returns Nx5 array
        if h is None:
            return None
        h_0 = uts(h)
        names = h_0.dtype.names
        h_1 = h_0[list(names[2:4] + names[:2])]  # x_to, y_to and x_fr, y_fr
        idx = np.isin(h_0, h_1)
        common = h_0[idx]
        return _fill_float_array(common)  # stu(common)

    def unique_segments(self):
        """Return the unique segments in poly features as an array of
        from-to pairs of points.
        """
        h = self.polys_to_segments()
        if h is None:
            return None
        h_0 = uts(h)
        names = h_0.dtype.names
        h_1 = h_0[list(names[-2:] + names[:2])]
        idx0 = ~np.isin(h_0, h_1)
        uniq01 = np.concatenate((h_0[idx0], h_0[~idx0]), axis=0)
        return _fill_float_array(uniq01)  # return stu(uniq01)

    # ---- (6) sort section -------------------------------------------------
    # Sorting the fc shape-related fields needs an advanced arcgis pro license.
    # The following applies to the various sort options.
    #
    def change_indices(self, new):
        """Return the old and new indices derived from the application of a
        function
        """
        if len(self.shp_ids) != len(new):
            print("Old and new ID lengths must be the same")
            return None
        dt = np.dtype([('Orig_ID', '<i4'), ('New_ID', '<i4')])
        out = np.asarray(list(zip(self.IDs, new)), dtype=dt)
        return out

    def point_indices(self, as_structured=False):
        """Returns the point ids and the feature that they belong to.
        """
        ids = np.arange(self.shape[0])
        p_ids = [np.repeat(i[0], i[2] - i[1]) for i in self.IFT]
        p_ids = np.concatenate(p_ids)
        if as_structured:
            dt = np.dtype([('Pnt_ID', '<i4'), ('Feature_ID', '<i4')])
            z = np.zeros((self.shape[0],), dtype=dt)
            z['Pnt_ID'] = ids
            z['Feature_ID'] = p_ids
        else:
            z = np.zeros((self.shape), dtype=np.int32)
            z[:, 0] = ids
            z[:, 1] = p_ids
        return z

    def sort_by_area(self, ascending=True, just_indices=False):
        """Sort the geometry by ascending or descending order by shape area.

        Parameters
        ----------
        ascending : boolean
            True, in ascending order of shape area. False, in descending order.
        just_indices : boolean
            True, returns a structured array of old and new IDs representing
            the change in order based on the area sort option.  The actual
            array is not returned.
        """
        vals = self.areas(by_shape=True)         # shape area
        idx = np.argsort(vals)                   # sort the areas
        sorted_ids = self.shp_ids[idx]           # use shape IDs not part IDs
        if not ascending:
            sorted_ids = sorted_ids[::-1]
        if just_indices:
            return self.change_indices(sorted_ids)
        sorted_array = self.pull_shapes(sorted_ids.tolist())
        return sorted_array

    def sort_by_length(self, ascending=True, just_indices=False):
        """Sort the geometry by ascending or descending order"""
        areas = self.lengths
        idx = np.argsort(areas)
        sorted_ids = self.IDs[idx]
        if not ascending:
            sorted_ids = sorted_ids[::-1]
        if just_indices:
            return self.change_indices(sorted_ids)
        sorted_array = self.pull_shapes(sorted_ids)
        return sorted_array

    def sort_by_extent(self, key=0, just_indices=False):
        """Sort the feature geometry based on the conditions outlined in the
        parameters.  The feature centers are used to determine sort order.

        Parameters
        ----------
        sort_type : int

        **Sorting... key - direction vector - azimuth**

        +------+---------+-------+
        | key  + vector  +azimuth|
        +======+=========+=======+
        |  0   | S to N  |    0  |
        +------+---------+-------+
        |  1   | SW - NE |   45  |
        +------+---------+-------+
        |  2   | W to E  |   90  |
        +------+---------+-------+
        |  3   | NW - SE |  135  |
        +------+---------+-------+
        |  4   | N to S  |  180  |
        +------+---------+-------+
        |  5   | NE - SW |  225  |
        +------+---------+-------+
        |  6   | E to W  |  270  |
        +------+---------+-------+
        |  7   | SE - NW |  315  |
        +------+---------+-------+

        Notes
        -----
        The key values are used to select the dominant direction vector.

        >>> z = sin(a) * [X] + cos(a) * [Y]  # - sort by vector

        `vector sort
        <https://gis.stackexchange.com/questions/60134/sort-a-feature-table-by
        -geographic-location>`_.
        """
        if key not in range(0, 8):
            print("Integer value between 0 and 7 inclusive required.")
            return None
        azim = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # azimuths
        val = np.radians(azim[key])              # sort angle in radians
        ext = self.extent_centers()              # get the extent centers
        ext_ids = self.shp_ids
        xs = ext[:, 0]
        ys = ext[:, 1]
        z = np.sin(val) * xs + np.cos(val) * ys  # - sort by vector
        idx = np.argsort(z)  # sort order
        sorted_ids = ext_ids[idx]
        if just_indices:
            return self.change_indices(sorted_ids)
        sorted_array = self.pull_shapes(sorted_ids)
        return sorted_array

    def sort_coords(self, x_ascending=True, y_ascending=True):
        """Sort points by coordinates.

        Parameters
        ----------
        x_ascending, y_ascending : boolean
            If False, sort is done in decending order by axis.

        See Also
        --------
        sort_by_extent :
            For polygons or polylines.
        """
        if x_ascending:
            if y_ascending:
                return self[np.lexsort((self[:, 1], self[:, 0]))]
            return self[np.lexsort((-self[:, 1], self[:, 0]))]
        if y_ascending:
            if x_ascending:
                return self[np.lexsort((self[:, 0], self[:, 1]))]
            return self[np.lexsort((-self[:, 0], self[:, 1]))]

    # ---- (7) info section -------------------------------------------------
    # ---- points, dupl_pnts

    def point_info(self, by_part=True):
        """Point count by feature or parts of feature.

        Parameters
        ----------
        by part: boolean
            True for each feature part or False for the whole feature.
        """
        chunks = self.split_by(by_part)
        return np.array([len(i) for i in chunks])

    def dupl_pnts(self, as_structured=True):
        """Duplicated points as a structured array."""
        uni, idx, cnts = np.unique(self, True, False, True, axis=0)
        dups = uni[cnts > 1]
        num = cnts[cnts > 1]
        if as_structured:
            dt = [('X_', '<f8'), ('Y_', '<f8'), ('Cnts', '<i4')]
            z = np.empty((dups.shape[0],), dtype=dt)
            z['X_'] = dups[:, 0]
            z['Y_'] = dups[:, 1]
            z['Cnts'] = num
            z = repack_fields(z)
            return z
        return dups, num


# ---- End of class definition -----------------------------------------------
#
# ---- (2) Geo from sequences ------------------------------------------------
#  Construct the Geo array from sequences.
#     (ndarrays, object arrays, nested lists, lists of arrays etcetera.
# ---- helper functions
#
def _area_part_(a):
    """Mini e_area, used by areas and centroids"""
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.sum((e0 - e1)*0.5)


def _arr_ift_(in_arrays):
    """Produce a 2D array stack of x,y points and an I(d) F(rom) T(o) list of
    the coordinate pairs in the resultant.

    Parameters
    ----------
    in_arrays : list, array
        The input data as a list of lists or arrays or an array.

    Notes
    -----
    Called by ``arrays_to_Geo``.
    Use ``fc_geometry`` to produce ``Geo`` objects directly from arcgis pro
    featureclasses.
    """
    id_too = []
    a_2d = []
    if isinstance(in_arrays, np.ndarray):
        if in_arrays.ndim == 2:
            in_arrays = [in_arrays]
    subs = []  # ** added
    for cnt, p in enumerate(in_arrays):
        p = np.asarray(p)
        kind = p.dtype.kind
        if (kind == 'O') or (len(p.shape) > 2):
            bits = []
            sub = []   # ** added
            b_id = 0
            for j in p:
                if len(j.shape) == 2:
                    bits.append(np.asarray(j).squeeze())  # append and no squ
                    x = [cnt, b_id, len(j)]
                    id_too.append(x)
                    b_id += 1
                elif (j.dtype.kind == 'O') or (len(j.shape) > 2):
                    for k in j:
                        bits.append(np.asarray(k).squeeze())
                        x = [cnt, b_id, len(k)]
                        id_too.append(x)
                        b_id += 1
            sub.append(np.vstack(bits))  # ** added
        elif kind in NUMS:
            sub = []
            p = p.squeeze()
            b_id = 0
            if len(p.shape) == 2:
                id_too.append([cnt, b_id, len(p)])
                sub.append(np.asarray(p))
            elif len(p.shape) == 3:
                for k in p:
                    id_too.append([cnt, b_id, len(k)])
                    b_id += 1
                sub.append([np.asarray(j) for i in p for j in i])
        subs = np.vstack(sub)
        a_2d.append(subs)
    a_stack = np.vstack(a_2d)
    id_too = np.vstack(id_too)
    ids = id_too[:, 0]
    part = id_too[:, 1] + 1
    id_prt = np.vstack((ids, part)).T
    uni, idx = np.unique(id_prt, True, axis=0)
    CW = np.zeros_like(ids)
    for i, k in enumerate(id_prt):
        if i in idx:
            CW[i] = 1
    too = np.cumsum(id_too[:, 2])
    frum = np.concatenate(([0], too))
    ar = np.where(CW == 1)[0]
    ar0 = np.stack((ar[:-1], ar[1:])).T
    pnt_nums = np.zeros_like(ids, dtype=np.int64)
    u, i, cnts = np.unique(ids, True, return_counts=True)
    pnt_nums = np.concatenate([np.arange(i) for i in cnts])
#    for (i, j) in ar0:
#        pnt_nums[i:j] = np.arange((j - i))
    IFT = np.array(list(zip(ids, frum, too, CW, part, pnt_nums)))
    extent = np.array([np.min(a_stack, axis=0), np.max(a_stack, axis=0)])
    a_stack = a_stack - extent[0]
    # recheck clockwise values for the array
    return a_stack, IFT, extent


# ---- main function
def arrays_to_Geo(in_arrays, kind=2, info=None):
    """Produce a Geo class object from a list/tuple of arrays.

    Parameters
    ----------
    in_arrays : list
        ``in_arrays`` can be created by adding existing 2D arrays to the list
        or produced from the conversion of poly features to arrays using
        ``poly2arrays``.
    Kind : integer
        Points (0), polylines (1) or polygons (2)

    Requires
    --------
    _arr_ift_

    Returns
    -------
    A ``Geo`` class object based on a 2D np.ndarray (a_2d) with an array of
    indices (IFT) delineating geometry from-to points for each shape and its
    parts.

    See Also
    --------
    **fc_geometry** to produce ``Geo`` objects directly from arcgis pro
    featureclasses.
    """
    a_2d, ift, extent = _arr_ift_(in_arrays)     # ---- call _arr_ift_
    rows, cols = ift.shape
    z0 = np.full((rows, 6), fill_value=-1, dtype=ift.dtype)
    z0[:, :cols] = ift
    # do the clockwise check here and correct any assumptions
    g = Geo(a_2d, z0, Kind=kind, Extent=extent, Info=info)
    if kind == 2:
        old_CW = g.CW
        CW_check = [1 if _area_part_(i) > 0 else 0 for i in g.bits]
        if not np.all(old_CW == CW_check):
            z0[:, 3] = CW_check
            g = Geo(a_2d, z0, Kind=kind, Extent=extent, Info=info)
    return g


# =========================================================================
def Update_Geo(a_2d, id_too=None, K=None, Info=None):
    """Create a new Geo array from a list of arrays.

    Parameters
    ----------
    a_2d : list/tuple/array
        Some form of nested 2D array-like structure that can be stacked.
    K : integer
        Points (0), polylines (1) or polygons (2).
    id_too : array-like
        If None, then the structure will be created.
    Info : text (optional)
        Provide any information that will help in identifying the array.

    Returns
    -------
    A new Geo array is returned given the inputs.
    """
    if not isinstance(a_2d, (np.ndarray, list, tuple)):
        return None
    if K not in (0, 1, 2):
        print("Output type not specified, or not in (0, 1, 2).")
        return None
    if id_too is None:  # create IFT structure
        if K == 0:
            ids = np.arange(0, len(a_2d))
            frum = ids
            too = np.arange(1, len(a_2d) + 1)
        else:
            id_too = [(i, len(a)) for i, a in enumerate(a_2d)]
            a_2d = np.concatenate(a_2d, axis=0)  # np.vstack(a_2d)
            id_too = np.asarray(id_too)
            ids = id_too[:, 0]
            too = np.cumsum(id_too[:, 1])
            frum = np.concatenate(([0], too))
    else:
        a_2d = np.concatenate(a_2d, axis=0)  # np.vstack(a_2d)
        ids = id_too[:, 0]
        too = np.cumsum(id_too[:, 1])
        frum = np.concatenate(([0], too))
    IFT = np.array(list(zip(ids, frum, too)))
    return Geo(a_2d, IFT, K, Info)


# ---- (3) Geo array to arrays
def Geo_to_arrays(g):
    """Geo array to array.

    Returns
    -------
    Most likely an object array of ndarrays (aka, a ragged array).  The
    coordinates are shifted back to their original extent
    """
    ift = g.IFT
    ids = g.IDs
    uniq_ids = g.U
    out = []
    for i in uniq_ids:
        shps = ift[ids == i]
        subs = []
        uniq, idx = np.unique(shps[:, 4], True)
        for u in uniq:
            sub = []
            s = shps[shps[:, 4] == u]
            for part in s:
                fr, too = part[1:3]
                xy = np.asarray(g.XY[fr:too] + g.LL)
                sub.append(xy)
            subs.append(np.asarray(sub))
        out.append(np.asarray(subs))
    return np.asarray(out)


# -------------------------------
def _fill_float_array(arr):
    """Fill an array of floats from a structured array floats, returning
    an ndarray.  This is a simplified version of ``stu``
    """
    names = arr.dtype.names
    n = len(names)
    a_2d = np.zeros((arr.size, n), dtype=np.float)
    for i, name in enumerate(names):
        a_2d[:, i] = arr[name]
    return a_2d


# ----------------------------------------------------------------------------
# ---- (4) other functions
#
def dirr(obj, colwise=False, cols=3, prn=True):
    """
    A `dir` variant for a variety of objects.  The output is sorted by columns
    or rows with `cols` of output to suite screen width.
    """
    from itertools import zip_longest as zl
    if ('Geo' in str(type(obj))) & (issubclass(obj.__class__, np.ndarray)):
        a = ['... Geo class ...']
        a.extend(sorted(list(set(dir(obj)).difference(set(dir(np.ndarray))))))
        a.extend(['... Geo helpers ...']*3 + sorted(npg.__all_geom__))
    else:
        a = dir(obj)
    w = max([len(i) for i in a])
    frmt = (("{{!s:<{}}} ".format(w)))*cols
    csze = len(a) / cols  # split it
    csze = int(csze) + (csze % 1 > 0)
    if colwise:
        a_0 = [a[i: i+csze] for i in range(0, len(a), csze)]
        a_0 = list(zl(*a_0, fillvalue=""))
    else:
        a_0 = [a[i: i+cols] for i in range(0, len(a), cols)]
    if hasattr(obj, '__module__'):
        args = ["-"*70, obj.__module__, obj.__class__]
    else:
        args = ["-"*70, type(obj), "py version"]
    txt_out = "\n{}\n| dir({}) ...\n|    {}\n-------".format(*args)
    cnt = 0
    for i in a_0:
        cnt += 1
        txt = "\n  ({:>03.0f})  ".format(cnt)
        frmt = (("{{!s:<{}}} ".format(w)))*len(i)
        txt += frmt.format(*i)
        txt_out += txt
    if prn:
        print(txt_out)
        return None
    return txt_out


def geo_info(g):
    """Differences between Geo and ndarray methods and properties."""
    from textwrap import indent, wrap
    arr_set = set(dir(g.base))
    geo_set = set(dir(g))
    srt = sorted(list(geo_set.difference(arr_set)))
    t = ", ".join([str(i) for i in srt])
    w = wrap(t, 70)
    print(">>> geo_info(g)\n... Geo methods and properties")
    for i in w:
        print(indent("{}".format(i), prefix="    "))
    print("\n>>> Geo.__dict_keys()")
    t = ", ".join(sorted(list(npg.Geo.__dict__.keys())))
    w = wrap(t, 70)
    for i in w:
        print(indent("{}".format(i), prefix="    "))


def check_geometry(self):
    """Run some geometry checks

    Performs clockwise/counterclockwise (CW/CCW) checks.  Outer rings must be
    CW.  Inner rings (aka, holes), are in CCW order.
    """
    # ----
    def _area_part_(a):
        """Mini e_area, used by areas and centroids"""
        x0, y1 = (a.T)[:, 1:]
        x1, y0 = (a.T)[:, :-1]
        e0 = np.einsum('...i,...i->...i', x0, y0)
        e1 = np.einsum('...i,...i->...i', x1, y1)
        return np.sum((e0 - e1)*0.5)

    # ----
    def _ccw_(a):
        """clockwise"""
        return 0 if _area_part_(a) > 0. else 0
        #
    m1 = """\
    These shapes have a first ring in CCW order or a CW ring is actually CCW.
    """
    ft = self.IFT[self.CW == 1][:, 1:3]
    check_0 = np.logical_and(self.CW == 0, self.Bit == 0)
    check_1 = [_ccw_(self[i: j]) for i, j in ft]
    if np.sum(check_0) > 0:
        print("\n{}\n... shapes {}\n".format(m1, self.IDs[check_0]))
        print("IFT information...\n")
        prn_tbl(self.IFT_str)
    elif np.sum(check_1) > 0:
        print("\n{}\n... shapes {}\n".format(m1, self.IDs[check_1]))
        print("IFT information...\n")
    else:
        print("no errors found")
    return None


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polylines2"
#    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygon2pnts"
