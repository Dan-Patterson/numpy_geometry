# -*- coding: utf-8 -*-
"""
=====
npGeo
=====

Script : npGeo.py
    A geometry class and methods based on numpy.

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-06-06
    Initial creation period 2019-05

Purpose : geometry tools
    A numpy geometry class, its properties and methods.

Notes
-----
**Class instantiation**

Quote from Subclassing ndarrays::

    As you can see, the object can be initialized in the __new__ method or the
    __init__ method, or both, and in fact ndarray does not have an __init__
    method, because all the initialization is done in the __new__ method.


**General notes**

The Geo class returns a 2D array of points which may consist of single or
multipart shapes with or without inner rings (holes).

The methods defined  in the Geo class allow one to operate on the parts of the
shapes separately or in combination.  Since the coordinate data are represented
as an Nx2 array, it is sometimes easier to perform calculations on the dataset
all at once using numpy ``nan`` functions.  For example, to determine the
minimum for the whole dataset:

>>> np.nanmin(Geo, axis=0)

All null points (nan, nan) are omitted from the calculations.


**Working with np.ndarray and geo class**

** Redo these all when done**

>>> arr_set = set(dir(g0.base))
>>> geo_set = set(dir(g0))
>>> sorted(list(geo_set.difference(arr_set)))
... ['FT', 'H', 'ID', 'IFT', 'K', 'N', 'X', 'XY', 'Y', 'Z', '__dict__',
...  '__module__', '_angle_', '_e_area', '_o_ring', 'aggregate', 'angles',
...  'areas', 'centers', 'centroids', 'chunk', 'cols', 'convex_hulls',
...  'extent_rect', 'extents', 'fill_holes', 'get', 'holes2shape',
...  'isconvex', 'lengths', 'maxs', 'means', 'mins', 'move', 'parts',
...  'parts', 'pnts', 'pull', 'rotate', 'rows', 'translate']

>>> geo_set = set(dir(Geo))
>>> sorted(list(geo_set.difference(arr_set)))
... ['__dict__', '__module__', '_angle_', '_e_area', '_o_ring_', 'aggregate',
...  'angles', 'areas', 'centers', 'centroids', 'chunk', 'extents', 'get',
...  'isconvex', 'lengths', 'maxs', 'means', 'mins', 'parts', 'parts',
...  'pnts', 'rotate', 'translate']

>>> Geo.__dict__.keys()  **REDO**
... dict_keys(['__module__', '__doc__', '__new__', '__array_finalize__',
...            '__array_wrap__', 'chunk', 'aggregate', 'parts', '_o_ring_',
...            '_angle_', 'angles', 'isconvex', '_e_area', 'areas', 'centers',
...            'centroids', 'lengths', 'extents', 'maxs', 'mins', 'means',
...            'parts', 'pnts', 'translate', 'rotate', 'get', '__dict__'])

**Useage of methods**

``s`` is a Geo instance with 2 shapes.  Both approaches yield the same results.

>>> Geo.centers(s)
array([[ 5.  , 14.93],
       [15.5 , 15.  ]])
>>> s.centers()
array([[ 5.  , 14.93],
       [15.5 , 15.  ]])

References
----------
`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.

**Sample file**

Saved in the arraytools folder and on GitHub::

    fname = 'C:/Git_Dan/arraytools/Data/geo_array.npz'
    npzfiles = np.load(fname")  # ---- the Geo, I(ds)F(rom)T(o) arrays
    npzfiles.files              # ---- will show ==> ['s2', 'IFT']
    s2 = npzfiles['s2']         # ---- slice by name from the npz file to get
    IFT = npzfiles['IFT']       #      each array
"""
# pylint: disable=R0904  # pylint issue
# pylint: disable=C0103  # invalid-name
# pylint: disable=E0611  # stifle the arcgisscripting
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=W0201  # attribute defined outside __init__... none in numpy
# pylint: disable=W0621  # redefining name

import sys
from textwrap import dedent
import numpy as np
#from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2, suppress=True,
                    threshold=100, formatter=ft)

script = sys.argv[0]  # print this should you need to locate the script

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS
TwoPI = np.pi*2.

__all__ = ['FLOATS', 'INTS', 'NUMS', 'TwoPI',  # constants
           'Geo', 'arrays_Geo', 'Geo_array',    # class and from, to methods
           '_angles_', '_ch_',                  # helpers
           '_o_ring_', '_pnts_on_line_',
           '_poly_segments_', '_updateGeo',
           'prn_geo', 'prn_tbl',                # printing
           '_simplify_lines_',
           '_nan_split_']    # unfinished
# ===========================================================================
# ---- Construct the Geo array from a list of ndarrays or an ndarray and
#       deconstruct, thevGeo array back to its origins
#
def arrays_Geo(in_arrays, Kind=2, Info=None):
    """Produce a Geo class object from a list/tuple of arrays.

    Parameters
    ----------
    in_arrays : list
        ``in_arrays`` can be created by adding existing 2D arrays to the list
         or produced from the conversion of poly features to arrays using
        ``poly2arrays``.
    Kind : integer
        Points (0), polylines (1) or polygons (2)

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
    id_too = []
    a_2d = []
    for i, p in enumerate(in_arrays):
        if p.ndim == 2:
            id_too.append([i, len(p)])
            a_2d.append(p)
        elif p.ndim == 3:
            id_too.extend([[i, len(k)] for k in p])
            a_2d.append([j for i in p for j in i])
    a_2d = np.vstack(a_2d)
    id_too = np.array(id_too)
    I = id_too[:, 0]
    too = np.cumsum(id_too[:, 1])
    frum = np.concatenate(([0], too))
    IFT = np.array(list(zip(I, frum, too)))
    new_geo = Geo(a_2d, IFT, Kind, Info)
    return new_geo

# ==== update Geo array, or create one from a list of arrays ================
#
def _updateGeo(a_2d, K=None, id_too=None, Info=None):
    """Create a new Geo from a list of arrays.

    Parameters
    ----------
    a_2d : list/tuple/array
        Some form of nested 2D array-like structure that can be stacked
    K : integer
        Points (0), polylines (1) or polygons (2)
    id_too : array-like
        If None, then the structure will be created.
    Info : text (optional)
        Provide any information that will help in identifying the array.
    Returns
    -------
    A new Geo array is returned given the inputs.
    """
    if K not in (0, 1, 2):
        print("Output type not specified")
        return None
    if id_too is None:
        id_too = [(i, len(a)) for i, a in enumerate(a_2d)]
#        id_too = []
#        for i, a in enumerate(a_2d):
#            id_too.append((i, len(a)))
    a_2d = np.vstack(a_2d)
    id_too = np.array(id_too)
    I = id_too[:, 0]
    too = np.cumsum(id_too[:, 1])
    frum = np.concatenate(([0], too))
    IFT = np.array(list(zip(I, frum, too)))
    new_geo = Geo(a_2d, IFT, K, Info)
    return new_geo


def Geo_array(in_geo):
    """Reconstruct the input arrays from the Geo array"""
    return np.asarray([in_geo.get(i) for i in range(in_geo.N)])

# ===========================================================================
#
class Geo(np.ndarray):
    """
    Point, polyline, polygon features represented as numpy ndarrays.
    The required inputs are created using ``fc_geometry(in_fc)`` or
    ``arrays_Geo``.

    Attributes
    ----------
    Normal ndarray parameters including shape, ndim, dtype

    - is_multipart : array of boolean
    - parts : ndarray of ids and counts
    - pnts : ndarray of ids and counts
    - shapes : the points for polyline, polygons
    - parts : multipart shapes and/or outer and inner rings for holes
    - bits : the final divisions to individual bits constituting the shape

    Parameters
    ----------
    arr : array-like
        A 2D array-like array-like sequence of points with shape (N, 2).
    IFT : array
        Defines, the I(d)F(rom)T(o) values identifying object parts if
        ``arr`` represents polylines or polygons.  Shape (N, 3) required.
    Kind : integer
        Shape dimension point (0), polyline (1), or polygon (2).

    Notes
    -----
    You can use ``arrays_Geo`` to produce the require 2D array from lists of
    array-like objects of the same dimension, or a single array.
    The IFT will be derived from breaks in the sequence and/or the
    presence of null points.

    >>> g = Geo(a, IFT)
    >>> g.__dict__.keys()
    dict_keys(['IDs', 'FT', 'IFT', 'K', 'Info', 'N', 'X', 'Y', 'XY', 'Z'])
    >>> sorted(g.__dict__.keys())
    ['FT', 'IDs', 'IFT', 'Info', 'K', 'N', 'X', 'XY', 'Y', 'Z']
    """
    # ----
    def __new__(cls, arr=None, IFT=None, Kind=2, Info=""):
        """
        Create a Geo array based on numpy ndarray.  The class focus is on
         geometry properties and methods.

        Requires
        --------
        arr : array-like
            A 2D array sequence of points with shape (N, 2)
        IFT : array-like
            Defines, the I(d)F(rom)T(o) values identifying object parts if
            ``arr`` represents polylines or polygons.  Shape (N, 3) required.
        Kind : integer
            Points (0), polylines/lines (1) and polygons (2)
        Info : string (optional)
            Optional information if needed.
        """
        msg = Geo.__new__.__doc__
        arr = np.asarray(arr)
        IFT = np.asarray(IFT)
        cond = [(arr.ndim != 2),
                (IFT.ndim != 2), (IFT.shape[-1] != 3),
                (Kind not in (0, 1, 2))]
        if all(cond):
            print(dedent(msg))
            return None
        # ----
        self = arr.view(cls)     # view as Geo class
        self.IFT = IFT
        self.IDs = IFT[:, 0]
        self.FT = IFT[:, 1:]
        self.K = Kind
        self.Info = Info
        # --- other properties
        self.N = len(np.unique(self.IDs))  # sample size, unique shapes
        if self.shape[1] >= 2:             # X,Y and XY initialize
            self.X = self[:, 0]
            self.Y = self[:, 1]
            self.XY = self[:, :2]
        if self.shape[1] >= 3:   # add Z, although not implemented
            self.Z = self[:, 2]  # directly, but kept for future additions
        else:
            self.Z = None
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
        self.IDs = getattr(src_arr, 'IDs', None)
        self.FT = getattr(src_arr, 'FT', None)
        self.K = getattr(src_arr, 'K', None)
        self.Info = getattr(src_arr, 'Info', None)
        self.N = getattr(src_arr, 'N', None)
        self.X = getattr(src_arr, 'X', None)
        self.Y = getattr(src_arr, 'Y', None)
        self.XY = getattr(src_arr, 'XY', None)

    def __array_wrap__(self, out_arr, context=None):
        """Wrap it up"""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # ------------------------------------------------------------------------
    """To do list:
    distance for sure
    angle and distance to
    boundary... envelope done
    buffer
    clip
    contains
    cut
    """
    # ------------------- End of class definition ----------------------------
    # ---- basic shape properties and methods to subdivide Geo
    @property
    def is_multipart(self):
        """For each shape, returns whether it has multiple parts.  A ndarray
        is returned with the first column being the shape number and the second
        is coded as 1 for True and 0 for False
        """
        partcnt = self.part_cnt
        w = np.where(partcnt[:, 1] > 1, 1, 0)
        return np.array(list(zip(np.arange(len(w)), w)))

    @property
    def part_cnt(self):
        """Part count for shapes. Returns IDs and count array"""
        return np.vstack(np.unique(self.IDs, return_counts=True)).T

    @property
    def pnts(self):
        """Point count for shapes excluding null points."""
        return np.array([(i, len(p[~np.isnan(p[:, 0])]))
                         for i, p in enumerate(self.shapes)])

    @property
    def shapes(self):
        """Subdivide the array into shapes which may be singlepart or multipart
        Returns an object array or ndarray of points
        """
        uniq = np.unique(self.IDs)
        c = [self.FT[self.IDs == i].ravel() for i in uniq]
        c1 = [(min(i), max(i)) for i in c]
        return np.array([np.asarray(self[f:t]) for f, t in c1]).squeeze()

    @property
    def parts(self):
        """Deconstruct the 2D array into its parts, generally returning an
        object array.  The reverse is np.vstack(self)
        formally: return np.asarray([(self[f:t]) for f, t in self.FT]) but with
        additions to add the FT and IFT properties
        """
        xy = self.base
        if xy is None:
            xy = self.XY.view(np.ndarray)
        return np.asarray(np.split(xy, self.IFT[:, 2]))[:-1]

    @property
    def bits(self):
        """Deconstruct the 2D array then parts of a piece if
        a piece contains multiple parts.
        """
        out = []
        prts = self.parts
        for ply in prts:
            s = np.isnan(ply[:, 0])
            if np.any(s):
                w = np.where(s)[0]
                ss = np.split(ply, w)
                out.append(ss[0])
            else:
                out.append(ply)
        return np.asarray(out)
    #
    # ---- areas, centrality, lengths/perimeter for polylines/polygons
    #
    @property
    def areas(self):
        """Area for the sub arrays using _e_area for the calculations"""
        def _e_area(a):
            """Mini e_area, used by areas and centroids"""
            x0, y1 = (a.T)[:, 1:]
            x1, y0 = (a.T)[:, :-1]
            e0 = np.einsum('...i,...i->...i', x0, y0)
            e1 = np.einsum('...i,...i->...i', x1, y1)
            return np.nansum((e0 - e1)*0.5)
        # ----
        if self.K != 2:
            print("Polygons required")
            return None
        chunks = self.parts
        subs = [_e_area(i) for i in chunks]           # call to _e_area
        totals = np.bincount(self.IDs, weights=subs)  # weight by IDs' area
        return totals

    @property
    def centers(self):
        """Return the center of an array. Duplicates are removed and rings and
        multiparts are used in the coordinate average.
        """
        out = [np.nanmean(np.unique(c, axis=0), axis=0) for c in self.shapes]
        return np.asarray(out)

    @property
    def centroids(self):
        """Centroid of the polygons.
        """
        def _cal_(p):
            """calculate area and centroid"""
            #x, y = p.T
            x0, y1 = (p.T)[:, 1:]
            x1, y0 = (p.T)[:, :-1]
            e0 = np.einsum('...i,...i->...i', x0, y0)
            e1 = np.einsum('...i,...i->...i', x1, y1)
            area = np.nansum((e0 - e1)*0.5)
            t = e1 - e0
            area = np.nansum((e0 - e1)*0.5)
            x_c = np.nansum((x1 + x0) * t, axis=0) / (area * 6.0)
            y_c = np.nansum((y1 + y0) * t, axis=0) / (area * 6.0)
            return np.asarray([-x_c, -y_c]), area
        # ----
        def weighted(x_y, I, areas):
            """Weighted coordinate by area, x_y is either the x or y"""
            w = x_y * areas            # area weighted x or y
            w1 = np.bincount(I, w)     # weight divided by bin size
            ar = np.bincount(I, areas) # areas per bin
            return w1/ar
        # ----
        if self.K != 2:
            print("Polygons required")
            return None
        centr = []
        areas = []
        uni = np.unique(self.IDs)
        for ID in uni:
            parts_ = self.FT[self.IDs == ID]
            out = np.asarray([np.asarray(self.XY[p[0]:p[1]]) for p in parts_])
            for prt in out:
                cen, area = _cal_(prt)
                centr.append(cen)
                areas.append(area)
        centr = np.asarray(centr)
        areas = np.asarray(areas)
        xs = weighted(centr[:, 0], self.IDs, areas)
        ys = weighted(centr[:, 1], self.IDs, areas)
        return np.array(list(zip(xs, ys)))

    @property
    def lengths(self):
        """Poly lengths/perimeter"""
        def _cal(a):
            """Perform the calculation, mini-e_leng"""
            diff = a[:-1] - a[1:]
            return np.nansum(np.sqrt(np.einsum('ij,ij->i', diff, diff)))
        # ----
        if self.K not in (1, 2):
            print("polyline/polygon representation is required")
            return None
        lengs = [_cal(i) for i in self.parts]
        totals = np.bincount(self.IDs, weights=lengs)
        return np.asarray(totals)
    #
    # ---- methods -----------------------------------------------------------
    # ---- extents
    def AOI_extent(self):
        """Determine the full extent of the dataset.
        This is the A(rea) O(f) I(nterest)
        """
        return np.concatenate((np.nanmin(self.XY, axis=0),
                               np.nanmax(self.XY, axis=0)))

    def AOI_rectangle(self):
        """The Area of Interest polygon as derived from the AOI_extent"""
        bounds = self.AOI_extent()
        L, B, R, T = bounds
        return np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])

    def extents(self, by_part=False):
        """Extents are returned as L(eft), B(ottom), R(ight), T(op)
        """
        def _extent_(i):
            """Extent of a sub-array in an object array"""
            return np.concatenate((np.nanmin(i, axis=0), np.nanmax(i, axis=0)))
        # ----
        if self.N == 1:
            by_part = True
        p_ext = [_extent_(i) for i in self.split(by_part)]
        return np.asarray(p_ext)

    def extent_rectangles(self):
        """Return extent polygons for all shapes.  Points are ordered clockwise
         from the bottom left, with the first and last points the same.
         Requires an Advanced license in Pro

        See Also
        --------
        AOI_extent and AOI_poly
         """
        ext_polys = []
        for ext in self.extents():
            L, B, R, T = ext
            poly = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
            ext_polys.append(poly)
        return np.asarray(ext_polys)

    # ---- slicing, sampling equivalents
    def get(self, ID, asGeo=True):
        """Return the shape associated with the feature ID as an Geo array or
        an ndarray.

        Parameters
        ----------
        ID : integer
            A single integer value.
        asGeo : Boolean
            True, returns an updated Geo array.  False returns an ndarray or
            object array.
        """
        if not isinstance(ID, (int)):
            print("An integer ID is required, see ``pull`` for multiple values.")
            return None
        if asGeo:
            f_t = self.IFT[self.IDs == ID]
            s_e = f_t.ravel()
            return Geo(self[s_e[1]: s_e[-1]], IFT=f_t, Kind=self.K)
        return self.shapes[ID]

    def outer_rings(self, asGeo=False):
        """Collect the outer ring of a polygon shape.  Returns a list of
        ndarrays or optionally a new Geo array.
        """
        K = self.K
        if K != 2:
            print("Polygons required")
            return None
        a_2d = []
        id_too = []
        for ift in self.IFT:
            i, f, t = ift
            ar = self[f:t]
            p = _o_ring_(ar)           # ---- call ``_o_ring_``
            a_2d.append(np.asarray(p))
            id_too.append([i, len(p)])
        info = "{} outer_rings".format(str(self.Info))
        if asGeo:
            return _updateGeo(a_2d, K, id_too, info)  # ---- update Geo
        return a_2d

    def pull(self, ID_list, asGeo=True):
        """Pull multiple shapes, in the order provided.  The original IDs are
        kept but the point sequence is altered to reflect the new order

        Parameters
        ----------
        ID_list : array-like
            A list, tuple or ndarray of ID values identifying which features
            to pull from the input.
        asGeo : Boolean
            True, returns an updated Geo array.  False returns an ndarray or
            object array.
        """
        if not isinstance(ID_list, (list, tuple, np.ndarray)):
            print("An array/tuple/list of IDs are required, see ``get``")
            return None
        parts_ = np.vstack([self.IFT[self.IDs == i] for i in ID_list])
        vals = [np.asarray(self.XY[p[1]:p[2]]) for p in parts_]
        if not asGeo:
            return np.asarray(vals)
        I = parts_[:, 0]
        too = np.cumsum([len(i) for i in vals])
        frum = np.concatenate(([0], too))
        IFT = np.array(list(zip(I, frum, too)))
        vals = np.vstack(vals)
        return Geo(vals, IFT, self.K)

    def split(self, by_part=False):
        """Split points by shape or by parts for each shape.
        Use self.parts or self.shapes directly"""
        if by_part:
            return self.parts
        return self.shapes

    def point_info(self, by_part=True, with_null=False):
        """Point count by feature or parts of feature.

        Parameters
        ----------
        by part: boolean
            True for each feature part or False for the whole feature.
        with_null : boolean
            True, to include nan/null points.
        """
        chunks = self.split(by_part)
        if with_null:
            return self.FT[:, 1] - self.FT[:, 0]
        return np.array([len(i[~np.isnan(i[:, 0])]) for i in chunks])
    #
    # ---- Methods to determine angles, convexity and other properties that
    #      enable you use methods by part or by whole
    #
    def is_convex(self, by_part=True):
        """Return True for convex, False for concave.  Holes are excluded,
        multipart shapes are included by setting by_part=True
        """
        def _x_(a):
            """cross product"""
            dx, dy = a[0] - a[-1]
            if np.allclose(dx, dy):    # closed loop
                a = a[:-1]
            ba = a - np.roll(a, 1, 0)  # vector 1
            bc = a - np.roll(a, -1, 0) # vector 2
            return np.cross(ba, bc)
        # ----
        if self.K != 2:
            print("Polygons are required")
            return None
        chunks = self.split(by_part)
        check = []
        for p in chunks:
            p = _o_ring_(p)       # ---- run ``_o_ring_
            check.append(_x_(p))  # cross-product
        check = np.array([np.all(np.sign(i) >= 0) for i in check])
        return check

    def angles(self, inside=True, in_deg=True):
        """Sequential 3 point angles from a poly* shape.  The outer ring for
        each part is used.  see ``_angles_`` and ``_o_ring_``.
        """
        # ----
        angles = []
        for p in self.parts:  # -- run _angle_ and _o_ring_ on the chunks
            angles.append(_angles_(p, inside, in_deg))
        return angles
    #
    # ---- maxs, mins, means, pnts for all features
    #
    def maxs(self, by_part=False):
        """Maximums per feature"""
        return np.asarray([np.nanmax(i, axis=0) for i in self.split(by_part)])

    def mins(self, by_part=False):
        """Minimums per feature"""
        return np.asarray([np.nanmin(i, axis=0) for i in self.split(by_part)])

    def means(self, by_part=False, remove_dups=True):
        """Mean per feature, duplicates removed"""
        chunks = self.split(by_part)
        if remove_dups:
            chunks = [np.unique(i, axis=0) for i in chunks]
        return np.asarray([np.nanmean(i, axis=0) for i in chunks])
    #
    # ---- return altered geometry
    #
    def move(self, dx=0, dy=0):
        """see ``translate``"""
        return Geo(self.XY + [dx, dy], self.IFT)

    def translate(self, dx=0, dy=0):
        """Move/shift/translate by dx, dy to a new location"""
        return Geo(self.XY + [dx, dy], self.IFT)

    def rotate(self, about_center=True, angle=0.0, clockwise=False):
        """Rotate shapes about their center, if center=True,
        otherwise rotate about the X/Y axis origin (0, 0).
        """
        if clockwise:
            angle = -angle
        angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, s), (-s, c)))
        chunks = self.shapes
        out = []
        if about_center:
            uniqs = []
            for chunk in chunks:
                _, idx = np.unique(chunk, True, axis=0)
                uniqs.append(chunk[np.sort(idx)])
            cents = [np.nanmean(i, axis=0) for i in uniqs]
            for i, chunk in enumerate(chunks):
                ch = np.einsum('ij,jk->ik', chunk-cents[i], R) + cents[i]
                out.append(ch)
        else:
            cent = np.nanmean(self, axis=0)
            for chunk in chunks:
                ch = np.einsum('ij,jk->ik', chunk-cent, R) + cent
                out.append(ch)
        info = "{} rotate".format(self.Info)
        out = _updateGeo(np.vstack(out), self.K, self.IFT, Info=info)
        return out
    #
    # ---- changes to geometry, derived from geometry
    #  convex_hulls, minimum area bounding rectangle
    #  **see also** extent properties above
    #
    def convex_hulls(self, by_part=False):
        """Convex hull for shapes.

        by_part : boolean
            False for whole shape.  True for shape parts if present.
        """
        # ----
        if by_part:
            shps = self.parts
        else:
            shps = self.shapes
        ch_out = []
        for s in shps:  # ---- run convex hull, _ch_, on point groups
            h = _ch_(s)
            ch_out.append(h)
        return ch_out

    def min_area_rect(self):
        """Determines the minimum area rectangle for a shape represented
        by a list of points.  If the shape is a polygon, then only the outer
        ring is used.  This is the MABR... minimum area bounding rectangle.
       """
        def _extent_area_(a):
            """Area of an extent polygon"""
            LBRT = np.concatenate((np.nanmin(a, axis=0), np.nanmax(a, axis=0)))
            dx, dy = np.diff(LBRT.reshape(2, 2), axis=0).squeeze()
            return dx * dy, LBRT
        # ----
        o_rings = self.outer_rings(asGeo=True)
        ang_ = o_rings.angles(True, True)
        xt = o_rings.extents(True)  # centre of the extent shape and part
        cent_ = np.c_[np.mean(xt[:, :2], axis=1), np.mean(xt[:, 2:], axis=1)]
        rects = []
        for i, p in enumerate(o_rings.parts):
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
                    min_area = area_
                    area_old = area_
                    Xmin, Ymin, Xmax, Ymax = LBRT
                    vals = [min_area, Xmin, Ymin, Xmax, Ymax]
            rects.append(vals)
        return rects
    #
    #---- conversions --------------------------------------------------------
    #
    def fill_holes(self):
        """Fill holes in polygon shapes.  Returns a Geo class"""
        a_2d = []
        id_too = []
        K = self.K
        if K < 2:
            print("Polygon geometry required.")
            return None
        for i, p in enumerate(self.parts):
            nan_check = np.isnan(p[:, 0]) # check the Xs for nan
            if np.any(nan_check):         # split at first nan
                w = np.where(np.isnan(p[:, 0]))[0]
                p = np.split(p, w)[0]     # keep the outer ring
            a_2d.append(np.array(p))
            id_too.append([i, len(p)])
        info = "{} fill_holes".format(self.Info)
        new_geo = _updateGeo(a_2d, K, id_too, info)  # run update
        return new_geo

    def holes_to_shape(self):
        """Return holes in polygon shapes.  Returns a Geo class or None"""
        a_2d = []
        id_too = []
        K = self.K
        if K < 2:
            print("polygon geometry required")
            return None
        for i, p in enumerate(self.parts):  # work with the parts
            nan_check = np.isnan(p[:, 0])   # check the Xs for nan
            if np.any(nan_check):           # split at first nan
                w = np.where(np.isnan(p[:, 0]))[0]
                p_new = np.split(p, w)[1]     # keep the outer ring
                p_new = p_new[1:][::-1]
                a_2d.append(np.array(p_new))
                id_too.append([i, len(p_new)])
        if not a_2d:  # ---- if empty
            return None
        new_geo = _updateGeo(a_2d, K, id_too)    # run update
        return new_geo

    def multipart_to_singlepart(self, info=""):
        """Convert multipart shapes to singleparts and return a new Geo array.
        """
        ift = self.IFT
        data = np.vstack(self.parts)
        ift[:, 0] = np.arange(len(self.parts))
        new_geo = Geo(data, IFT=ift, Kind=self.K, Info=info)
        return new_geo

    def od_pairs(self):
        """Construct origin-destination pairs for traversing around the
        perimeter of polygons or along polylines or between sequences
        of points.

        Returns
        -------
        A 2D ndarray is returned of origin-distination pairs.
        """
        od = []
        for p in self.bits:
            od.append(np.c_[p[:-1], p[1:]])
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
        polylines = self.copy()
        polylines.K = 1
        return polylines

    def polys_to_points(self, keep_order=True):
        """Convert all feature vertices to an ndarray of points.
        NaN's are removed.  Optionally, retain point order"""
        uni, idx = np.unique(self, True, axis=0)
        if keep_order:
            uni = self[np.sort(idx)]
        uni = uni[~np.isnan(uni[:, 0])]
        return np.asarray(uni)

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
        outGeo = _updateGeo(polys, K=out_kind, id_too=None, Info=None)
        return outGeo

    def densify_by_distance(self, spacing=1):
        """Densify poly features by a specified distance.  Converts multipart
        to singlepart features during the process.
        Calls ``_pnts_on_line_`` for Geo bits
        """
        polys = []
        for a in self.bits:
            polys.append(_pnts_on_line_(a, spacing))
        out = _updateGeo(polys, K=self.K)
        return out

    def polys_to_segments(self, by_part=True):
        """Polyline or polygons boundaries segmented to individual lines.
        Shape is retained, but multivertice features are segmented

        Returns
        -------
        A Nx4 array or optionally a structured array for use in ArcGIS Pro

        Requires
        --------
        This performs the work.
        >>> _poly_segments_(a, as_2d=True, as_structured=False)
        """
        if by_part:
            shps = self.parts
        else:
            shps = self.shapes
        out = [_poly_segments_(s) for s in shps]
        return out

    #
    #----------------End of class definition-

# ===== Working with Geo ndarrays. ==========================================
#  Keep for now
#def _alterGeo_(vals, K):
#    """Alter Geo called by Geo.pull
#    """
#    id_too = []
#    for i, a in enumerate(vals):
#        id_too.append((i, len(a)))
#    arr = np.vstack(vals)
#    id_too = np.array(id_too)
#    I = id_too[:, 0]
#    too = np.cumsum(id_too[:, 1])
#    frum = np.concatenate(([0], too))
#    IFT = np.array(list(zip(I, frum, too)))
#    new_geo = Geo(arr, IFT, K)
#    return new_geo


def _o_ring_(arr):
    """Collect the outer ring of a shape.  An outer ring is separated from
    its inner ring, a hole, by a ``null_pnt``.  Each shape is examined for
    these and the outer ring is split off for each part of the shape.
    Called by::
        angles, outer_rings, is_convex and convex_hulls
    """
    nan_check = np.isnan(arr[:, 0])
    if np.any(nan_check):  # split at first nan to do outer
        w = np.where(np.isnan(arr[:, 0]))[0]
        arr = np.split(arr, w)[0]
    return arr

def _angles_(a, inside=True, in_deg=True):
    """Worker for Geo.angles. sequential points, a, b, c.

    Parameters
    ----------
    inside : boolean
        True, for interior angles.
    in_deg : boolean
        True for degrees, False for radians
    """
    #
    a = _o_ring_(a)             # work with the outer rings only
    dx, dy = a[0] - a[-1]
    if np.allclose(dx, dy):     # closed loop, remove duplicate
        a = a[:-1]
    ba = a - np.roll(a, 1, 0)   # just as fastish as concatenate
    bc = a - np.roll(a, -1, 0)  # but defitely cleaner
    cr = np.cross(ba, bc)
    dt = np.einsum('ij,ij->i', ba, bc)
    ang = np.arctan2(cr, dt)
    TwoPI = np.pi*2.
    if inside:
        ang = np.where(ang < 0, ang + TwoPI, ang)
    else:
        ang = np.where(ang > 0, TwoPI - ang, ang)
    if in_deg:
        angles = np.degrees(ang)
    return angles

def _ch_(points):
    """Calculates the convex hull for given points.
    """
    def _x_(o, a, b):
        """Cross-product for vectors o-a and o-b"""
        xo, yo = o
        xa, ya = a
        xb, yb = b
        return (xa - xo)*(yb - yo) - (ya - yo)*(xb - xo)
    # ----
    points = points[~np.isnan(points[:, 0])]
    _, idx = np.unique(points, return_index=True, axis=0)
    points = points[idx]
    if len(points) <= 3:
        return points
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and _x_(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _x_(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    ch = np.array(lower[:-1] + upper)[::-1]  # sort clockwise
    return ch


def _pnts_on_line_(a, spacing=1):  # densify by distance
    """Add points, at a fixed spacing, to an array representing a line.
    **See**  ``densify_by_distance`` for documentation.

    Parameters
    ----------
    a : array
        A sequence of `points`, x,y pairs, representing the bounds of a polygon
        or polyline object
    spacing : number
        Spacing between the points to be added to the line.
    """
    N = len(a) - 1                                    # segments
    dxdy = a[1:, :] - a[:-1, :]                       # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy)) # segment lengths
    steps = leng/spacing                              # step distance
    deltas = dxdy/(steps.reshape(-1, 1))              # coordinate steps
    pnts = np.empty((N,), dtype='O')                  # construct an `O` array
    for i in range(N):              # cycle through the segments and make
        num = np.arange(steps[i])   # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1, -1)        # add the final point and concatenate
    return np.concatenate((*pnts, a0), axis=0)


def _poly_segments_(a, as_2d=True, as_structured=False):
    """Segment poly* structures into o-d pairs from start to finish

    Parameters
    ----------
    a: array
        A 2D array of x,y coordinates representing polyline or polygons.
    as_2d : boolean
        Returns a 2D array of from-to point pairs, [xf, yf, xt, yt] if True.
        If False, they are returned as a 3D array in the form
        [[xf, yf], [xt, yt]]
    as_structures : boolean
        Optional structured/recarray output.  Field names are currently fixed.

    Notes
    -----
    Any row containing np.nan is removed since this would indicate that the
    shape contains the null_pnt separator.

   Use ``prn_tbl`` if you want to see a well formatted output.
    """
    s0, s1 = a.shape
    fr_to = np.zeros((s0-1, s1 * 2), dtype=a.dtype)
    fr_to[:, :2] = a[:-1]
    fr_to[:, 2:] = a[1:]
    fr_to = fr_to[~np.any(np.isnan(fr_to), axis=1)]
    if as_structured:
        dt = np.dtype([('X_orig', 'f8'), ('Y_orig', 'f8'),
                       ('X_dest', 'f8'), ('Y_dest', 'f8')])
        return uts(fr_to, dtype=dt)
    if not as_2d:
        s0, s1 = fr_to.shape
        return fr_to.reshape(s0, s1//2, s1//2)
    return fr_to


def _simplify_lines_(a, deviation=10):
    """Simplify array
    """
    ang = _angles_(a, inside=True, in_deg=True)
    idx = (np.abs(ang - 180.) >= deviation)
    sub = a[1: -1]
    p = sub[idx]
    return a, p, ang

# ===========================================================================
#  Keep???
def _nan_split_(arr):
    """Split at an array with nan values for an  ndarray."""
    s = np.isnan(arr[:, 0])                 # nan is used to split the 2D arr
    if np.any(s):
        w = np.where(s)[0]
        ss = np.split(arr, w)
        subs = [ss[0]]                      # collect the first full section
        subs.extend(i[1:] for i in ss[1:])  # slice off nan from the remaining
        return np.asarray(subs)
    return arr


# ===========================================================================
# ----print section  ------------------------------------
#
def prn_tbl(a, rows_m=25, names=None, deci=2, width=100):
    """Format a structured array with a mixed dtype.  Derived from
    arraytools.frmts and the prn_rec function therein.

    Parameters
    ----------
    a : array
        A structured/recarray
    rows_m : integer
        The maximum number of rows to print.  If rows_m=10, the top 5 and
        bottom 5 will be printed.
    names : list/tuple or None
        Column names to print, or all if None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters
    """
    def _ckw_(a, name, deci):
        """columns `a` kind and width"""
        c_kind = a.dtype.kind
        if (c_kind in FLOATS) and (deci != 0):  # float with decimals
            c_max, c_min = np.round([np.nanmin(a), np.nanmax(a)], deci)
            c_width = len(max(str(c_min), str(c_max), key=len))
        elif c_kind in NUMS:      # int, unsigned int, float wih no decimals
            c_width = len(max(str(np.nanmin(a)), str(np.nanmax(a)), key=len))
        elif c_kind in ('U', 'S', 's'):
            c_width = len(max(a, key=len))
        else:
            c_width = len(str(a))
        c_width = max(len(name), c_width) + deci
        return [c_kind, c_width]

    def _col_format(pairs, deci):
        """Assemble the column format"""
        form_width = []
        dts = []
        for c_kind, c_width in pairs:
            if c_kind in INTS:  # ---- integer type
                c_format = ':>{}.0f'.format(c_width)
            elif c_kind in FLOATS: # and np.isscalar(c[0]):  # float rounded
                c_format = ':>{}.{}f'.format(c_width, deci)
            else:
                c_format = "!s:<{}".format(c_width)
            dts.append(c_format)
            form_width.append(c_width)
        return dts, form_width
    # ----
    dtype_names = a.dtype.names
    if dtype_names is None:
        print("Structured/recarray required")
        return None
    if names is None:
        names = dtype_names
    # ---- slice off excess rows, stack upper and lower slice using rows_m
    if a.shape[0] > rows_m*2:
        a = np.hstack((a[:rows_m], a[-rows_m:]))
    # ---- get the column formats from ... _ckw_ and _col_format ----
    pairs = [_ckw_(a[name], name, deci) for name in names]  # -- column info
    dts, wdths = _col_format(pairs, deci)                   # format column
    # ---- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    a = a[list(names[:N])]
    # ---- Assemble the formats and print
    tail = ['', ' ...'][N < len(names)]
    row_frmt = "  ".join([('{' + i + '}') for i in dts[:N]])
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = "  ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " ... " + hdr2.format(*names[:N]) + tail
    header = "\n{}\n{}".format(header, "-"*len(header))
    txt = [header]
    for idx, i in enumerate(range(a.shape[0])):
        txt.append(" {:>03.0f} ".format(idx) + row_frmt.format(*a[i]) + tail)
    msg = "\n".join([i for i in txt])
    print(msg)
    # return row_frmt, hdr2  # uncomment for testing

def prn_geo(a, rows_m=100, names=None, deci=2, width=100):
    """Format a structured array with a mixed dtype.  Derived from
    arraytools.frmts and the prn_rec function therein.

    Parameters
    ----------
    a : array
        A structured/recarray
    rows_m : integer
        The maximum number of rows to print.  If rows_m=10, the top 5 and
        bottom 5 will be printed.
    names : list/tuple or None
        Column names to print, or all if None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters

    Notes
    -----
    >>> toos = s0.IFT[:,2]
    >>> nans = np.where(np.isnan(s0[:,0]))[0]  # array([10, 21, 31, 41]...
    >>> dn = np.digitize(nans, too)            # array([1, 2, 3, 4]...
    >>> ift[:, 0][dn]                          # array([1, 1, 2, 2])
    >>> np.sort(np.concatenate((too, nans)))
    ... array([ 5, 10, 16, 21, 26, 31, 36, 41, 48, 57, 65], dtype=int64)
    """
    def _ckw_(a, name, deci):
        """columns `a` kind and width"""
        c_kind = a.dtype.kind
        if (c_kind in FLOATS) and (deci != 0):  # float with decimals
            c_max, c_min = np.round([np.nanmin(a), np.nanmax(a)], deci)
            c_width = len(max(str(c_min), str(c_max), key=len))
        elif c_kind in NUMS:      # int, unsigned int, float wih no decimals
            c_width = len(max(str(np.nanmin(a)), str(np.nanmax(a)), key=len))
        else:
            c_width = len(name)
        c_width = max(len(name), c_width) + deci
        return [c_kind, c_width]

    def _col_format(pairs, deci):
        """Assemble the column format"""
        form_width = []
        dts = []
        for c_kind, c_width in pairs:
            if c_kind in INTS:  # ---- integer type
                c_format = ':>{}.0f'.format(c_width)
            elif c_kind in FLOATS: # and np.isscalar(c[0]):  # float rounded
                c_format = ':>{}.{}f'.format(c_width, deci[-1])
            else:
                c_format = "!s:^{}".format(c_width)
            dts.append(c_format)
            form_width.append(c_width)
        return dts, form_width
    # ----
    if names is None:
        names = ['shape', 'part', 'X', 'Y']
    # ---- slice off excess rows, stack upper and lower slice using rows_m
    if not hasattr(a, 'IFT'):
        print("Requires a Geo array")
        return None
    ift = a.IFT
    c = [np.repeat(ift[i, 0], ift[i, 2] - ift[i, 1])
         for i, p in enumerate(ift[:, 0])]
    c = np.concatenate(c)
    # ---- p: __ shape end, p0: x parts, p1: o start of parts, pp: concatenate
    p = np.where(np.diff(c, append=0) == 1, "___", "")
    p0 = np.where(np.isnan(a[:, 0]), "x", "")
    p1 = np.asarray(["" if i not in ift[:, 2] else 'o' for i in range(len(p))])
    pp = np.asarray([p[i]+p0[i]+p1[i] for i in range(len(p))])
    if a.shape[0] > rows_m:
        a = a[:rows_m]
        c = c[:rows_m]
        p = p[:rows_m]
    # ---- get the column formats from ... _ckw_ and _col_format ----
    deci = [0, 0, deci, deci]
    flds = [c, pp, a[:, 0], a[:, 1]]
    pairs = [_ckw_(flds[n], names[n], deci[n]) for n, name in enumerate(names)]  # -- column info
    dts, wdths = _col_format(pairs, deci)                   # format column
    # ---- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    # ---- Assemble the formats and print
    row_frmt = " {:>03.0f} " + "  ".join([('{' + i + '}') for i in dts[:N]])
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = "  ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " pnt " + hdr2.format(*names[:N])
    header = "\n{}\n{}".format(header, "-"*len(header))
    txt = [header]
    for i in range(a.shape[0]):
        txt.append(row_frmt.format(i, c[i], pp[i], a[i, 0], a[i, 1]))
    msg = "\n".join([i for i in txt])
    print(msg)
    # return row_frmt, hdr2  # uncomment for testing

# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    #msg = _demo_()
# in_fc3 = r"C:\Arc_projects\Canada\Canada.gdb\Ontario_LCConic"

'''
in_fc2 = r"C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2"
SR = getSR(in_fc2)
shapes = fc_shapes(in_fc2)

tmp, IFT = fc_geometry(in_fc2)
m = [300000., 5000000.]

a = tmp  - m
kind = 2
info = None
g = Geo(a, IFT, kind, info)

'''
