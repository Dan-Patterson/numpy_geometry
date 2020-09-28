# -*- coding: utf-8 -*-
"""
--------------------------------------------
  npGeo: Geo class, properties and methods
--------------------------------------------

The Geo class is a subclass of numpy's ndarray.  Properties that are related
to geometry have been assigned and methods developed to return geometry
properties.

----

Author :
    Dan Patterson

Modified :
    2020-09-24

Script : npGeo.py
    A geometry class and methods based on numpy.

Notes
-----
Alternate slicing used for `od_pairs`, `segment_angles`, `rotate`
When appropriate, you can slice using from-to pairs, just subtract 1 from the
`to` value to remove values that traverse the last point in one feature with
the first point in a subsequent pair

>>> current = [ ...do something with p... for p in self.bits]
>>> z = ...do something with self...
>>> slice [z[ft[0]:ft[1] - 1] for ft in self.FT]
"""
# pycodestyle D205 gets rid of that one blank line thing
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0402,E0633,E0611,E1136,E1121,R0904,R0914,
# pylint: disable=W0201,W0212,W0221,W0612,W0621,W0105
# pylint: disable=R0902,R0912,R0913,R0915


import sys
from textwrap import indent, dedent, wrap
import numpy as np
# from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import repack_fields

# import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import npg_geom as geom
import npg_helpers
from npg_helpers import (
    _angles_3pnt_, _area_centroid_, _bit_area_, _bit_crossproduct_,
    _bit_extent_, _bit_length_, _rot_, _rotate_, polyline_angles)
import npg_io
import npg_min_circ as sc


from npgDocs import Geo_hlp

# noqa: E501
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.2f}'.format}
np.set_printoptions(edgeitems=10, linewidth=120, precision=2, suppress=True,
                    threshold=200, formatter=ft)

script = sys.argv[0]  # print this should you need to locate the script

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS
TwoPI = np.pi * 2.0

__all__ = [
    'Geo', 'is_Geo', 'roll_coords', 'check_geometry',
    'array_IFT', 'arrays_to_Geo', 'Geo_to_arrays', 'Geo_to_lists',
    '_fill_float_array', 'dirr', 'geo_info', '_svg'
]


# ----------------------------------------------------------------------------
# ---- (1) ... Geo class, properties and methods ... -------------------------
#
class Geo(np.ndarray):
    """Geo class.  Based on the ndarray."""

    __name__ = "npGeo"
    __module__ = "npgeom"
    __author__ = "Dan Patterson"
    __doc__ += Geo_hlp

    def __new__(cls,
                arr=None,
                IFT=None,
                Kind=2,
                Extent=None,
                Info="Geo array",
                SR=None
                ):
        """See ``npgDocs`` for construction notes."""
        arr = np.ascontiguousarray(arr)
        IFT = np.ascontiguousarray(IFT)
        if (arr.ndim != 2) or (IFT.ndim != 2):
            m = "Input error... arr.ndim != 2 : {} or IFT.dim != 2 : {}"
            print(dedent(m).format(arr.ndim, IFT.ndim))
            return None
        if (IFT.shape[-1] < 6) or (Kind not in (0, 1, 2)):
            print(dedent(Geo_hlp))
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
        self.N = len(uni)         # sample size, unique shapes
        self.U = uni              # unique ids self.IDs[idx]
        self.SR = SR
        self.XT = Extent
        if self.XT is not None:
            self.LL, self.UR = self.XT  # extent of featureclass
        else:
            self.LL = np.array([0., 0.])
            self.UR = None
        if self.shape[1] >= 2:    # X,Y and XY initialize
            self.X = arr[:, 0]
            self.Y = arr[:, 1]
            self.XY = arr[:, :2]
        if self.shape[1] >= 3:    # add Z, although not implemented
            self.Z = arr[:, 2]    # but kept for future additions
        else:
            self.Z = None
        self.hlp = "use obj.H"
        self.SVG = ""
        return self

    def __array_finalize__(self, src_arr):
        """Finalize new object....

        This is where housecleaning takes place for explicit, view casting or
        new from template... ``src_arr`` is either None, any subclass of
        ndarray including our own (words from documentation) OR another
        instance of our own array.
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
        self.SVG = getattr(src_arr, 'SVG', None)

    def __array_wrap__(self, out_arr, context=None):
        """Wrap it up."""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # ------------------------------------------------------------------------
    # ---- End of class definition -------------------------------------------
    #
    # ---- help information
    @property
    def H(self):
        """Print the parameter documentation for an instance of the Geo class.

        If you want to save this as a string, use::
            >>> doc_string = g.__docs__  # note the ``s`` in doc``s``
        """
        print(Geo_hlp)

    @property
    def info(self):
        """Convert an IFT array to full information.

        Only the first 50 records maximum will be printed. To see the data
        structure, and/or more records use the `prn_geo` method.
        """
        info_ = self.IFT_str[:50]
        frmt = "-" * 14 + "\nExtents :\n  LL {}\n  UR {}" + \
            "\nShapes :{:>6.0f}\nParts  :{:>6.0f}" + \
            "\nPoints :{:>6.0f}\n"
        args = [str(self.LL), str(self.UR), len(self.U), info_.shape[0],
                info_['To_pnt'][-1]]
        print(dedent(frmt).format(*args))
        npg_io.prn_tbl(info_)

    # ---- IFT : shape, part, bit
    # see also self.U, self.PID
    @property
    def shp_IFT(self):
        """Shape IFT values.  ``shp_IFT == part_IFT`` for singlepart shapes."""
        if np.sum(self.is_multipart()[:, 1]) == 0:  # no multiparts check
            return self.part_IFT
        df = self.To - self.Fr
        cnt = np.bincount(self.IDs, df)
        too = np.cumsum(cnt, axis=0, dtype=np.int32)[1:]
        fr = np.concatenate(([0], too[:-1]), axis=0)
        ift = np.full((len(fr), 6), -1, dtype=np.int32)
        ift[:, :3] = np.asarray([self.U, fr, too]).T
        return ift

    @property
    def part_IFT(self):
        """Part IFT values."""
        idx = np.concatenate((self.Fr[self.Bit == 0], [len(self)]))
        fr_to = np.array([idx[:-1], idx[1:]]).T
        w = np.nonzero(self.Bit == 0)[0]
        ifts = self.IFT[w]     # slice by bit sequence
        ifts[:, 1:3] = fr_to   # substitute in the new from-to values
        return ifts

    @property
    def bit_IFT(self):
        """Bit IFT values."""
        return self.IFT

    @property
    def IFT_str(self):
        """Geo array structure.  See self.structure for more information."""
        nmes = ["OID_", "Fr_pnt", "To_pnt", "CW_CCW", "Part_ID", "Bit_ID"]
        return uts(self.IFT, names=nmes, align=False)

    # ---- ids : shape, part, bit
    @property
    def shp_ids(self):
        """Shape ID values. Note: they may not be sequential or continuous."""
        return self.U

    @property
    def part_ids(self):
        """Return the ID values of the shape parts.  See, shp_ids."""
        return self.part_IFT[:, 0]

    @property
    def bit_ids(self):
        """Return the ID values for each bit in a shape.
        Ids are repeated for each part or ring in a shape. See, shp_ids.
        """
        return self.IDs

    @property
    def bit_seq(self):
        """Return the bit sequence for each bit in a shape.
        The sequence is numbered from zero to ``n``.  A shape can consist of a
        single bit or multiple bits consisting of outer rings and holes.
        """
        return self.Bit

    @property
    def pnt_ids(self):
        """Feature id that each point belongs to.
        Useful for slicing the points of poly features.  See, shp_ids warning.
        """
        return np.repeat(self.IDs, self.To - self.Fr)

    @property
    def shp_pnt_ids(self):
        """Return the sequential and feature ids for each point."""
        z1 = np.repeat(self.IDs, self.To - self.Fr)
        z0 = np.arange(len(z1) + 1)
        return np.concatenate((z0[:-1, None], z1[:, None]), axis=1)

    @property
    def xy_id(self):
        """Return a structured array of numbered points.
        The points are shifted back to their original bounding box.
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

    # ---- counts : shape, part, bit
    @property
    def shp_pnt_cnt(self):
        """Points in each shape.  Columns: shape, points."""
        df = self.To - self.Fr
        cnt = np.bincount(self.IDs, df)[1:]
        out = np.zeros((self.N, 2), dtype=np.int32)
        out[:, 0] = self.U
        out[:, 1] = cnt
        return out

    @property
    def shp_part_cnt(self):
        """Part count for shapes. Returns IDs and count array."""
        uni, cnts = np.unique(self.part_ids, return_counts=True)
        return np.concatenate((uni[:, None], cnts[:, None]), axis=1)

    @property
    def bit_pnt_cnt(self):
        """Point count by bit. Columns: shape_id, bit sequence, point count."""
        b_ids = self.bit_ids
        b_seq = self.bit_seq  # self.IFT[5]
        return np.array([(b_ids[i], b_seq[i], len(p))
                         for i, p in enumerate(self.bits)])

    # ---- coordinates: shape, parts, bits.  Returns copies, not views.
    # old check for conversion to `O` dtype
    # sze = [len(i) for i in tmp]
    # dt = 'float64'
    # if max(sze) != min(sze):
    #     dt = 'O'
    # return np.asarray(tmp, dtype=dt)

    @property
    def shapes(self):
        """Deconstruct the array keeping all shape parts together."""
        df = self.To - self.Fr
        cnt = np.bincount(self.IDs, df)
        too = np.cumsum(cnt, axis=0, dtype=np.int32)[1:]
        fr = np.concatenate(([0], too[:-1]), axis=0)
        fr_to = np.concatenate((fr[:, None], too[:, None]), axis=1)
        return [self.XY[ft[0]:ft[1]] for ft in fr_to]

    @property
    def parts(self):
        """Deconstruct the 2D array into its parts."""
        fr_to = self.part_IFT[:, 1:3]
        return [self.XY[ft[0]:ft[1]] for ft in fr_to]

    @property
    def bits(self):
        """Deconstruct the 2D array returning all rings."""
        return [self.XY[ft[0]:ft[1]] for ft in self.FT]

    # ------------------------------------------------------------------------
    # ---- methods and derived properties section
    # ---- (1) slicing, sampling equivalents
    #
    # Common Parameters
    #
    # splitter : b, p, s
    #     Split by (b)it, (p)art (s)hape.
    #
    def outer_rings(self, asGeo=True):
        """Get the first bit of multipart shapes and/or shapes with holes."""
        if np.any(self.CW == 0):
            ift_s = self.IFT[self.Bit == 0]
            a_2d = [self.XY[ft[0]:ft[1]] for ft in self.FT]
            if asGeo:
                a_2d = np.concatenate(a_2d, axis=0)
                ft = np.concatenate(([0], ift_s[:, 2] - ift_s[:, 1]))
                c = np.cumsum(ft)
                ift_s[:, 1] = c[:-1]
                ift_s[:, 2] = c[1:]
                return Geo(a_2d, ift_s, self.K, self.XT, Info="outer rings")
            else:
                return a_2d
        if asGeo:
            return self
        return self.bits

    def inner_rings(self, asGeo=False, to_clockwise=False):
        """Collect the holes of a polygon shape."""
        info = "{} inner rings".format(str(self.Info))
        ift_s = self.IFT[self.Bit != 0]
        if ift_s.size == 0:
            print("\nNo inner rings\n")
            return
        if to_clockwise:
            a_2d = [self.XY[ft[1]:ft[2]][::-1] for ft in ift_s]
        else:
            a_2d = [self.XY[ft[1]:ft[2]] for ft in ift_s]
        if asGeo:
            a_2d = np.concatenate(a_2d, axis=0)
            ft = np.concatenate(([0], ift_s[:, 2] - ift_s[:, 1]))
            c = np.cumsum(ft)
            ift_s[:, 1] = c[:-1]
            ift_s[:, 2] = c[1:]
            if to_clockwise:
                ift_s[:, 3] = 1
            z = np.asarray([np.concatenate(
                                    (np.min(pnts, axis=0),
                                     np.max(pnts, axis=0))) for pnts in a_2d])
            xtent = np.concatenate(
                (np.min(z[:, :2], axis=0), np.max(z[:, 2:], axis=0)))
            return Geo(a_2d, ift_s, self.K, xtent, info)
        return np.asarray(a_2d, dtype='O')

    def first_bit(self, asGeo=False):
        """Implement `outer_rings`."""
        if self.K != 2:
            print("Polygons required...")
            return None
        if asGeo:
            return self.outer_rings(asGeo=True)
        return self.outer_rings(asGeo=False)

    def first_part(self, asGeo=True):
        """Return the first part of a multipart shape or a shape with holes.
        Holes are retained.  The IFT is altered to account for point removal.
        """
        info = "{} first part".format(str(self.Info))
        ift_s = self.IFT[self.PID == 1]
        a_2d = [self.XY[ft[0]:ft[1]] for ft in self.FT]
        if asGeo:
            a_2d = np.concatenate(a_2d, axis=0)
            ft = np.concatenate(([0], ift_s[:, 2] - ift_s[:, 1]))
            c = np.cumsum(ft)
            ift_s[:, 1] = c[:-1]
            ift_s[:, 2] = c[1:]
            return Geo(a_2d, ift_s, self.K, self.XT, info)
        return a_2d

    def get_shapes(self, ids=None, asGeo=True):
        """Pull multiple shapes, in the order provided."""
        if isinstance(ids, (int)):
            ids = [ids]
        ids = np.asarray(ids)
        if (ids.ndim and ids.size) == 0:
            print("An array/tuple/list of IDs are required.")
            return None
        if not np.all([a in self.IDs for a in ids]):
            print("Not all required IDs are in the list provided")
            return None
        xys = []
        for id_num in ids:
            case = self.FT[self.IDs == id_num]
            for c in case:
                xys.append(self.XY[c[0]: c[-1]])
        if asGeo:
            info = "Old_order" + (" {}"*len(ids)).format(*ids)
            return arrays_to_Geo(xys, kind=self.K, info=info)
        return xys

    def split_by(self, splitter="bit"):
        """Split points by shape or by parts for each shape.

        **keep for now**
        Use self.bits, self.parts or self.shapes directly.
        """
        case = splitter[0].lower()  # use first letter for case
        if case == "b":
            vals = self.bits
        elif case == "p":
            vals = self.parts
        elif case == "s":
            vals = self.shapes
        else:
            print("\nSplitter not in by (b)it, (p)art (s)hape")
            vals = None
        return vals

    def original_arrays(self):
        """Convert the Geo arrays back to the original input arrays."""
        g0 = self.shift(self.LL[0], self.LL[1])
        b = g0.bits
        w = np.where(self.CW == 1)[0]
        arrs = np.split(b, w)[1:]
        return arrs

    # ---- (2) areas, centrality, lengths/perimeter for polylines/polygons
    #
    def areas(self, by_shape=True):
        """Area for the sub arrays using einsum based area calculations.

        Uses `_bit_area_` to calculate the area.
        The ``by_shape=True`` parameter returns the area for each shape. If
        False, each bit area is returned.  Negative areas are holes.
        """
        if self.K != 2:
            print("Polygons required.")
            return None
        bit_totals = [_bit_area_(i) for i in self.bits]  # by bit
        if by_shape:
            b_ids = self.bit_ids
            return np.bincount(b_ids, weights=bit_totals)[self.U]
        return bit_totals

    def lengths(self, by_shape=True):
        """Polyline lengths or polygon perimeter. Uses bit_length"""
        if self.K not in (1, 2):
            print("Polyline/polygon representation is required.")
            return None
        bit_lengs = [np.sum(_bit_length_(i)) for i in self.bits]
        if by_shape:
            b_ids = self.bit_ids
            return np.bincount(b_ids, weights=bit_lengs)[self.U]
        return bit_lengs

    def centers(self, by_shape=True):
        """Return the center of a shape's outer ring points."""
        if by_shape:
            shps = self.shapes
            if self.K == 2:
                return np.stack([np.mean(s[:-1], axis=0) for s in shps])
            return np.stack([np.mean(s, axis=0) for s in shps])
        # ---- part centers
        o_rings = self.outer_rings(False)
        # Remove duplicate start-end for polygons. Use all points otherwise.
        if self.K == 2:
            return np.stack([np.mean(r[:-1], axis=0) for r in o_rings])
        return np.stack([np.mean(r, axis=0) for r in o_rings])

    def centroids(self):
        """Return the centroid of the polygons using `_area_centroid_`."""
        def weighted(x_y, Ids, areas):
            """Weight coordinate by area, x_y is either the x or y."""
            w = x_y * areas                   # area weighted x or y
            w1 = np.bincount(Ids, w)[1:]      # [Ids] weight / bin size
            ar = np.bincount(Ids, areas)[1:]  # [I]  # areas per bin
            return w1/ar
        # ----
        if self.K != 2:
            print("Polygons required.")
            return None
        centr = []
        areas = []
        ids = self.part_ids  # unique shape ID values
        for ID in self.U:
            parts_ = self.part_IFT[ids == ID]
            out = np.asarray([np.asarray(self.XY[p[1]:p[2]]) for p in parts_])
            for prt in out:
                area, cen = _area_centroid_(prt)  # ---- determine both
                centr.append(cen)
                areas.append(area)
        centr = np.asarray(centr)
        areas = np.asarray(areas)
        xs = weighted(centr[:, 0], ids, areas)
        ys = weighted(centr[:, 1], ids, areas)
        return np.asarray(list(zip(xs, ys)))

    # ---- (3) extents and extent shapes
    #
    def aoi_extent(self):
        """Return geographic extent of the `aoi` (area of interest)."""
        return np.concatenate((np.min(self.XY, axis=0),
                               np.max(self.XY, axis=0)))

    def aoi_rectangle(self):
        """Derive polygon bounds from the aoi_extent."""
        L, B, R, T = self.aoi_extent()
        return np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])

    def extents(self, splitter="part"):
        """Return extent as L(eft), B(ottom), R(ight), T(op)."""
        if self.N == 1:
            splitter = "bit"
        chunks = self.split_by(splitter)
        return np.asarray([_bit_extent_(c) for c in chunks])

    def extent_centers(self, splitter="shape"):
        """Return extent centers."""
        ext = self.extents(splitter)
        xs = (ext[:, 0] + ext[:, 2])/2.
        ys = (ext[:, 1] + ext[:, 3])/2.
        return np.concatenate((xs[:, None], ys[:, None]), axis=1)

    def extent_corner(self, corner='LB'):
        """Return extent centers. Use `LB` or `LT` for left bottom or top."""
        if corner not in ('LB', 'LT'):
            print("Only Left-bottom (LB) and Left-top (LT) are implemented")
            return
        ext = self.extents("shape")
        xs = ext[:, 0]
        if corner == 'LB':
            ys = ext[:, 1]
        else:
            ys = ext[:, 3]
        return np.concatenate((xs[:, None], ys[:, None]), axis=1)

    def extent_pnts(self, splitter="shape", shift_back=False, asGeo=False):
        """Derive the LB and RT point for a shape geometry."""
        ext_polys = []
        for ext in self.extents(splitter):
            L, B, R, T = ext
            poly = np.array([[L, B], [R, T]])
            if shift_back:
                poly = poly + self.LL
            ext_polys.append(poly)
        if asGeo:
            ext_polys = arrays_to_Geo(ext_polys, kind=2, info="extent pnts")
        return ext_polys

    def extent_rectangles(
            self, splitter='shape', shift_back=False, asGeo=False):
        """Return extent polygons for the whole shape or the shape by bit.

        Points are ordered clockwise from the bottom left, with the first and
        last points the same.  Requires an Advanced license in Pro for
        equivalent functionality.

        See Also
        --------
        `aoi_extent` and `aoi_rectangles`
        """
        ext_polys = []
        for ext in self.extents(splitter):
            L, B, R, T = ext
            poly = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
            if shift_back:
                poly = poly + self.LL
            ext_polys.append(poly)
        if asGeo:
            ext_polys = arrays_to_Geo(ext_polys, kind=2, info="extent polys")
        return ext_polys

    def boundary(self):
        """Alias for polygons_to_polylines."""
        if self.K == 2:
            return self.polygons_to_polylines()
        return self

    # ---- (4) maxs, mins, means for all features
    #
    # useful # b_id = self.IDs[self.Bit == 0]
    def maxs(self, by_bit=False):
        """Maximums per feature or part."""
        if len(self.shp_part_cnt) == 1:
            return np.asarray(np.max(self.XY, axis=0))
        if by_bit:
            return np.asarray([np.max(i, axis=0) for i in self.bits])
        return np.asarray([np.max(i, axis=0) for i in self.shapes])

    def mins(self, by_bit=False):
        """Minimums per feature or part."""
        if len(self.shp_part_cnt) == 1:
            return np.asarray(np.min(self.XY, axis=0))
        if by_bit:
            return [np.min(i, axis=0) for i in self.bits]
        return np.asarray([np.min(i, axis=0) for i in self.shapes])

    def means(self, by_bit=False, remove_dups=True):
        """Mean per feature or part, optionally keep duplicates."""
        if len(self.shp_part_cnt) == 1:
            chunks = [self]
        if by_bit:
            chunks = self.bits
        if remove_dups:
            chunks = [np.unique(i, axis=0) for i in chunks]
        return np.asarray([np.mean(i, axis=0) for i in chunks])

    # ------------------------------------------------------------------------
    # ---- npg_geom methods/properties required
    #
    # ---- (1) **is** section, condition/case checking, kept to a minimum
    def is_clockwise(self, is_closed_polyline=False):
        """Utilize the `shoelace` area calculation to determine orientation.

        If the geometry represent a closed-loop polyline, then set the
        `is_closed_polyline` to True.  Geometry is not checked.
        """
        msg = "Polygons or closed-loop polylines are required."
        if self.K not in (1, 2):
            print(msg)
            return None
        if (self.K == 1) and (not is_closed_polyline):
            print(msg)
            return None
        out = np.concatenate((self.bit_ids[:, None], self.CW[:, None]), axis=1)
        return uts(np.asarray(out), names=['IDs', 'Clockwise'])

    def is_convex(self):
        """Return True (convex), False (concave). Uses `_bit_crossproduct_`.

        Holes are excluded.  The first part of a multipart shapes is used.
        Duplicate start-end points removed prior to cross product.
        The cross product does it for the whole shape all at once.
        """
        if self.K != 2:
            print("Polygons are required.")
            return None
        f_bits = self.first_bit(False)
        check = [_bit_crossproduct_(p[:-1]) for p in f_bits]  # cross product
        return np.array([np.all(np.sign(i) >= 0) for i in check])

    def is_multipart(self, as_structured=False):
        """For each shape, returns whether it has multiple parts.

        An ndarray is returned with the first column being the shape number
        and the second is coded as 1 for True and 0 for False.

        >>> np.array(list(zip(self.U, w)))  # old.... and ids = self.U
        """
        partcnt = self.shp_part_cnt
        w = np.where(partcnt[:, 1] > 1, 1, 0)
        arr = np.concatenate((self.U[:, None], w[:, None]), axis=1)
        if as_structured:
            dt = np.dtype([('IDs', '<i4'), ('Parts', '<i4')])
            return uts(arr, dtype=dt)
        return arr

    def is_in(self, pnts, reverse_check=False, values=True, indices=True):
        """Return pnts that are in the Geo array.

        Parameters
        ----------
        pnts : Nx2 array
            The array representing the point to query.
        reverse_check : boolean
            True queries the pnts array for common points in the Geo array.
        values, indices : boolean
            True, returns the values in common and their indices.  False for
            either omits that particular return.
        """
        if reverse_check:
            lead = pnts
            check = (pnts[:, None] == self.XY).all(-1).any(-1)
        else:
            lead = self
            check = (self.XY[:, None] == pnts).all(-1).any(-1)
        if indices and values:
            return lead[check], np.nonzero(check)[0]
        if indices:
            return np.nonzero(check)[0]
        return lead[check]

    # ---- (2) angles
    #
    def segment_angles(self, fromNorth=False):
        """Segment angles for all bits of a Geo array.  Uses alternate slicing
        described in `Notes` in npGeo.
        """
        xy = self.XY
        dxy = xy[1:] - xy[:-1]
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        if fromNorth:
            ang = np.mod((450.0 - ang), 360.)
        return [ang[ft[0]:ft[1] - 1] for ft in self.FT]

    def polyline_angles(self, right_side=True, fromNorth=False):
        """Polyline/segment angles.  *** needs work***."""
        f_bits = self.first_bit(False)
        inside = False
        if right_side:
            inside = True
        return [_angles_3pnt_(p, inside, True) for p in f_bits]  # npg_helper

    def polygon_angles(self, inside=True, in_deg=True):
        """Sequential 3 point angles for a poly* outer rings for each shape."""
        f_bits = self.first_bit(False)
        return [_angles_3pnt_(p, inside, in_deg) for p in f_bits]  # npg_helper

    # ---- (3) alter geometry
    #
    def moveto(self, x=0, y=0):
        """Shift/translate the dataset origin is the lower-left corner."""
        dx, dy = x, y
        if dx == 0 and dy == 0:
            dx, dy = np.min(self.XY, axis=0)
        return Geo(self.XY + [-dx, -dy], self.IFT, self.K, self.XT)

    def shift(self, dx=0, dy=0):
        """See `translate`."""
        return Geo(self.XY + [dx, dy], self.IFT, self.K, self.XT)

    def translate(self, dx=0, dy=0):
        """Move/shift/translate by dx, dy to a new location."""
        return Geo(self.XY + [dx, dy], self.IFT, self.K, self.XT)

    def rotate(self, as_group=True, angle=0.0, clockwise=False):
        """Rotate shapes about the group center or individually.
        Rotation is done by `_rotate_`, a new geo array is returned.
        """
        if clockwise:
            angle = -angle
        angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, s), (-s, c)))
        out = _rotate_(self, R, as_group)  # requires a Geo array, self
        info = "{} rotated".format(self.Info)
        if not as_group:
            out = np.vstack(out)
        return Geo(out, self.IFT, self.K, self.XT, info)

    # ---- (4) bounding circle, convex_hulls, mabr, Delaunay
    # **see also** extent properties above
    #
    def bounding_circles(self, angle=5, shift_back=False, return_xyr=False):
        """Bounding circles for features.

        Parameters
        ----------
        angle : number
            Angles to form n-gon.  A value of 10 will yield 36 point circle.
        shift_back : boolean
            Return to original coordinates if applicable.
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
        if shift_back:
            circs = [circ + self.LL for circ in circs]
        circs = arrays_to_Geo(circs, kind=2, info="circs")
        if return_xyr:
            return xyr, circs
        return circs

    def convex_hulls(self, by_bit=False, shift_back=False, threshold=50):
        """Convex hull for shapes.  Calls `_ch_` to control the method used.

        Parameters
        ----------
        by_part : boolean
            False for whole shape.  True for shape parts if present.
        threshold : integer
            Points... less than threshold uses simple CH method, greater than,
            uses scipy.
        """
        # ----
        shps = self.first_bit(asGeo=False) if by_bit else self.shapes
        # run convex hull, _ch_, on point groups
        ch_out = [geom._ch_(s, threshold) for s in shps]
        for i, c in enumerate(ch_out):  # check for closed
            if np.all(c[0] != c[-1]):
                ch_out[i] = np.vstack((c, c[0]))
        if shift_back:
            ch_out = [i + self.LL for i in ch_out]
        out = arrays_to_Geo(ch_out, kind=2, info="convex hulls")
        return out

    def min_area_rect(self, shift_back=False, as_structured=False):
        """Determine the minimum area bounding rectangle for polygons.

        Requires
        --------
        - `mabr` from npg_geom
        - `_area_centroid_` from npg_helpers

        >>> p = np.array([[1., 0], [0., 1], [9.,10], [10, 9], [1., 0.]])
        """
        def _r_(a, cent, angle, clockwise):
            """Rotate by `angle` in degrees, about the center."""
            angle = np.radians(angle)
            if clockwise:
                angle = -angle
            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c, -s), (s, c)))
            return np.einsum('ij,jk->ik', a - cent, R) + cent

        chs = self.convex_hulls(True, False, 50)
        polys = chs.bits
        p_centers = [_area_centroid_(i)[1] for i in polys]
        # p_centers = chs.extent_centers()  # -- try extent centers
        # p_angles = _bit_segment_angles_(polys, fromNorth=False)
        p_angles = polyline_angles(polys, fromNorth=False)
        rects = geom.mabr(polys, p_centers, p_angles)
        if as_structured:
            dt = np.dtype([('Rect_area', '<f8'), ('Angle_', '<f8'),
                           ('Xmin', '<f8'), ('Ymin', '<f8'),
                           ('Xmax', '<f8'), ('Ymax', '<f8')])
            return uts(rects, dtype=dt)
        # ---- return extent polygons
        mabrs = []
        LL = self.XT[0]
        for i, rect in enumerate(rects):
            c = rect[1]
            ang = rect[2]
            L, B, R, T = rect[3:]
            tmp = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
            tmp = _rot_(tmp, cent=c, angle=ang, clockwise=True)
            if shift_back:
                tmp += LL
            mabrs.append(tmp)
        return arrays_to_Geo(mabrs, kind=2, info="mabr")

    def triangulate(self, as_one=False, as_polygon=True):
        """Delaunay triangulation for point groupings."""
        if as_one:
            shps = [self.XY]
        else:
            shps = self.shapes
        out = [geom._tri_pnts_(s) for s in shps]
        kind = 2 if as_polygon else 1
        g, ift, extent = array_IFT(out)
        return Geo(g, ift, Kind=kind, Extent=self.XT, Info="triangulation")

    # ---- (5) conversions ---------------------------------------------
    #
    def fill_holes(self):
        """Fill holes in polygon shapes.  Returns a Geo array."""
        if self.K < 2:
            print("Polygon geometry required.")
            return None
        a_2d = np.vstack(self.first_bit(False))
        tmp_ift = self.IFT[self.Bit == 0]
        tmp_ft = [(j - i) for i, j in tmp_ift[:, 1:3]]
        id_too = np.zeros((len(tmp_ft), 2), dtype=np.int32)
        cs = np.cumsum(tmp_ft)
        id_too[1:, 0] = cs[:-1]
        id_too[:, 1] = cs
        info = "{} fill_holes".format(self.Info)
        tmp_ift[:, 1:3] = id_too
        return Geo(a_2d, IFT=tmp_ift, Kind=2, Extent=self.XT, Info=info)

    def holes_to_shape(self):
        """Return holes in polygon as shapes.  Returns a Geo array or None."""
        if self.K < 2:
            print("Polygon geometry required.")
            return None
        return self.inner_rings(asGeo=True, to_clockwise=True)

    def multipart_to_singlepart(self, info=""):
        """Convert multipart shapes to singleparts.  Return a new Geo array."""
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
        """Construct origin-destination pairs.

        Traversing around the perimeter of polygons, along polylines or
        between point sequences.

        Returns
        -------
        An list of arrays of origin-destination pairs for each bit in a shape.

        See Also
        --------
        `segment_polys` Geo method and `polys_to_segments` in npg_geom.
        """
        # return [np.concatenate((p[:-1], p[1:]), axis=1) for p in self.bits]
        z = np.concatenate((self.XY[:-1], self.XY[1:]), axis=1)
        return [z[ft[0]:ft[1] - 1] for ft in self.FT]

    def polylines_to_polygons(self):
        """Return a polygon Geo type from a polyline Geo.

        It is assumed that the polylines form closed-loops, otherwise use
        `close_polylines`.
        """
        if self.K == 2:
            print("Already classed as a polygon.")
            return self
        polygons = self.copy()
        polygons.K = 2
        return polygons

    def polygons_to_polylines(self):
        """Return a polyline Geo type from a polygon Geo."""
        if self.K == 1:
            print("Already classed as a polyline.")
            return self
        polylines = Geo(self.XY, self.IFT, Kind=1, Extent=self.XT,
                        SR=self.SR)
        return polylines

    def polys_to_points(self, keep_order=True, as_structured=False):
        """Convert all feature vertices to an ndarray of unique points.

        Optionally, retain point order.  Optionally return a structured array.
        """
        if as_structured:
            arr = self + self.LL
            return geom._polys_to_unique_pnts_(arr, as_structured=True)
        uni, idx = np.unique(self, True, axis=0)
        if keep_order:
            uni = self[np.sort(idx)]
        return uni

    def segment_polys(self, as_basic=True, shift_back=False, as_3d=False):
        """Call `polys_to_segments`.  See `od_pairs` also."""
        return geom.polys_to_segments(
                   self, as_basic=True, shift_back=False, as_3d=False)

    def close_polylines(self, out_kind=1):
        """Produce closed-loop polylines (1) or polygons (2) from polylines.

        Multipart features are converted to single part.
        """
        polys = []
        for s in self.bits:  # shape as bits
            if len(s) > 2:
                if np.all(s[0] == s[-1]):
                    polys.append(s)
                else:
                    polys.append(np.concatenate((s, s[..., :1, :]), axis=0))
        g, ift, extent = array_IFT(polys)
        if out_kind not in (1, 2):
            out_kind = self.K
        return Geo(g, ift, out_kind, self.XT, "Closed polylines.")

    def densify_by_distance(self, spacing=1):
        """Densify poly features by a specified distance.

        Convert multipart to singlepart features during the process.
        Calls `_pnts_on_line_` for Geo bits.
        """
        polys = [geom._pnts_on_line_(a, spacing) for a in self.bits]
        g, ift, extent = array_IFT(polys)
        return Geo(g, ift, self.K, self.XT, "Densify by distance")

    def densify_by_percent(self, percent=50):
        """Densify poly features by a percentage for each segment.

        Converts multipart to singlepart features during the process.
        Calls `_percent_along`.
        """
        bits = self.bits
        polys = [geom._pnts_on_line_(a, spacing=percent, is_percent=True)
                 for a in bits]
        polys = [a + self.LL for a in polys]
        g0, ift, extent = array_IFT(polys)
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
            r = np.asarray([geom._dist_along_(a, dist=val) for a in self.bits])
        else:
            val = min(abs(val), 100.)
            r = np.asarray(
                [geom._percent_along_(a, percent=val) for a in self.bits])
        if as_structured:
            z = np.empty((r.shape[0], ), dtype=dt)
            z['OID_'] = np.arange(r.shape[0])
            z['X_'] = r[:, 0]
            z['Y_'] = r[:, 1]
            return z
        return r

    # ---- (6) segments for poly* boundaries
    #
    def segment_pnt_ids(self):
        """Return the segment point IDs."""
        p_ind = self.pnt_indices(as_structured=False)
        w = np.where(np.diff(self.pnt_ids, prepend=p_ind[0, 1]) == 1)[0]
        id_vals = np.split(self.pnt_indices(), w, axis=0)
        # ft_ids = [np.array(list(zip(b[:-1, 0], b[1:, 0]))) for b in id_vals]
        ft_ids = [np.concatenate((i[:-1, 0][:, None],
                                  i[1:, 0][:, None]), axis=1)
                  for i in id_vals]
        return ft_ids

    def to_segments(self, ignore_holes=True):
        """Segment poly* structures into (x0, y0, x1, y1) data per shape, as
        an object array.  Holes are removed from polygons by default.

        See Also
        --------
        `npg_helpers.polys_to_segments` for other output options.
        """
        if self.K not in (1, 2):
            print("Poly* features required.")
            return None
        # ---- basic return as ndarray used by common_segments
        if ignore_holes:
            fr_to = self.outer_rings(False)
        else:
            fr_to = self.bits
        segs = np.asarray([np.concatenate((b[:-1], b[1:]), axis=1)
                           for b in fr_to], dtype='O')
        return segs

    def common_segments(self, shift_back=False):
        """Return the common segments in poly features.

        The result is an array of  from-to pairs of points.  ft, tf pairs are
        evaluated to denote common and duplicates.

        Parameters
        ----------
        shift_back : boolean
            Whether to shift back to real-world coordinates.
        """
        bts = self.bits
        if len(bts) == 1:
            bts = bts[0]
            fr_to = np.concatenate((bts[:-1], bts[1:]), axis=1)
        else:
            fr_to = np.concatenate([np.concatenate((b[:-1], b[1:]), axis=1)
                                    for b in bts], axis=0)
        if fr_to is None:
            return None
        h_0 = uts(fr_to)
        names = h_0.dtype.names
        h_1 = h_0[list(names[2:4] + names[:2])]  # x_to, y_to and x_fr, y_fr
        idx = np.isin(h_0, h_1)
        common = h_0[idx]
        if shift_back:
            common[:, :2] += self.LL
            common[:, 2:] += self.LL
        return _fill_float_array(common), idx  # stu(common)

    def unique_segments(self, shift_back=False):
        """Return the unique segments in poly features.

        The output is an ndarray of from-to pairs of points.

        Parameters
        ----------
        shift_back : boolean
            Whether to shift back to real-world coordinates.
        """
        fr_to = np.concatenate([np.concatenate((b[:-1], b[1:]), axis=1)
                                for b in self.bits], axis=0)
        if fr_to is None:
            return None
        h_0 = uts(fr_to)
        names = h_0.dtype.names
        h_1 = h_0[list(names[2:4] + names[:2])]
        idx0 = ~np.isin(h_0, h_1)
        uniq01 = np.concatenate((h_0[idx0], h_0[~idx0]), axis=0)
        vals = _fill_float_array(uniq01)
        if shift_back:
            return vals + np.concatenate((self.LL, self.LL)), idx0
        return vals, idx0  # return stu(uniq01)

    # ---- (7) sort section -------------------------------------------------
    # Sorting the fc shape-related fields needs an advanced arcgis pro license.
    # The following applies to the various sort options.
    #
    def change_indices(self, new):
        """Return the old and new indices.

        Indices are derived from the application of a function.
        """
        if len(self.shp_ids) != len(new):
            print("Old and new ID lengths must be the same.")
            return None
        dt = np.dtype([('Orig_ID', '<i4'), ('New_ID', '<i4')])
        out = np.asarray(list(zip(self.IDs, new)), dtype=dt)
        return out

    def sort_by_area(self, ascending=True, just_indices=False):
        """Return geometry sorted by shape area.

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
        """Sort the geometry by ascending or descending order."""
        areas = self.lengths
        idx = np.argsort(areas)
        sorted_ids = self.IDs[idx]
        if not ascending:
            sorted_ids = sorted_ids[::-1]
        if just_indices:
            return self.change_indices(sorted_ids)
        sorted_array = self.pull_shapes(sorted_ids)
        return sorted_array

    def sort_by_extent(self, extent_pnt='LB', key=0, just_indices=False):
        """Sort the geometry using the conditions outlined below."""
        if extent_pnt not in ('LB', 'LT', 'C'):
            print("Extent point is incorrect... read the docs.")
            return
        if key not in range(0, 8):
            print("Integer value between 0 and 7 inclusive required.")
            return
        if extent_pnt == 'LB':
            ext = self.extent_corner('LB')  # extent left-bottom
        elif extent_pnt == 'LT':
            ext = self.extent_corner('LT')  # extent left-top
        else:
            ext = self.extent_centers()     # extent centers
        ext_ids = self.shp_ids
        xs = ext[:, 0]
        ys = ext[:, 1]
        azim = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # azimuths
        val = np.radians(azim[key])              # sort angle in radians
        z = np.sin(val) * xs + np.cos(val) * ys  # - sort by vector
        idx = np.argsort(z)  # sort order
        sorted_ids = ext_ids[idx]
        if just_indices:
            return sorted_ids
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
        `sort_by_extent` for polygons or polylines.
        """
        a = x_ascending
        b = y_ascending
        if a and b:
            return self.XY[np.lexsort((self.Y, self.X))]
        if a and not b:
            return self.XY[np.lexsort((-self.Y, self.X))]
        if not a and b:
            return self.XY[np.lexsort((self.X, self.Y))]
        if not a and not b:
            return self.XY[np.lexsort((-self.X, self.Y))]
        return self

    def radial_sort(self, asGeo=True):
        """Sort the coordinates of polygon/polyline features."""
        def _radsrt_(a, dup_first=False):
            """Worker for radial sort."""
            uniq = np.unique(a, axis=0)
            cent = np.mean(uniq, axis=0)
            dxdy = uniq - cent
            angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])
            idx = angles.argsort()
            srted = uniq[idx]
            if dup_first:
                return np.concatenate((srted, [srted[0]]), axis=0)[::-1]
            return srted
        # ----
        ift = self.IFT
        cw = self.CW
        kind = self.K
        dup_first = False
        if kind == 2:
            dup_first = True
        tmp = []
        for i, a in enumerate(self.bits):
            arr = _radsrt_(a, dup_first)
            if cw[i] == 0:
                arr = arr[::-1]
            tmp.append(arr)
        if asGeo:
            return Geo(np.vstack(tmp), IFT=ift)
        return tmp

    # ---- (8) info section -------------------------------------------------
    # ---- points: indices, info, find duplicates
    def pnt_indices(self, as_structured=False):
        """Return the point ids and the feature that they belong to."""
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

    def pnt_counts(self, splitter="bit"):
        """Point count by splitter.  Split by (b)it, (p)art (s)hape."""
        chunks = self.split_by(splitter)
        return np.array([len(i) for i in chunks])

    def dupl_pnts(self, as_structured=False):
        """Duplicate points as a structured array."""
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

    def uniq_pnts(self, as_structured=False):
        """Return unique points as a structured array."""
        uni, idx, cnts = np.unique(self, True, False, True, axis=0)
        uni = uni[cnts == 1]
        num = cnts[cnts == 1]
        if as_structured:
            dt = [('X_', '<f8'), ('Y_', '<f8'), ('Cnts', '<i4')]
            z = np.empty((uni.shape[0],), dtype=dt)
            z['X_'] = uni[:, 0]
            z['Y_'] = uni[:, 1]
            z['Cnts'] = num
            z = repack_fields(z)
            return z
        return uni

    def geom_check(self):
        """Run some geometry checks.  See __check_geometry__ for details."""
        return check_geometry(self)

    def structure(self):
        """Print array structure."""
        docs = """
        Geo array structure
        -------------------
        OID_    : self.Id   shape id
        Fr_pnt  : self.Fr   from point id
        To_pnt  : self.To   to point id for a shape
        CW_CCW  : self.CW   outer (1) inner/hole (0)
        Part_ID : self.PID  part id for each shape
        Bit_ID  : self.Bit  sequence order of each part in a shape
        ----
        """
        print(dedent(docs))
        npg_io.prn_tbl(self.IFT_str)


# End of class definition
#
# ---- (2) helper functions
#
# def is_ccw(a):
#     """Counterclockwise.  Used by `check_geometry"""
#     return 0 if _bit_area_(a) > 0. else 1


def is_Geo(obj, verbose=False):
    """Check the input to see if it is a Geo array.  Used by `roll_coords."""
    if hasattr(obj, "IFT"):
        return True
    if verbose:
        msg = "`{}`, is not a Geo array`. Use `arrays_toGeo` to convert."
        print(msg.format(obj.__class__))
    return False


# ---- main functions
# ---- (3) Geo from sequences
#  Construct the Geo array from sequences.
#     ndarrays, object arrays, nested lists, lists of arrays etcetera.
#
def roll_coords(self):
    """Roll point coordinates to a new starting position.

    Notes
    -----
    Rolls the coordinates of the Geo array attempting to put the start/end
    points as close to the lower-left of the ring extent as possible.
    """
    # ----
    def _LL_(arr):
        """Return the closest point to the lower left of the polygon."""
        LL = np.min(arr, axis=0)
        idx = (np.abs(arr - LL)).argmin(axis=0)
        return idx[0]
    # ----
    msg = "\nPolygon Geo array required for `npg.roll_coords`...\n"
    if is_Geo(self) and self.K != 2:
        print(msg)
        return None
    # ----
    arrs = []
    extent = self.XT  # aoi_extent().reshape(2, 2)
    for ar in self.bits:
        num = _LL_(ar)
        arrs.append(np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0))
    g = np.concatenate(arrs, axis=0)
    g = Geo(g, self.IFT, self.K, extent, "rolled", None)
    return g


def array_IFT(in_arrays, shift_to_origin=False):
    """Produce the Geo array construction information.

    in_arrays : array-like
        A container (array, list etc) whose entities represent valid 2D shapes.
    shift_to_origin : boolean
        True returns coordinates shifted to the origin by subtracting x-y min
        from each value.
    """
    id_too = []
    a_2d = []
    if isinstance(in_arrays, (list, tuple)):
        in_arrays = np.asarray(in_arrays, dtype='O')
    if isinstance(in_arrays, np.ndarray):
        if in_arrays.ndim == 2 or len(in_arrays) == 1:
            in_arrays = [in_arrays]
    subs = []  # ** added
    for cnt, p in enumerate(in_arrays):
        if len(p[0]) > 2:
            p = np.asarray(p, dtype='O')
        else:
            p = np.asarray(p, dtype='float')
        kind = p.dtype.kind
        if (kind == 'O') or (len(p.shape) > 2):  # -- object and ndim=3 arrays
            bits = []
            sub = []   # ** added
            b_id = 0
            for j in p:
                if isinstance(j, (list, tuple)):
                    j = np.asarray(j)
                if len(j.shape) == 2:
                    bits.append(np.asarray(j).squeeze())
                    id_too.append([cnt, b_id, len(j)])
                    b_id += 1
                elif (j.dtype.kind == 'O') or (len(j.shape) > 2):
                    for k in j:
                        bits.append(np.asarray(k).squeeze())
                        id_too.append([cnt, b_id, len(k)])
                        b_id += 1
            sub.append(np.vstack(bits))  # ** added
        elif kind in NUMS:                       # -- ndarrays, ndim=2
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
    pnt_nums = np.zeros_like(ids, dtype=np.int32)
    u, i, cnts = np.unique(ids, True, return_counts=True)
    pnt_nums = np.concatenate([np.arange(i) for i in cnts])
    IFT = np.array(list(zip(ids, frum, too, CW, part, pnt_nums)))
    extent = np.array([np.min(a_stack, axis=0), np.max(a_stack, axis=0)])
    if shift_to_origin:
        a_stack = a_stack - extent[0]
    # recheck clockwise values for the array
    return a_stack, IFT, extent


def arrays_to_Geo(in_arrays, kind=2, info=None):
    """Produce a Geo class object from a list/tuple of arrays.

    Parameters
    ----------
    in_arrays : list
        `in_arrays` can be created by adding existing 2D arrays to a list.
        You can also convert poly features to arrays using ``poly2arrays``.
    Kind : integer
        Points (0), polylines (1) or polygons (2).

    Requires
    --------
    array_IFT

    Returns
    -------
    A `Geo` class object based on a 2D np.ndarray (a_2d) with an array of
    indices (IFT) delineating geometry from-to points for each shape and its
    parts.

    Notes
    -----
    `a_2d` will be returned as an object array from `array_IFT` in many cases.
    I needs to be recast to a `float` array to proceed.

    See Also
    --------
    **npg_arc_npg.fc_geometry** to produce `Geo` objects directly from arcgis
     pro featureclasses.
    """
#    if kind == 2:  # check for proper polygon points
#        in_arrays = [np.asarray(i) for i in in_arrays]
    a_2d, ift, extent = array_IFT(in_arrays)     # ---- call array_IFT
    a_2d = a_2d.astype(np.float)  # see Notes
    rows, cols = ift.shape
    z0 = np.full((rows, 6), fill_value=-1, dtype=ift.dtype)
    z0[:, :cols] = ift
    # do the clockwise check here and correct any assumptions
    g = Geo(a_2d, z0, Kind=kind, Extent=extent, Info=info)
    if kind == 2:  # ---- Polygons require a recheck of ring order.
        old_CW = g.CW
        _c = [_bit_area_(i) > 0 for i in g.bits]
        CW_check = np.asarray(_c, dtype='int')
        if not np.all(old_CW == CW_check):
            z0[:, 3] = CW_check
            fix_prt = [np.cumsum(g.CW[g.IDs == i]) for i in g.U]
            z0[:, 4] = np.concatenate(fix_prt)
            w = np.where(z0[:, 3] == 1)[0]
            dif = w[1:] - w[:-1]
            if len(dif) > 1:  # *** added
                fix_seq = np.concatenate([np.arange(i) for i in dif])
                z0[:len(fix_seq), 5] = fix_seq
            g = Geo(a_2d, z0, Kind=kind, Extent=extent, Info=info)
    return g


# =========================================================================
# ---- (4) Geo to arrays
def Geo_to_arrays(g, shift_back=True):
    """Geo array to ndarray.

    Returns
    -------
    Most likely an object array of ndarrays (aka, a ragged array).  To shift
    the coordinates back to their original extent, set `shift_back` to True.
    """
    ift = g.IFT
    ids = g.IDs
    uniq_ids = g.U
    out = []
    shift = [0.0, 0.0]  # null shift
    if shift_back:
        shift = g.LL
    for i in uniq_ids:
        shps = ift[ids == i]
        subs = []
        uniq, idx = np.unique(shps[:, 4], True)
        for u in uniq:
            sub = []
            s = shps[shps[:, 4] == u]
            for part in s:
                fr, too = part[1:3]
                xy = np.asarray(g.XY[fr:too] + shift)
                sub.append(xy.squeeze())
            check = [len(i) for i in sub]
            if min(check) != max(check):
                subs.append(np.asarray(sub, dtype='O'))
            else:
                subs.append(np.asarray(sub, dtype=np.float).squeeze())
        out.extend(subs)  # np.asarray(subs))
    return np.asarray(out, dtype='O')


def Geo_to_lists(g, shift_back=True):
    """Geo array to lists.

    Returns
    -------
    Most likely an object array of ndarrays (aka, a ragged array).  To shift
    the coordinates back to their original extent, set `shift_back` to True.
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
                if shift_back:
                    xy = (g.XY[fr:too] + g.LL).tolist()
                else:
                    xy = g.XY[fr:too].tolist()
                sub.append(xy)
            if len(sub) > 1:
                subs.append(sub)
            else:
                subs.append(sub[0])
        out.extend(subs)
    return out


# -------------------------------
def _fill_float_array(arr):
    """Fill an array of floats from a structured array of floats as an ndarray.

    This is a simplified version of `stu`.  Used by the `common_segments` and
    `unique_segments` methods.
    """
    names = arr.dtype.names
    n = len(names)
    a_2d = np.zeros((arr.size, n), dtype=np.float)
    for i, name in enumerate(names):
        a_2d[:, i] = arr[name]
    return a_2d


# ----------------------------------------------------------------------------
# ---- (5) other functions
#
def check_geometry(self):
    """Run some geometry checks.

    Requires
    --------
    `_is_ccw_` and `_bit_area_` for the calculations.

    Performs clockwise/counterclockwise (CW/CCW) checks.

    - Outer rings must consist of CW ordered points.
    - First ring must be CW.
    - Inner rings (aka, holes), have points in CCW order.
    """
    def _is_ccw_(a):
        """Counterclockwise.  Used by `check_geometry"""
        return 0 if _bit_area_(a) > 0. else 1

    ift = self.IFT
    c0 = self.To[-1]
    c1 = self.shape
    shape_check = c0 == c1[0]
    if not shape_check:
        args = [c0, c1[0], c1, ift]
        print("g.To and g.IFT error\n{} != {}, shape={}\n{}".format(*args))
        return None
    ft = ift[self.CW == 1][:, 1:3]
    check_0 = np.logical_and(self.CW == 0, self.Bit == 0)
    check_1 = [_is_ccw_(self[i: j]) for i, j in ft]
    if np.sum(check_0) > 0:
        c2 = self.IDs[check_0]
        print("\ng.CW and g.Bit mismatch\n{}\n...IFT info\n".format(c2))
        npg_io.prn_tbl(self.IFT_str)
    elif np.sum(check_1) > 0:
        c3 = self.IDs[check_1]
        print("\nError in ring orientation\n... shapes {}\n".format(c3))
        print("IFT information...\n")
    else:
        print("no errors found")
    return


#  use `dirr` for extended documentation
#
def dirr(obj, colwise=False, cols=3, prn=True):
    r"""Return a formatted `dir` listing of an object, module, function."""
    if is_Geo(obj):
        # ('Geo' in str(type(obj))) & (issubclass(obj.__class__, np.ndarray))
        sdo = set(dir(obj))
        sda = set(dir(np.ndarray))
        a = ['... Geo class ...', "", ""]
        a.extend(sorted(list(sdo.difference(sda))))
        a.extend(["\n... Helpers ...", "", ""])
        a.extend(sorted(npg_helpers.__all__))
        a.extend(["\n... Geom helpers ...", "", ""])
        a.extend(sorted(geom.__all__))
        a.extend(["\n... I/O helpers ...", "", ""])
        a.extend(sorted(npg_io.__all__))
    else:
        a = dir(obj)
    w = max([len(i) for i in a])
    frmt = (("{{!s:<{}}} ".format(w)))*cols
    csze = len(a) / cols  # split it
    csze = int(csze) + (csze % 1 > 0)
    if colwise:
        a_0 = [a[i: i+csze] for i in range(0, len(a), csze)]
        a_0 = list(zip(*a_0, fillvalue=""))
    else:
        a_0 = [a[i: i+cols] for i in range(0, len(a), cols)]
    if hasattr(obj, '__module__'):
        args = ["-"*70, obj.__module__, obj.__class__]
    else:
        args = ["-"*70, type(obj), "npg.dirr..."]
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
    if g.__class__.__name__ != "Geo":
        print("\nGeo array expected...\n")
        return
    arr_set = set(dir(g.XY))
    geo_set = set(dir(g))
    srt = sorted(list(geo_set.difference(arr_set)))
    t = ", ".join([str(i) for i in srt])
    w = wrap(t, 70)
    print(">>> geo_info(geo_array)\n... Geo methods and properties.")
    for i in w:
        print(indent("{}".format(i), prefix="    "))
    print("\n... Geo base properties.")
    s0 = set(srt)
    s1 = set(Geo.__dict__.keys())
    s0s1 = sorted(list(s0.difference(s1)))
    t = ", ".join([str(i) for i in s0s1])
    w = wrap(t, 70)
    for i in w:
        print(indent("{}".format(i), prefix="    "))
    print("\n... Geo special.")
    s1s0 = sorted(list(s1.difference(s0)))
    t = ", ".join([str(i) for i in s1s0])
    w = wrap(t, 70)
    for i in w:
        print(indent("{}".format(i), prefix="    "))
    return


# ---- displaying Geo and ndarrays
#
def _svg(g, as_polygon=True):
    """Format and show a Geo array, np.ndarray or list structure in SVG format.

    Notes
    -----
    Geometry must be expected to form polylines or polygons.
    IPython required.

    >>> from IPython.display import SVG

    Alternate colors::

        white, silver, gray black, red, maroon, purple, blue, navy, aqua,
        green, teal, lime, yellow, magenta, cyan
    """
    def svg_path(g_bits, scale_by, o_f_s):
        """Make the svg from a list of 2d arrays"""
        opacity, fill_color, stroke = o_f_s
        pth = [" M {},{} " + "L {},{} "*(len(b) - 1) for b in g_bits]
        ln = [pth[i].format(*b.ravel()) for i, b in enumerate(g_bits)]
        pth = "".join(ln) + "z"
        s = ('<path fill-rule="evenodd" fill="{0}" stroke="{1}" '
             'stroke-width="{2}" opacity="{3}" d="{4}"/>'
             ).format(fill_color, stroke, 1.5 * scale_by, opacity, pth)
        return s
    # ----
    msg0 = "\nImport error..\n>>> from IPython.display import SVG\nfailed."
    msg1 = "A Geo array or ndarray (with ndim >=2) is required."
    # ----
    # Geo array, np.ndarray check
    try:
        from IPython.core.display import SVG  # 2020-07-02
        # from IPython.display import SVG
    except ImportError:
        print(dedent(msg0))
        return None
    # ---- checks for Geo or ndarray. Convert lists, tuples to np.ndarray
    if isinstance(g, (list, tuple)):
        g = np.asarray(g)
    if ('Geo' in str(type(g))) & (issubclass(g.__class__, np.ndarray)):
        GA = True
        g_bits = g.bits
        L, B = g.min(axis=0)
        R, T = g.max(axis=0)
    elif isinstance(g, np.ndarray):
        GA = False
        if g.ndim == 2:
            g_bits = [g]
            L, B = g.min(axis=0)
            R, T = g.max(axis=0)
        elif g.ndim == 3:
            g_bits = [g[i] for i in range(g.shape[0])]
            L, B = g.min(axis=(0, 1))
            R, T = g.max(axis=(0, 1))
        elif g.dtype.kind == 'O':
            g_bits = []
            for i, b in enumerate(g):
                b = np.array(b)
                if b.ndim == 2:
                    g_bits.append(b)
                elif b.ndim == 3:
                    g_bits.extend([b[i] for i in range(b.shape[0])])
            L, B = np.min(np.vstack([np.min(i, axis=0) for i in g_bits]),
                          axis=0)
            R, T = np.max(np.vstack([np.max(i, axis=0) for i in g_bits]),
                          axis=0)
        else:
            print(msg1)
            return None
    else:
        print(msg1)
        return None
    # ----
    # derive parameters
    if as_polygon:
        o_f_s = ["0.75", "red", "black"]  # opacity, fill_color, stroke color
    else:
        o_f_s = ["1.0", "none", "red"]
    # ----
    d_x, d_y = (R - L, T - B)
    hght = min([max([150., d_y]), 200])  # ---- height 150 to 200
    width = int(d_x/d_y * hght)
    scale_by = max([d_x, d_y]) / max([width, hght])
    # ----
    # derive the geometry path
    pth_geom = svg_path(g_bits, scale_by, o_f_s)  # ---- svg path string
    # construct the final output
    view_box = "{} {} {} {}".format(L, B, d_x, d_y)
    transform = "matrix(1,0,0,-1,0,{0})".format(T + B)
    hdr = '<svg xmlns="http://www.w3.org/2000/svg" ' \
          'xmlns:xlink="http://www.w3.org/1999/xlink" '
    f0 = 'width="{}" height="{}" viewBox="{}" '.format(width, hght, view_box)
    f1 = 'preserveAspectRatio="xMinYMin meet">'
    f2 = '<g transform="{}">{}</g></svg>'.format(transform, pth_geom)
    s = hdr + f0 + f1 + f2
    if GA:  # Geo array display
        g.SVG = s
        return SVG(g.SVG)  # plot the representation
    return SVG(s)  # np.ndarray display


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
