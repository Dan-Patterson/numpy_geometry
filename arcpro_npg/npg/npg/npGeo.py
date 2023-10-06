# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
------------------------------------------
  npGeo: Geo class, properties and methods
------------------------------------------

The Geo class is a subclass of numpy's ndarray.  Properties that are related
to geometry have been assigned and methods developed to return geometry
properties.

----

"""
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E1101,E1121
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0621
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915

import sys
from textwrap import indent, dedent, wrap
import numpy as np
# from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import repack_fields

from npg import npg_geom as geom
from npg import (npg_helpers, npg_io, npg_prn)  # npg_create)
from npg import npg_min_circ as sc

from npg.npg_helpers import (
    _angles_3pnt_, _area_centroid_, _bit_area_, _bit_crossproduct_,
    _bit_min_max_, _bit_length_, _rotate_, polyline_angles, uniq_1d)

from npg.npgDocs import (
    Geo_hlp, array_IFT_doc, dirr_doc, shapes_doc, parts_doc, get_shapes_doc,
    inner_rings_doc, outer_rings_doc, is_in_doc, convex_hulls_doc,
    bounding_circles_doc,
    extent_rectangles_doc, od_pairs_doc, pnt_on_poly_doc,
    sort_by_area_doc,
    radial_sort_doc, sort_by_extent_doc)  #

np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 6.2f}'.format})

script = sys.argv[0]  # print this should you need to locate the script

FLOATS = np.typecodes['Float']
INTS = np.typecodes['Integer']
NUMS = FLOATS + INTS
TwoPI = np.pi * 2.0

__all__ = [
    'Geo', 'is_Geo',
    'roll_coords', 'roll_arrays', 'array_IFT', 'arrays_to_Geo',
    'Geo_to_arrays', 'Geo_to_lists', '_fill_float_array',
    'is_Geo', 'reindex_shapes', 'remove_seq_dupl', 'check_geometry',
    'dirr'
]


# ----------------------------------------------------------------------------
# ---- (1) ... Geo class, properties and methods ... -------------------------
#
class Geo(np.ndarray):
    """Geo class.

    This class is based on NumPy's ndarrays.  They are created using
    `npg.arrays_to_Geo`, `npg.array_IFT` and `npg.roll_coords`.
    See `npg.npg_arc_npg` and `npg.npg_io` for methods to acquire the ndarrays
    needed from other data sources.
    """

    __name__ = "npGeo"
    __module__ = "npg"
    __author__ = "Dan Patterson"
    __doc__ = Geo_hlp

    def __new__(cls,
                arr=None,
                IFT=None,
                Kind=2,
                Extent=None,
                Info="Geo array",
                SR=None
                ):
        """See `npgDocs` for construction notes."""
        arr = np.ascontiguousarray(arr)
        IFT = np.ascontiguousarray(IFT)
        if (arr.ndim != 2) or (IFT.ndim != 2):
            m = "Input error... arr.ndim != 2 : {} or IFT.dim != 2 : {}"
            print(dedent(m).format(arr.ndim, IFT.ndim))
            return None
        if (IFT.shape[-1] < 6) or (Kind not in (1, 2)):
            print(dedent(Geo_hlp))
            return None
        # --
        self = arr.view(cls)      # view as Geo class
        self.IFT = IFT            # array id, fr-to, cl, part id
        self.K = Kind             # Points (0), Polylines (1), Polygons (2)
        self.Info = Info          # any useful information
        self.IDs = IFT[:, 0]      # shape id
        self.Fr = IFT[:, 1]       # from point id
        self.To = IFT[:, 2]       # to point id
        self.CL = IFT[:, 3]       # clockwise and outer/inner ring identifier
        self.PID = IFT[:, 4]      # part identifier per shape
        self.Bit = IFT[:, 5]      # bit sequence for each shape... arr.bit_seq
        self.FT = IFT[:, 1:3]     # from-to sequence
        self.IP = IFT[:, [0, 4]]  # shape and part id together
        # --- other properties
        uni = np.unique(self.IDs)
        self.N = len(uni)         # sample size, unique shapes
        self.U = uni              # unique ids self.IDs[idx]
        self.SR = SR              # spatial reference
        self.XT = Extent          # extent of all features
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
        self.SVG = ""
        return self

    def __array_finalize__(self, src_arr):
        """
        Finalize new object....

        This is where housecleaning takes place for explicit, view casting or
        new from template... `src_arr` is either None, any subclass of
        ndarray including our own (words from documentation) OR another
        instance of our own array.
        You can use the following with a dictionary instead of None:

        >>> self.info = getattr(obj,'info',{})
        """
        if src_arr is None:
            return None
        self.IFT = getattr(src_arr, 'IFT', None)
        self.K = getattr(src_arr, 'K', None)
        self.Info = getattr(src_arr, 'Info', None)
        self.IDs = getattr(src_arr, 'IDs', None)
        self.Fr = getattr(src_arr, 'Fr', None)
        self.To = getattr(src_arr, 'To', None)
        self.CL = getattr(src_arr, 'CL', None)
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
        self.SVG = getattr(src_arr, 'SVG', None)

    def __array_wrap__(self, out_arr, context=None):
        """Wrap it up."""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # ----  ------------------------------------------------------------------
    # ---- End of class definition -------------------------------------------
    #
    # ---- help : information
    @property
    def H(self):
        """Print the documentation for an instance of the Geo class."""
        print(Geo_hlp)

    @property
    def facts(self):
        """Convert an IFT array to full information.

        Only the first 50 records (maximum) will be printed. To see the data
        structure, and/or more records use the `prn_geo` method.
        """
        info_ = self.IFT_str[:25]
        frmt = """
        {}\nExtents :\n  LL {}\n  UR {}
        Shapes :{:>6.0f}
        Parts  :{:>6.0f}
        Points :{:>6.0f}
        \nSp Ref : {}
        """
        args = ["-" * 14, self.LL, self.UR, len(self.U), self.IFT.shape[0],
                self.IFT[-1, 2], self.SR]
        print(dedent(frmt).format(*args))
        npg_prn.prn_tbl(info_)

    @property
    def structure(self):
        """Print array structure."""
        docs = """
        Geo array structure
        -------------------
        OID_    : self.Id   shape id
        Fr_pnt  : self.Fr   from point id
        To_pnt  : self.To   to point id for a shape
        CL      : self.CL   K=2 outer (1) inner (0): K=1 closed(1) open (0)
        Part_ID : self.PID  part id for each shape
        Bit_ID  : self.Bit  sequence order of each part in a shape
        ----
        """
        print(dedent(docs))
        npg_prn.prn_tbl(self.IFT_str)

    @property
    def geo_props(self):
        """Differences between Geo and ndarray methods and properties."""
        arr_set = set(dir(self.XY))
        geo_set = set(dir(self))
        srt = sorted(list(geo_set.difference(arr_set)))
        t = ", ".join([str(i) for i in srt])
        w = wrap(t, 79)
        print(">>> geo_info(geo_array)\n... Geo methods and properties.")
        for i in w:
            print(indent("{}".format(i), prefix="    "))
        print("\n... Geo base properties.")
        s0 = set(srt)
        s1 = set(Geo.__dict__.keys())
        s0s1 = sorted(list(s0.difference(s1)))
        t = ", ".join([str(i) for i in s0s1])
        w = wrap(t, 79)
        for i in w:
            print(indent("{}".format(i), prefix="    "))
        print("\n... Geo special.")
        s1s0 = sorted(list(s1.difference(s0)))
        t = ", ".join([str(i) for i in s1s0])
        w = wrap(t, 79)
        for i in w:
            print(indent("{}".format(i), prefix="    "))
        # return

    # ----  ---------------------------
    # ---- IFT : shape, part, bit
    # see also self.U, self.PID
    @property
    def shp_IFT(self):
        """Shape IFT values.  `shp_IFT == part_IFT` for singlepart shapes."""
        if self.is_multipart():  # multiparts check
            return self.part_IFT
        df = self.To - self.Fr
        cnt = np.bincount(self.IDs, df)  # -- check to account for
        gt0 = np.nonzero(cnt)[0]         # -- discontinuous ids
        too = np.cumsum(cnt[gt0], axis=0, dtype=np.int32)
        fr = np.concatenate(([0], too[:-1]), axis=0)
        ift = np.full((len(fr), 6), -1, dtype=np.int32)
        ift[:, :3] = np.asarray([self.U, fr, too]).T
        return ift

    @property
    def part_IFT(self):
        """Part IFT values."""
        idx = np.concatenate((self.Fr[self.Bit == 0], [len(self)]))
        fr_to = np.array([idx[:-1], idx[1:]]).T
        w = np.nonzero(self.Bit == 0)[0]  # self.CL == 1 2021-06-06 check
        ifts = self.IFT[w]     # slice by bit sequence
        ifts[:, 1:3] = fr_to   # substitute in the new from-to values
        return ifts

    @property
    def bit_IFT(self):
        """Bit IFT values."""
        return self.IFT

    @property
    def inner_IFT(self):
        """Inner ring/hole IFT values."""
        if self.K == 2:
            return self.IFT[self.Bit != 0]
        print("Polylines don't have inner rings.")
        return None

    @property
    def outer_IFT(self):
        """Outer ring IFT values."""
        if self.K == 2:
            return self.IFT[self.Bit == 0]
        print("Polylines don't have inner rings.")
        return None

    @property
    def IFT_str(self):
        """Geo array structure.  See self.structure for more information."""
        if self.K == 2:
            nmes = ["OID_", "Fr_pnt", "To_pnt", "CW/CCW", "Part_ID", "Bit_ID"]
        else:
            nmes = ["OID_", "Fr_pnt", "To_pnt", "Closed", "Part_ID", "Bit_ID"]
        return uts(self.IFT, names=nmes, align=False)

    # ----  ---------------------------
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

        The sequence is numbered from zero to `n`.  A shape can consist of a
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
        """Return a structured array of each shape's points coordinates.

        Points are shifted back to their original bounding box.
        """
        N = self.shape[0]
        dt = [('ID_', '<i4'), ('ShpID_', '<i4'), ('X_', '<f8'), ('Y_', '<f8')]
        z = np.empty((N,), dtype=dt)
        z['ID_'] = np.arange(N)
        z['ShpID_'] = self.pnt_ids
        z['X_'] = self.X + self.LL[0]
        z['Y_'] = self.Y + self.LL[1]
        z = repack_fields(z)
        return z

    # ----  ---------------------------
    # ---- counts : shape, part, bit
    @property
    def shp_pnt_cnt(self):
        """Points in each shape.  Columns: shape, points."""
        df = self.To - self.Fr
        cnt = np.bincount(self.IDs, weights=df)[1:]
        gt0 = np.nonzero(cnt)[0]         # -- discontinuous ids
        out = np.zeros((self.N, 2), dtype=np.int32)
        out[:, 0] = self.U
        out[:, 1] = cnt[gt0]
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

    # ----  ---------------------------
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
        gt0 = np.nonzero(cnt)[0]         # -- account for discontinuous ids
        too = np.cumsum(cnt[gt0], axis=0, dtype=np.int32)[1:]
        fr = np.concatenate(([0], too[:-1]), axis=0)
        fr, too = [uniq_1d(i) for i in [fr, too]]  # use uniq_1d vs unique
        fr_to = np.concatenate((fr[:, None], too[:, None]), axis=1)
        return [self.XY[ft[0]:ft[1]] for ft in fr_to]
    # shapes.__doc__ += shapes_doc

    @property
    def parts(self):
        """Deconstruct the 2D array into its parts."""
        fr_to = self.part_IFT[:, 1:3]
        return [self.XY[ft[0]:ft[1]] for ft in fr_to]
    # parts.__doc__ += parts_doc

    @property
    def bits(self):
        """Deconstruct the 2D array returning all rings."""
        return [self.XY[ft[0]:ft[1]] for ft in self.FT]

    # ----  ------------------------------------------------------------------
    # ---- methods and derived properties section
    # ---- (1) slicing, sampling equivalents
    #
    # Common Parameters
    #
    # splitter : b, p, s
    #     Split by (b)it, (p)art (s)hape.
    #
    def all_shapes(self):
        """Return the shapes."""
        return self.shapes

    def all_parts(self):
        """Return the shapes."""
        return self.parts

    def outer_rings(self, asGeo=True):
        """Get the first bit of multipart shapes and/or shapes with holes."""
        if np.any(self.CL == 0):
            ift_s = self.outer_IFT  # self.IFT[self.Bit == 0]
            a_2d = [self.XY[ft[1]:ft[2]] for ft in ift_s]
            if asGeo:
                a_2d = np.concatenate(a_2d, axis=0)
                ft = np.concatenate(([0], ift_s[:, 2] - ift_s[:, 1]))
                c = np.cumsum(ft)
                ift_s[:, 1] = c[:-1]
                ift_s[:, 2] = c[1:]
                return Geo(a_2d, ift_s, self.K, self.XT, Info="outer rings")
            return a_2d
        if asGeo:
            return self
        return self.bits

    def inner_rings(self, asGeo=False, to_clockwise=False):
        """Collect the holes of a polygon shape."""
        ift_s = self.inner_IFT  # self.IFT[self.Bit != 0]
        if ift_s.size == 0:  # print("\nNo inner rings\n")
            return None
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
            if a_2d.ndim == 2:
                bits = [a_2d]
            else:
                bits = a_2d
            z = np.asarray([_bit_min_max_(pnts) for pnts in bits])
            xtent = np.concatenate(
                (np.min(z[:, :2], axis=0), np.max(z[:, 2:], axis=0)))
            xtent = xtent.reshape(2, 2)
            return Geo(a_2d, ift_s, self.K, xtent, Info="inner rings")
        return a_2d  # np.asarray(a_2d, dtype='O')

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
        info = "{} first part".format(self.Info)
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
        """Pull multiple shapes, in the order provided.  Holes are retained."""
        if ids is None:
            return None
        if isinstance(ids, (int)):
            ids = [ids]
        ids = np.asarray(ids)
        # if not np.all([a in self.IDs for a in ids]):
        if not (ids[:, None] == self.IDs).any(-1).all():
            print("Not all required IDs are in the list provided")
            return None
        xys = []
        for id_num in ids:
            case = self.FT[self.IDs == id_num]
            if len(case) > 1:
                sub = [self.XY[c[0]: c[-1]] for c in case]
                xys.append(sub)
            else:
                c0, c1 = case.squeeze()
                xys.append(self.XY[c0:c1])
        if asGeo:
            info = "Old_order" + (" {}" * len(ids)).format(*ids)
            if len(ids) == 1:  # cludge workaround for length 1 ids
                arr = arrays_to_Geo(xys, kind=self.K, info=info)
                arr.IFT[:, 0] = 0
                # arr.IFT[:, 5] = np.arange(len(arr.IFT))  # fixed in arr2geo
                return arr
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

    # ----  ---------------------------
    # ---- (2) areas, centrality, lengths/perimeter for polylines/polygons
    #
    def areas(self, by_shape=True):
        """Area for the sub arrays using einsum based area calculations.

        Uses `npg_helpers._bit_area_` to calculate the area.
        The `by_shape=True` parameter returns the area for each shape. If
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
        """Polyline lengths or polygon perimeter. Uses bit_length."""
        if self.K not in (1, 2):
            print("Polyline/polygon representation is required.")
            return None
        bit_lengs = [np.sum(_bit_length_(i)) for i in self.bits]
        if by_shape:
            b_ids = self.bit_ids
            return np.bincount(b_ids, weights=bit_lengs)[self.U]
        return bit_lengs

    def perimeters(self, by_shape=True):
        """Polyline lengths or polygon perimeter. Calls `lengths`."""
        return self.lengths(by_shape=True)

    def centers(self, by_shape=True):
        """Return the center of outer ring points."""
        if by_shape:
            shps = self.shapes
            if self.K == 2:
                return np.stack([np.mean(s[:-1], axis=0) for s in shps])
            return np.stack([np.mean(s, axis=0) for s in shps])
        # -- part centers
        o_rings = self.outer_rings(False)
        # Remove duplicate start-end for polygons. Use all points otherwise.
        if self.K == 2:
            return np.stack([np.mean(r[:-1], axis=0) for r in o_rings])
        return np.stack([np.mean(r, axis=0) for r in o_rings])

    def centroids(self):
        """Return the centroid of the polygons using `_area_centroid_`."""
        def weighted(x_y, ids, areas):
            """Weight coordinate by area, x_y is either the x or y."""
            w = x_y * areas               # area weighted x or y
            w1 = np.bincount(ids, w)      # [Ids] weight / bin size
            ar = np.bincount(ids, areas)  # [I]  # areas per bin
            gt0 = np.nonzero(w1)[0]       # account for discontinuous ids
            w1 = w1[gt0]
            ar = ar[gt0]
            return w1 / ar
        # --
        if self.K != 2:
            print("Polygons required.")
            return None
        centr = []
        areas = []
        o_rings = self.outer_rings(True)
        ids = o_rings.part_ids  # unique shape ID values
        for ID in o_rings.U:
            parts_ = o_rings.part_IFT[ids == ID]
            out = [np.asarray(o_rings.XY[p[1]:p[2]]) for p in parts_]
            for prt in out:
                area, cen = _area_centroid_(prt)  # -- determine both
                centr.append(cen)
                areas.append(area)
        centr = np.asarray(centr)
        areas = np.asarray(areas)
        xs = weighted(centr[:, 0], ids, areas)
        ys = weighted(centr[:, 1], ids, areas)
        return np.concatenate((xs[:, None], ys[:, None]), axis=1)

    # ----  ---------------------------
    # ---- (3) extents and extent shapes
    #
    def aoi_extent(self, shift_back=False):
        """Return geographic extent of the `aoi` (area of interest)."""
        if shift_back:
            return self.XT.ravel()
        return (self.XT - self.XT[0]).ravel()

    def aoi_rectangle(self, shift_back=False):
        """Derive polygon bounds from the aoi_extent."""
        L, B, R, T = self.aoi_extent(shift_back)
        return np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])

    def extents(self, splitter="part"):
        """Return extent as L(eft), B(ottom), R(ight), T(op)."""
        if self.N == 1:
            splitter = "bit"
        chunks = self.split_by(splitter)
        return np.asarray([_bit_min_max_(c) for c in chunks])

    def extent_centers(self, splitter="shape"):
        """Return extent centers."""
        ext = self.extents(splitter)
        xs = (ext[:, 0] + ext[:, 2]) / 2.
        ys = (ext[:, 1] + ext[:, 3]) / 2.
        return np.concatenate((xs[:, None], ys[:, None]), axis=1)

    def extent_corner(self, corner='LB'):
        """Return extent centers. Use `LB` or `LT` for left bottom or top."""
        if corner not in ('LB', 'LT'):
            print("Only Left-bottom (LB) and Left-top (LT) are implemented")
            return None
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
        """Return extent polygons for the whole shape or the shape by bit."""
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

    # ----  ---------------------------
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
        if by_bit:
            chunks = self.bits
        else:
            chunks = self.shapes
        if remove_dups:
            chunks = [np.unique(i, axis=0) for i in chunks]
        return np.asarray([np.mean(i, axis=0) for i in chunks])

    # ---- ===================================================================
    # ---- npg_geom methods/properties required
    # ----  ------------------------------------------------------------------
    # ---- (1) **is** section, condition/case checking, kept to a minimum
    def is_clockwise(self, is_closed_polyline=False, as_structured=False):
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
        out = np.concatenate((self.bit_ids[:, None], self.CL[:, None]), axis=1)
        if as_structured:
            return uts(np.asarray(out), names=['IDs', 'Clockwise'])
        return out

    def is_convex(self, as_structured=False):
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
        out = np.array([np.all(np.sign(i) >= 0) for i in check])
        if as_structured:
            return uts(np.asarray(out), names=['IDs', 'Convex'])
        return out

    def is_multipart(self):
        """Return if any shape is a multipart."""
        return np.any(self.shp_part_cnt[:, 1] > 1)

    def is_multipart_report(self, as_structured=False):
        """For each shape, returns whether it has multiple parts."""
        partcnt = self.shp_part_cnt
        w = np.where(partcnt[:, 1] > 1, 1, 0)
        arr = np.concatenate((self.U[:, None], w[:, None]), axis=1)
        if as_structured:
            return uts(arr, names=['IDs', 'multipart'])
        return arr

    def is_in(self, pnts, reverse_check=False, values=True, indices=True):
        """Return pnts, indices that are in the Geo array or the reverse."""
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

    # ----  ---------------------------
    # ---- (2) angles -----------------
    #
    def segment_angles(self, fromNorth=False):
        """Segment angles for all bits of a Geo array.

        Uses alternate slicing described in `Notes` in npGeo.
        """
        dxy = self.XY[1:] - self.XY[:-1]
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        if fromNorth:
            ang = np.mod((450.0 - ang), 360.)
        return [ang[ft[0]:ft[1] - 1] for ft in self.FT]

    def polyline_angles(self, right_side=True):
        """Polyline angles defined by sequential 3 points."""
        f_bits = self.first_bit(False)
        inside = False
        if right_side:
            inside = True
        return [_angles_3pnt_(p, inside, True) for p in f_bits]  # npg_helper

    def polygon_angles(self, inside=True, in_deg=True):
        """Sequential 3 point angles for a poly* outer rings for each shape."""
        f_bits = self.first_bit(False)
        return [_angles_3pnt_(p, inside, in_deg) for p in f_bits]  # npg_helper

    # ----  ---------------------------
    # ---- (3) alter geometry ---------
    #
    def moveto(self, x=0, y=0):
        """Shift/translate the dataset origin is the lower-left corner."""
        dx, dy = x, y
        if dx == 0 and dy == 0:
            dx, dy = np.min(self.XY, axis=0)
        return Geo(self.XY + [-dx, -dy], self.IFT, self.K,
                   self.XT + [-dx, -dy])

    def shift(self, dx=0, dy=0):
        """See `translate`."""
        return Geo(self.XY + [dx, dy], self.IFT, self.K, self.XT + [dx, dy])

    def translate(self, dx=0, dy=0):
        """Move/shift/translate by dx, dy to a new location."""
        return Geo(self.XY + [dx, dy], self.IFT, self.K, self.XT + [dx, dy])

    def rotate(self, as_group=True, angle=0.0, clockwise=False):
        """Rotate shapes about the group center or individually.

        Rotation is done by `_rotate_`.  A new geo array is returned.
        """
        if clockwise:
            angle = -angle
        angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, s), (-s, c)))
        out = _rotate_(self, R, as_group)  # requires a Geo array, self
        info = f"{self.Info} rotated"  # "{} rotated".format(self.Info)
        if not as_group:
            out = np.vstack(out)
        ext = np.concatenate((np.min(out, axis=0), np.max(out, axis=0)))
        ext = ext.reshape(2, 2)
        return Geo(out, self.IFT, self.K, ext, info)

    # ----  ---------------------------
    # ---- (4) bounding containers ----
    # **see also** extent properties above
    #
    def bounding_circles(self, angle=5, shift_back=False, return_xyr=False):
        """Bounding circles for features."""
        chk = False if len(self.IFT) > 1 else True
        chs = self.convex_hulls(chk, False, 50)  # check for singlepart shapes
        xyr = [sc.small_circ(s) for s in chs.shapes]
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
        """Convex hull for shapes.  Calls `_ch_` to control the method used."""
        # --

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
            tmp = _r_(tmp, cent=c, angle=ang, clockwise=True)
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
        out = [geom.triangulate_pnts(s) for s in shps]
        kind = 2 if as_polygon else 1
        g, ift, extent = array_IFT(out)
        return Geo(g, ift, Kind=kind, Extent=self.XT, Info="triangulation")

    # ----  ---------------------------
    # ---- (5) conversions ------------
    #
    def fill_holes(self):
        """Fill holes in polygon shapes.  Returns a Geo array."""
        if self.K < 2:
            print("Polygon geometry required.")
            return None
        if not np.any(self.CL == 0):
            return self
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
        z = self.outer_rings()
        data = z.XY
        # z1 = self.holes_to_shape()
        ift = z.IFT
        ids = np.arange(ift.shape[0])
        ift[:, 0] = ids                          # reset the ids
        # ift[:, -2] = np.ones(len(ift))           # reset the part ids to 1
        return Geo(data, IFT=ift, Kind=self.K, Extent=self.XT, Info=info)

    def fr_to_pnts(self):
        """See `od_pairs`."""
        return self.od_pairs()

    def od_pairs(self):
        """Construct origin-destination pairs."""
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
            return geom.polys_to_unique_pnts(arr, as_structured=True)
        uni, idx = np.unique(self, True, axis=0)
        if keep_order:
            uni = self[np.sort(idx)]
        return uni

    def segment_polys(self, as_basic=True, shift_back=False, as_3d=False):
        """Call `polys_to_segments`.

        See Also
        --------
        `common_segments`, `od_pairs`, `fr_to_segments` and `to_fr_segments`
        """
        return geom.polys_to_segments(self, as_basic=as_basic,
                                      to_orig=shift_back, as_3d=as_3d)

    def fr_to_segments(self, as_basic=True, shift_back=False, as_3d=False):
        """Return `from-to` points segments for poly* features as 2D array.

        Call `polys_to_segments`.  See `segment_polys`.
        """
        return geom.polys_to_segments(self, as_basic=as_basic,
                                      to_orig=shift_back, as_3d=as_3d)

    def to_fr_segments(self, as_basic=True, shift_back=False, as_3d=False):
        """Return to-from points segments to compare to fr_to_segments."""
        b_vals = self.bits
        # -- Do the concatenation
        to_fr = np.concatenate(
            [np.concatenate((b[1:], b[:-1]), axis=1) for b in b_vals], axis=0)
        return to_fr

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
        Calls `_add_pnts_on_line_` for Geo bits.
        """
        polys = [geom._add_pnts_on_line_(a, spacing) for a in self.bits]
        g, ift, extent = array_IFT(polys)
        return Geo(g, ift, self.K, self.XT, "Densify by distance")

    def densify_by_factor(self, factor=2):
        """Densify poly features by a specified distance.

        Convert multipart to singlepart features during the process.
        Calls `_pnts_on_line_` for Geo bits.
        """
        polys = [geom.densify_by_factor(a, factor) for a in self.bits]
        g, ift, extent = array_IFT(polys)
        return Geo(g, ift, self.K, self.XT, "Densify by distance")

    def densify_by_percent(self, percent=50, shift_back=True):
        """Densify poly features by a percentage for each segment.

        Converts multipart to singlepart features during the process.
        Calls `_percent_along`.
        """
        bits = self.bits
        polys = [geom._add_pnts_on_line_(a, spacing=percent, is_percent=True)
                 for a in bits]
        if shift_back:
            polys = [a + self.LL for a in polys]
        g0, ift, extent = array_IFT(polys)
        return Geo(g0, ift, self.K, self.XT, "Densify by percent")

    def pnt_on_poly(self, by_dist=True, val=1, as_structured=True):
        """Place a point on polyline/polygon by distance or percent."""
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

    def as_arrays(self):
        """Return a quick view as an object arrays. Calls `Geo_to_arrays`."""
        return Geo_to_arrays(self, shift_back=False)

    def as_lists(self):
        """Return a quick view as lists. Calls `Geo_to_lists`."""
        return Geo_to_lists(self, shift_back=False)

    # ----  ---------------------------
    # ---- (6) segments for poly* boundaries
    #
    def segment_pnt_ids(self):
        """Return the segment point IDs."""
        p_ind = self.pnt_indices(as_structured=False)
        w = np.where(np.diff(self.pnt_ids, prepend=p_ind[0, 1]) == 1)[0]
        id_vals = np.split(self.pnt_indices(), w, axis=0)
        ft_ids = [np.concatenate((i[:-1, 0][:, None],
                                  i[1:, 0][:, None]), axis=1)
                  for i in id_vals]
        return ft_ids

    def to_segments(self, ignore_holes=True):
        """Segment poly* structures into (x0, y0, x1, y1) data per shape.

        An object array is returned. Holes removed from polygons by default,
        but multipart shapes are not.

        See Also
        --------
        `npGeo.segment_polys()` and `npg_helpers.polys_to_segments` for other
        output options.
        """
        if self.K not in (1, 2):
            print("Poly* features required.")
            return None
        # -- basic return as ndarray used by common_segments
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
        evaluated to denote common segments or duplicates.

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
        h_0 = uts(fr_to)  # view as structured array to facilitate unique test
        names = h_0.dtype.names
        h_1 = h_0[list(names[2:4] + names[:2])]  # x_to, y_to and x_fr, y_frr
        idx = np.isin(h_0, h_1)
        common = h_0[idx]
        common = uniq_1d(common)  # replace np.unique(common) sorts output
        if shift_back:
            common[:, :2] += self.LL
            common[:, 2:] += self.LL
        return _fill_float_array(common)  # , idx  # stu(common)

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
        uniq02 = uniq_1d(uniq01)  # np.unique(uniq01)
        vals = _fill_float_array(uniq02)
        if shift_back:
            return vals + np.concatenate((self.LL, self.LL))  # , idx0
        return vals  # , idx0  # return stu(uniq01)

    # ----  ---------------------------
    # ---- (7) sort section -----------
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
        """Return geometry sorted by shape area."""
        if self.K == 1:
            print("Polyline class. You cannot sort by area")
            return self
        vals = self.areas(by_shape=True)         # shape area
        idx = np.argsort(vals)                   # sort the areas
        sorted_ids = self.shp_ids[idx]           # use shape IDs not part IDs
        if not ascending:
            sorted_ids = sorted_ids[::-1]
        if just_indices:
            return self.change_indices(sorted_ids)
        sorted_array = self.get_shapes(sorted_ids.tolist())
        return sorted_array

    def sort_by_length(self, ascending=True, just_indices=False):
        """Sort the geometry by ascending or descending order."""
        vals = self.lengths(by_shape=True)
        idx = np.argsort(vals)
        sorted_ids = self.shp_ids[idx]
        if not ascending:
            sorted_ids = sorted_ids[::-1]
        if just_indices:
            return self.change_indices(sorted_ids)
        sorted_array = self.get_shapes(sorted_ids)
        return sorted_array

    def sort_by_extent(self, extent_pnt='LB', key=0, just_indices=False):
        """Sort the geometry using the conditions outlined below."""
        if extent_pnt not in ('LB', 'LT', 'C'):
            print("Extent point is incorrect... read the docs.")
            return None
        if key not in range(0, 8):
            print("Integer value between 0 and 7 inclusive required.")
            return None
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
        sorted_array = self.get_shapes(sorted_ids)
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
            return self.XY[np.lexsort((self.Y, -self.X))]
        if not a and not b:
            return self.XY[np.lexsort((-self.Y, -self.X))]
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
        # --
        ift = self.IFT
        cw = self.CL
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

    # ----  ---------------------------
    # ---- (8) point info section -----
    # -- points: indices, info, find duplicates
    #
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
        """Duplicate points as structured array, or as ndarray with counts."""
        uni, cnts = np.unique(self, return_counts=True, axis=0)
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

    def singleton_pnts(self, as_structured=False):
        """Return unique points, optionally, as a structured array."""
        uni, cnts = np.unique(self, return_counts=True, axis=0)
        uni = uni[cnts == 1]
        if as_structured:
            num = cnts[cnts == 1]
            dt = [('X_', '<f8'), ('Y_', '<f8'), ('Cnts', '<i4')]
            z = np.empty((uni.shape[0],), dtype=dt)
            z['X_'] = uni[:, 0]
            z['Y_'] = uni[:, 1]
            z['Cnts'] = num
            z = repack_fields(z)
            return z
        return uni

    def uniq_pnts(self, as_structured=False):
        """Return unique points, optionally, as a structured array."""
        uni, cnts = np.unique(self, return_counts=True, axis=0)
        if as_structured:
            dt = [('X_', '<f8'), ('Y_', '<f8'), ('Cnts', '<i4')]
            z = np.empty((uni.shape[0],), dtype=dt)
            z['X_'] = uni[:, 0]
            z['Y_'] = uni[:, 1]
            z['Cnts'] = cnts
            z = repack_fields(z)
            return z
        return uni

    def geom_check(self):
        """Run some geometry checks.  See `check_geometry` for details."""
        return check_geometry(self)

    def roll_shapes(self):
        """Run `roll_coords` as a method."""
        return roll_coords(self)

    # ----  ---------------------------
    # ---- (9) print, display Geo
    def prn(self, ids=None):
        """Print all shapes if `ids=None`, otherwise, provide an id list."""
        npg_prn.prn_Geo_shapes(self, ids)

    def prn_arr(self):
        """Print the constituent arrays.  See `npg_prn.prn_arrays`."""
        npg_prn.prn_arrays(self.as_arrays())

    def prn_obj(self, full=False):
        """Print as an object array."""
        npg_prn.prn_as_obj(self, full)

    def svg(self, as_polygon=True):
        """View the Geo array as an svg."""
        return npg_prn._svg(self, as_polygon)


# ---- == End of class definition ==
# ---- Main Functions
# ----  ---------------------------
# ---- (2) Geo from sequences
#  Construct the Geo array from sequences.
#     ndarrays, object arrays, nested lists, lists of arrays etcetera.
#
def roll_coords(self):
    """Roll point coordinates to a new starting position.

    Notes
    -----
    Rolls the coordinates of the Geo array attempting to put the start/end
    points as close to the lower-left of the ring extent as possible.

    `roll_shapes` implements this as a Geo method.
    """
    # --
    def _closest_to_LL_(a, p, sqrd_=False):
        """Return point distance closest to the `lower-left, LL`."""
        diff = a - p[None, :]
        if sqrd_:
            return np.einsum('ij,ij->i', diff, diff)
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    # --
    msg = "\nPolygon Geo array required for `npg.roll_coords`...\n"
    if is_Geo(self) and self.K != 2:
        print(msg)
        return None
    # --
    arrs = []
    extent = self.XT  # aoi_extent().reshape(2, 2)
    SR = self.SR
    for ar in self.bits:
        LL = np.min(ar, axis=0)
        dist = _closest_to_LL_(ar, LL, sqrd_=True)
        num = np.argmin(dist)
        arrs.append(np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0))
    g = np.concatenate(arrs, axis=0)
    g = Geo(g, self.IFT, self.K, extent, "rolled", None)
    g.SR = SR
    return g


def roll_arrays(arrs):
    """Roll point coordinates to a new starting position.

    Parameters
    ----------
    arrs : list of arrays or a single array

    Notes
    -----
    Rolls the coordinates of the Geo array or ndarray to put the start/end
    points as close to the lower-left of the ring extent as possible.

    If a single array is passed, a single array is returned otherwise a list
    of arrays.
    """
    # --
    def _closest_to_LL_(a, p, sqrd_=False):
        """Return point distance closest to the `lower-left, LL`."""
        diff = a - p[None, :]
        if sqrd_:
            return np.einsum('ij,ij->i', diff, diff)
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    # --
    if not isinstance(arrs, (list, tuple)):
        arrs = [arrs]
    out = []
    for ar in arrs:
        LL = np.min(ar, axis=0)
        dist = _closest_to_LL_(ar, LL, sqrd_=False)
        num = np.argmin(dist)
        out.append(np.concatenate((ar[num:-1], ar[:num], [ar[num]]), axis=0))
    if len(out) == 1:
        return out[0]
    return out


def array_IFT(in_arrays, shift_to_origin=False):
    """Produce the Geo array.  Construction information in `npgDocs`."""
    id_too = []
    a_2d = []
    if isinstance(in_arrays, (list, tuple)):  # next line, for single multipart
        in_arrays = np.asarray(in_arrays, dtype='O')  # .squeeze() removed
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
            if len(p) < 3:
                continue  # bust out, only 3 or fewer points
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
    CL = np.zeros_like(ids)
    for i, k in enumerate(id_prt):
        if i in idx:
            CL[i] = 1
    too = np.cumsum(id_too[:, 2])
    frum = np.concatenate(([0], too))
    pnt_nums = np.zeros_like(ids, dtype=np.int32)
    u, i, cnts = np.unique(ids, True, return_counts=True)
    pnt_nums = np.concatenate([np.arange(i) for i in cnts])
    IFT = np.array(list(zip(ids, frum, too, CL, part, pnt_nums)))
    extent = np.array([np.min(a_stack, axis=0), np.max(a_stack, axis=0)])
    if shift_to_origin:
        a_stack = a_stack - extent[0]
    # recheck clockwise values for the array
    return a_stack, IFT, extent


def arrays_to_Geo(in_arrays, kind=2, info=None, to_origin=False):
    r"""Produce a Geo class object from a list/tuple of arrays.

    Parameters
    ----------
    in_arrays : arrays/lists/tuples
        `in_arrays` can be created by adding existing 2D arrays to a list.
        You can also convert poly features to arrays using `poly2arrays`.
    kind : integer
        Points (0), polylines (1) or polygons (2).
    info : text
        Optional descriptive text.
    to_origin : boolean
        True, shifts the data so that the lower left of the data is set to the
        (0, 0) origin.

    Requires
    --------
    `array_IFT`

    Returns
    -------
    A `Geo` class object, based on a 2D np.ndarray (a_2d), and an array of
    indices (IFT) delineating geometry from-to points for each shape and its
    parts.

    Notes
    -----
    `a_2d` will usually be an object array from `array_IFT`.
    It needs to be recast to a `float` array with shape Nx2 to proceed.

    See Also
    --------
    Use `npg_arc_npg.fc_geometry` to produce `Geo` objects directly from
    arcgis pro featureclasses.
    """
    # -- call array_IFT
    a_2d, ift, extent = array_IFT(in_arrays, shift_to_origin=to_origin)
    a_2d = a_2d.astype(np.float64)  # see Notes
    rows, cols = ift.shape
    z0 = np.full((rows, 6), fill_value=-1, dtype=ift.dtype)
    z0[:, :cols] = ift
    # -- do the clockwise check here and correct any assumptions
    # Polygons require a recheck of ring order.
    g = Geo(a_2d, z0, Kind=kind, Extent=extent, Info=info)
    if kind == 2:  # polygon checks for ring orientation
        old_CL = g.CL
        _c = [_bit_area_(i) > 0 for i in g.bits]
        CL_check = np.asarray(_c, dtype='int')
        if not np.all(old_CL == CL_check):
            z0[:, 3] = CL_check
            fix_prt = [np.cumsum(g.CL[g.IDs == i]) for i in g.U]
            z0[:, 4] = np.concatenate(fix_prt)
            w = np.where(z0[:, 3] == 1)[0]
            # note, below changed z0.shape[-1] to z0.shape[0]
            w = np.concatenate((np.where(z0[:, 3] == 1)[0], [z0.shape[0]]))
            dif = w[1:] - w[:-1]
            if len(dif) > 1:  # *** added
                fix_seq = np.concatenate([np.arange(i) for i in dif])
                z0[:len(fix_seq), 5] = fix_seq
            g = Geo(a_2d, z0, Kind=kind, Extent=extent, Info=info)
    elif kind == 1:  # check for closed-loop polylines
        _c = [(i[-1] == i[0]).all() for i in g.bits]
        CL_check = np.asarray(_c, dtype='int')
        g = Geo(a_2d, z0, Kind=kind, Extent=extent, Info=info)
    return g


# ===========================================================================
# ----  ---------------------------
# ---- (3) Geo to arrays/lists
#
def Geo_to_arrays(g, shift_back=True):
    """Convert the Geo arrays back to the original input arrays.

    Parameters
    ----------
    g : Geo array
    shift_back : boolean
        True, return the coordinates to their original coordinate space.

    Returns
    -------
    Most likely an object array of ndarrays (aka, a ragged array).  To shift
    the coordinates back to their original extent, set `shift_back` to True.
    """
    if not hasattr(g, "IFT"):
        print("\nGeo array required...\n")
        return None
    if shift_back:
        g = g.shift(g.LL[0], g.LL[1])
    bits_ = g.bits
    N = np.array([len(bits_)])
    # --
    u, idx = np.unique(g.IDs, True, False, axis=0)
    w0 = np.concatenate((idx, N))
    # frto0 = np.concatenate((w0[:-1, None], w0[1:, None]), axis=1)
    u0, idx0 = np.unique(g.IP, True, False, axis=0)
    w1 = np.concatenate((idx0, N))
    frto1 = np.concatenate((w1[:-1, None], w1[1:, None]), axis=1)
    # --
    arrs = []
    subs = []
    for ft in frto1:
        vals = bits_[ft[0]:ft[1]]
        dt = "O" if len(vals) > 1 or len(subs) > 1 else "float"
        if ft[1] in w0:
            subs.append(np.asarray(vals, dtype=dt).squeeze())
            # dt = "O" if len(subs) > 1 else "float"
            arrs.append(np.asarray(subs, dtype=dt).squeeze())
            subs = []
        else:
            subs.append(np.asarray(vals, dtype=dt).squeeze())
    return np.asarray(arrs, dtype="O")


def Geo_to_lists(g, shift_back=True):
    """Convert the Geo arrays back to the original input arrays.

    Parameters
    ----------
    g : Geo array
    shift_back : boolean
        True, return the coordinates to their original coordinate space.

    Returns
    -------
    Most likely an object array of ndarrays (aka, a ragged array).  To shift
    the coordinates back to their original extent, set `shift_back` to True.
    """
    if not hasattr(g, "IFT"):
        print("\nGeo array required...\n")
        return None
    if shift_back:
        g = g.shift(g.LL[0], g.LL[1])
    bits_ = g.bits
    N = np.array([len(bits_)])
    # --
    u, idx = np.unique(g.IDs, True, False, axis=0)
    w0 = np.concatenate((idx, N))
    # frto0 = np.concatenate((w0[:-1, None], w0[1:, None]), axis=1)
    u0, idx0 = np.unique(g.IP, True, False, axis=0)
    w1 = np.concatenate((idx0, N))
    frto1 = np.concatenate((w1[:-1, None], w1[1:, None]), axis=1)
    # --
    arrs = []
    subs = []
    for ft in frto1:
        vals = bits_[ft[0]:ft[1]]
        if ft[1] in w0:
            v = [i.tolist() for i in vals]
            if len(v) >= 1:
                v = [[tuple(j) for j in i] for i in v]
            subs.extend(v)     # -- or append or extend... keep checking
            arrs.append(subs)  # -- or extend... keep checking
            subs = []
        else:
            v = [i.tolist() for i in vals]
            if len(v) >= 1:
                v = [[tuple(j) for j in i] for i in v]
            subs.append(v)
    return arrs  # np.asarray(arrs, dtype="O")


def _fill_float_array(arr):
    """Fill an array of floats from a structured array of floats as an ndarray.

    This is a simplified version of `stu`.  Used by the `common_segments` and
    `unique_segments` methods.
    """
    names = arr.dtype.names
    n = len(names)
    a_2d = np.zeros((arr.size, n), dtype=np.float64)
    for i, name in enumerate(names):
        a_2d[:, i] = arr[name]
    return a_2d


# ===========================================================================
# ----  ---------------------------
# ---- (4) check/fix functions
#
def remove_seq_dupl(g, asGeo=True):
    """Remove sequential duplicate points in a geo array.

    Parameters
    ----------
    g : Geo array
        The Geo array to check.
    asGeo : boolean
        Determines output type.

    Notes
    -----
    Account for duplicate start/end points in polygons that need to be
    added back into `sub` after removing other duplicates.

    Returns
    -------
    A new Geo array or list of lists, with sequential duplicates removed.
    """

    def dup_idx(sub, poly):
        """Check bits for sequential duplicates."""
        idx = np.unique(sub, return_index=True, axis=0)[1]
        if poly:
            tmp = sorted(idx.tolist())
            tmp.append(tmp[0])
            return sub[tmp]
        return sub[np.sort(idx)]

    chunks = Geo_to_arrays(g)
    poly = True if g.K == 2 else False
    out = []
    for i, c in enumerate(chunks):
        n = len(c.shape)
        if n == 1:
            out2 = []
            for j in c:
                out2.append(dup_idx(j, poly))
            out.append(out2)
        else:
            out.append(dup_idx(c, poly))
    if asGeo:
        return arrays_to_Geo(out, g.K, info='cleaned', to_origin=False)
    ret = [dup_idx(i, poly) for i in g.bits]
    return ret


def check_geometry(g):
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
        """Counterclockwise.  Used by `check_geometry."""
        return 0 if _bit_area_(a) > 0. else 1

    ift = g.IFT
    c0 = g.To[-1]
    c1 = g.shape
    shape_check = c0 == c1[0]
    if not shape_check:
        args = [c0, c1[0], c1, ift]
        print("g.To and g.IFT error\n{} != {}, shape={}\n{}".format(*args))
        return None
    ft = ift[g.CL == 1][:, 1:3]
    check_0 = np.logical_and(g.CL == 0, g.Bit == 0)
    check_1 = [_is_ccw_(g[i: j]) for i, j in ft]
    if np.sum(check_0) > 0:
        c2 = g.IDs[check_0]
        print("\ng.CL and g.Bit mismatch\n{}\n...IFT info\n".format(c2))
        npg_prn.prn_tbl(g.IFT_str)
    elif np.sum(check_1) > 0:
        c3 = g.IDs[check_1]
        print("\nError in ring orientation\n... shapes {}\n".format(c3))
        print("IFT information...\n")
    else:
        print("no errors found")
    return None


def is_Geo(obj, verbose=False):
    """Check the input to see if it is a Geo array.  Used by `roll_coords`.

    Notes
    -----
    More simply, just use `hasattr(obj, "IFT")` since all Geo arrays have
    this attribute.
    """
    if hasattr(obj, "IFT"):
        return True
    if verbose:
        msg = "`{}`, is not a Geo array`. Use `arrays_toGeo` to convert."
        print(msg.format(obj.__class__))
    return False


def reindex_shapes(a, prn=True, start=0, end=-1):
    """Provide a resequence of point ids for polygons.

    Parameters
    ----------
    a : Geo array
    prn : boolean
        True to print the result from `start` to `end` values.
    start : integer
        Start number in the sequence.
    end : integer
        End number in the sequence.  A value of -1, prints to the end.

    Returns
    -------
    None or two arrays depending on `prn`.  The first array is a summary of the
    poly boundaries with the old and unique point ids on a per shape basis.
    The second array is the summary of new and old id values.
    """
    if not hasattr(a, "IFT"):
        print("\nGeo array required...")
        return None
    a_xy = a.XY
    idx = []
    uni = []
    z = np.arange(len(a_xy))
    for i in z:
        w = (a_xy[i] == a_xy[:, None]).all(-1).any(-1)
        w0 = list(z[w])
        idx.append(w0)
        if w0 not in uni:
            uni.append(w0)
    #
    q = []
    for i, val in enumerate(uni):
        q.append([i, val])
    #
    out = np.zeros(len(a_xy), int)
    for i, val in enumerate(idx):
        out[i] = np.min(val)
    #
    u = uniq_1d(out)  # np.unique(out)
    d = np.arange(len(u))
    ud = np.vstack((u, d)).T
    du = ud[::-1]
    #
    new = np.arange(len(out))
    for i in du:
        if i[0] - i[1] != 0:
            new[out == i[0]] = i[1]
        else:
            new[out == i[0]] = i[0]
    #
    sp = a.shp_pnt_ids  # shape point ids for input array
    final = np.concatenate((sp, new[:, None]), axis=1)
    dt = [('Pnt_Id', '<i4'), ('Shp_ID', '<i4'), ('Uniq_Id', '<i4')]
    z = np.zeros((len(a),), dtype=dt)
    z['Pnt_Id'] = final[:, 0]
    z['Shp_ID'] = final[:, 1]
    z['Uniq_Id'] = final[:, 2]
    #
    z0 = np.zeros((len(a.IFT), 3), dtype='int')
    z0[:, 0] = a.IFT[:, 0]
    z0[:, 1] = a.Fr
    z0[0:, 2] = a.To - 1
    if prn:
        msg = """
            Pnt_ID  : point id in the polygon sequences
            Shp_ID  : polygon that the point belongs to
            Uniq_ID : unique points used to construct polygons
            See : new/old summary at the end
            """
        print("{}\n{}".format(dedent(msg), '-' * 24))
        print(("{:>8}" * 3).format(*z.dtype.names))
        if end == -1:
            end = len(z)
        start, end = sorted((abs(start), abs(end)))
        end = min((len(z), end))
        for i in z[start: end]:
            print(("{:>8}" * 3).format(*i))
        #
        print("\nNew   Old")
        for i in q:
            print("{:<3} : {}".format(i[0], i[1]))
        #
        print("\n ID  start / end  points")
        for i in z0:
            print("{:<3} : {:>4} {:>5}".format(*i))
        return None
    return z, q, z0


# ===========================================================================
# ----  ---------------------------
# ---- (5) other functions
#
# -- update the docstrings
#    Geo_hlp, array_IFT_doc, dirr_doc, get_shapes_doc,
#    inner_rings_doc, outer_rings_doc, is_in_doc, bounding_circles_doc,
#    radial_sort_doc, sort_by_extent_doc, shapes_doc, parts_doc
#  use `dirr` for extended documentation
#
def dirr(obj, cols=3, prn=True):
    r"""Return a formatted `dir` listing of an object, module, function."""

    def _sub_(arr, cols):
        """Split subarray by cols."""
        return [arr[i: i + cols] for i in range(0, len(arr), cols)]

    if is_Geo(obj):
        # ('Geo' in str(type(obj))) & (issubclass(obj.__class__, np.ndarray))
        sdo = set(dir(obj))
        sda = set(dir(np.ndarray))
        a = ['... Geo class ...\n']
        a.extend(_sub_(sorted(list(sdo.difference(sda))), cols))
        a.extend(["\n\n... Functions from modules ..."])
        a.extend(["\n... npg_geom ..."])
        a.extend(_sub_(sorted(geom.__all__), cols))
        a.extend(["\n... npg_geom  helpers ..."])
        a.extend(_sub_(sorted(geom.__helpers__), cols))
        a.extend(["\n... npg_helpers ..."])
        a.extend(_sub_(sorted(npg_helpers.__all__), cols))
        a.extend(["\n... npg_helpers helpers ..."])
        a.extend(_sub_(sorted(npg_helpers.__helpers__), cols))
        a.extend(["\n... npg_io ..."])
        a.extend(_sub_(npg_io.__all__, cols))
        a.extend(["\n... npg_prn ..."])
        a.extend(_sub_(sorted(npg_prn.__all__), cols))
        # a.extend(["\n... create ...", "", ""])
        # a.extend(sorted(npg_create.__all__))
    else:
        a = dir(obj)
        a = [a[i: i + cols] for i in range(0, len(a), cols)]
    # w = len(max(a, key=len))  # max([len(i) for i in a])
    w = 30
    # frmt = (("{{!s:<{}}} ".format(w))) * cols
    # csze = len(a) / cols  # split it
    # csze = int(csze) + (csze % 1 > 0)
    #
    if hasattr(obj, '__module__'):
        args = ["-" * 70, obj.__module__, obj.__class__]
    else:
        args = ["-" * 70, type(obj), "npg.dirr..."]
    txt_out = "\n{}\n| dir({}) ...\n|    {}\n-------".format(*args)
    cnt = 0
    for i in a:
        if isinstance(i, (list, tuple)):
            cnt += 1
            txt = "\n  ({:>03.0f})  ".format(cnt)
            frmt = f"{{!s:<{w}}} " * len(i)  # (("{{!s:<{}}} ".format(w)))
            txt += frmt.format(*i)
        else:
            txt = i
        txt_out += txt
    if prn:
        print(txt_out)
        return None
    return txt_out


array_IFT.__doc__ += array_IFT_doc
dirr.__doc__ += dirr_doc
Geo.all_shapes.__doc__ += shapes_doc
Geo.all_parts.__doc__ += parts_doc
Geo.get_shapes.__doc__ += get_shapes_doc
Geo.inner_rings.__doc__ += inner_rings_doc
Geo.outer_rings.__doc__ += outer_rings_doc
Geo.is_in.__doc__ += is_in_doc
Geo.bounding_circles.__doc__ += bounding_circles_doc
Geo.convex_hulls.__doc__ += convex_hulls_doc
Geo.extent_rectangles.__doc__ += extent_rectangles_doc
Geo.od_pairs.__doc__ += od_pairs_doc
Geo.pnt_on_poly.__doc__ += pnt_on_poly_doc
Geo.radial_sort.__doc__ += radial_sort_doc
Geo.sort_by_area.__doc__ += sort_by_area_doc
Geo.sort_by_extent.__doc__ += sort_by_extent_doc

# ---- == __main__ section ==
if __name__ == "__main__":
    """optional location for parameters"""
    # in_fc = r"C:\arcpro_npg\Project_npg\npgeom.gdb\Polygons"
