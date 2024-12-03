# -*- coding: utf-8 -*-
# noqa: D205, D400, F401
r"""
-------------
_npg_samples_
-------------

General description.

----

Script :
  _load_sample_.py

Author :
    `Dan_Patterson
    <https://github.com/Dan-Patterson>`_.

Modified :
    2024-03-29

Purpose
-------
Functions for ...

See Also
--------
None

Notes
-----
None

References
----------
None
"""

import sys
from textwrap import dedent
import numpy as np

# ---- optional imports
from numpy.lib.recfunctions import structured_to_unstructured as stu  # noqa
from numpy.lib.recfunctions import unstructured_to_structured as uts  # noqa

import npg
from npg_io import load_geo

from npg.npg_arc_npg import fc_to_Geo  # get_shapes
from npg.npg_plots import plot_polygons  # noqa

from arcpy.da import FeatureClassToNumPyArray, NumPyArrayToFeatureClass  # noqa
# or ...
# from arcgisscripting import da
# then use
# da.FeatureClassToNumPyArray  # ... etc

# def _object_format(a):
#     """Object array formatter."""
#     fmt = str(a).replace("([", "\n"+" "*7)
#     return fmt

#  --- alter or use below
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}  # "object":  lambda: _object_format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    legacy='1.21',
    formatter=ft
    )
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# __all__ = []

# ------------
# -- large ontario file
# fc00 = r"C:/arcpro_npg/Project_npg/npgeom.gdb/Ontario_bounds"

# fc00 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\Ontario_LCConic"
# SR = "NAD 1983 Statistics Canada Lambert"
# ont = FeatureClassToNumPyArray(
#     fc00, ['OID@', 'SHAPE@X', 'SHAPE@Y'], spatial_reference=SR,
#     explode_to_points=True)
# minX = np.min(ont["SHAPE@X"])
# minY = np.min(ont["SHAPE@Y"])
# Ont = fc_to_Geo(
#     fc00, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")

minX = 300000
minY = 5000000
SR = r'NAD 1983 CSRS MTM  9'

# -- square
aoi = np.array([[0., 0.], [0., 10.], [10., 10.], [10., 0.], [0., 0.]])

# -- sq2
fc0 = r'C:\\arcpro_npg\\Project_npg\\tests.gdb\\sq2'
sq2 = fc_to_Geo(fc0, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info='')
# -- or ...
sq2_pth = r"C:\arcpro_npg\data\npz_npy\sq2.npz"
sq2 = load_geo(sq2_pth, extras=False, prn_info=False)
sq2 = npg.roll_coords(sq2)
b0, b1, b2, b3, b4, b5, b6 = sq2.outer_rings(False)

pth = r"C:\arcpro_npg\data\npz_npy\pl_cl_data.npy"
pl_cl_data = np.load(pth)
oid_ = pl_cl_data['OBJECTID']
shp = pl_cl_data['Shape']
shp = shp - [300000., 5000000.]
dif = np.nonzero(oid_[1:] - oid_[:-1] != 0)[0] + 1
a_ = np.array_split(shp, dif)
ag = npg.arrays_to_Geo(a_, kind=2)  # pl_cl split into its components

# pl_cl_data = npg.npg_io.load_geo(pth, extras=False, prn_info=False)

# -- load geo
# pth = r"C:\arcpro_npg\data\npz_npy\multi.npz"
# multi = npg.npg_io.load_geo(pth, extras=False, prn_info=False)
# multi, arrs, names  = npg.npg_io.load_geo(pth, True, True)
#

# -- pl, cl
fc1 = r"C:\arcpro_npg\Project_npg\TestShapes.gdb\clp"
fc2 = r"C:\arcpro_npg\Project_npg\TestShapes.gdb\poly"
cl_ = fc_to_Geo(fc1, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
pl_ = fc_to_Geo(fc2, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
cl_ = npg.roll_coords(cl_)
pl_ = npg.roll_coords(pl_)
#
fc3 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\edgy1"
fc4 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\e_clip"
edgy1 = fc_to_Geo(fc3, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
edgy1 = npg.roll_coords(edgy1)
eclip = fc_to_Geo(fc4, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
eclip = npg.roll_coords(eclip)

# #
# fc5 = r"C:\arcpro_npg\Project_npg\TestShapes.gdb\d0_"
# d0_ = fc_to_Geo(fc5, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
# d0_ = npg.roll_coords(d0_)
# d0_ = d0_.XY  # ** d0 is good **

laps = np.array([[0., 0.], [0., 6.], [0., 2.], [0., 10.], [0., 11.],
                 [0., 10.], [6., 10.], [2., 10.], [10., 10.],
                 [10., 0.], [0., 0.]])

A = np.array([[1., 0.], [4, 10.], [6., 10.], [9., 0.], [7., 0.], [6., 4.],
              [4., 4.], [3., 0.], [1., 0.]])

B = np.array([[1., 1.], [2., 8.], [3., 9.], [7., 9.], [8., 8.], [8., 6.],
              [5., 5.],
              [8., 4.], [8., 2.], [7., 1.], [1., 1.]])

C = np.array([[0., 0.], [0., 10.], [10., 10.], [10., 8.], [2., 8.], [2., 2.],
              [10., 2.], [10., 0.], [0., 0.]])

D = np.array([
    np.array([[1., 1.], [2., 9.], [7., 9.], [8., 8.], [8., 6.],
              [8., 4.], [7.5, 2.], [6.5, 1.], [1., 1]]),
    np.array([[3., 8.], [2., 2.], [6., 2.], [7., 4.], [7., 6.], [6.5, 8.],
              [3., 8.]])], dtype='O')

E = np.array([[0., 0.], [0., 10.], [10., 10.], [10., 8.], [2., 8.],
              [2., 5.5], [5., 5.5], [5., 4], [2., 4.], [2., 2.],
              [10., 2.], [10., 0.], [0., 0.]])

F = np.array([[0., 0.], [1., 10.], [10., 10.], [9., 8.], [3., 8.], [2., 5.5],
              [5., 5.5], [5., 4.], [3., 4.], [2., 0.], [0., 0.]])

Io = np.array([[0., 0.], [2., 10.], [4., 10.], [2., 0.], [0., 0.]])

K = np.array([[0., 0.], [2., 10.], [4., 10.], [3., 5], [6., 10], [8., 10.],
              [5., 4.], [6., 0.], [4., 0.], [3., 3.], [2., 0.], [0., 0.]])

M = np.array([[0., 0.], [1., 10.], [4., 10.], [5., 9.], [6., 10], [9., 10.],
              [10., 0.], [8., 0.], [7., 7.], [5., 5.], [3., 7.], [2., 0.],
              [0., 0.]])

W = np.array([[1.0, 0.0], [0.0, 10.0], [2.0, 10.0], [3., 3.], [5., 5.],
              [7., 3.], [8., 10.], [10.0, 10.0], [9.0, 0.0],
              [6., 0.], [5., 1.], [4., 0.], [1.0, 0.0]])  # np.flip(E, 1)


d0_ = np.array([[0., 5.], [3., 7.], [5., 10.], [7., 7.], [10., 5.], [7., 3.],
                [5., 0.], [3., 3.], [0., 5.]])


p00 = np.array([[0., 0], [0., 2], [8., 2], [8., 0.], [0., 0]])
p01 = np.array([[0., 0], [0., 2], [8., 2], [1, 1.], [8., 0.], [0., 0]])
p02 = np.array([[0., 0], [7, 1.], [0., 2], [8., 2], [8., 0.], [0., 0]])
p03 = np.array([[0., 0], [0., 2], [2., 2], [4, 4.], [6., 2], [8., 2],
                [8., 0.], [0., 0]])
c00 = np.array([[1., 1.], [2., 2.], [5., 2.], [6., 1.], [1., 1.]])
c01 = np.array([[1., 1.], [3., 3.], [4., 3.], [6., 1.], [1., 1.]])
c02 = np.array([[1., 1.], [3., 3.], [3.5, 1.5], [4., 3.], [6., 1.],
                [1., 1.]])
c03 = np.array([[1., 1.], [2., 2], [3., 3.], [3.5, 1.5], [4., 3.],
                [6., 1.], [1., 1.]])  # extra point on line
c04 = np.array([[1., 1.], [3., 3.], [6., 1.], [1., 1.]])  # triangle


lst = [
       [[1, 2, 3], [4, 5]],
       [[1, 2]],
       [1, 2],
       [[]],
       [[1, 2, 3], [4, 5], [6, 7, 8]],
       [[[1, 2, 3], [4, 5]], [[6, 7, 8]]],
       ]

"""
ont_attr = "C:/arcpro_npg/data/npz_npy/ontario_attrib.npz"
ont_geo = "C:/arcpro_npg/data/npz_npy/ontario_sp.npz"
#  C:/arcpro_npg/Project_npg/npgeom.gdb
ont_bnds = "C:/arcpro_npg/data/npz_npy/Ont_bnds.npz"
geo, arrs, names0 = npg.npg_io.load_geo(ont_geo, extras=True, prn_info=False)
# names0  # ['g', 'ift', 'kind', 'extents', 'spatial_ref']

# bg = npg.npg_io.load_geo(ont_geo, extras=False, prn_info=False)
# bg2 = npg.npg_io.load_geo(ont_bnds, extras=False, prn_info=False)

names, arrs = npg.npg_io.load_geo_attr(ont_attr, prn_info=False)
n0, n1 = names
bg_attr = arrs[n0]
flds = arrs[n1]
"""

# -- reported error for clip
# p1 = [(38.0, 60.6), (37.9, 60.4), (33.7, 56.9), (33.2, 56.5), (32.6, 56.4),
#       (31.2, 56.2), (29.5, 56.2), (27.3, 55.9), (27.1, 55.9), (27.0, 55.9),
#       (26.3, 55.9), (24.7, 56.0), (24.5, 56.0), (24.4, 56.0), (24.2, 56.0),
#       (23.9, 56.1), (24.3, 57.4), (25.8, 61.5), (27.2, 62.9), (27.4, 63.1),
#       (28.7, 63.5), (36.8, 66.0), (37.1, 66.3), (37.3, 66.4), (37.8, 65.4),
#       (37.9, 65.1), (38.0, 62.8), (38.1, 62.1), (38.1, 61.4), (38.0, 60.6)]

# p2 = [(30.2, 58.5), (29.7, 58.2), (28.2, 58.8), (26.7, 59.2), (26.7, 59.5),
#       (27.2, 61.6), (27.5, 63.1), (27.6, 63.2), (27.7, 63.2), (28.8, 63.1),
#       (30.8, 62.6), (31.0, 62.5), (32.5, 61.1), (32.8, 60.3), (32.3, 59.5),
#       (32.0, 59.3), (30.2, 58.5)]

# p1 = np.array(p1)

# p2 = np.array(p2)

# ---- Final main section ----------------------------------------------------
#
if __name__ == "__main__":
    """optional location for parameters"""
    msg = """
    Running... {}
    Producing :
    - `bg`    : ontario, singlepart shapes, 750K points, 580 polygons
    - `bg_attr` : attribute array for the above
    - `multi` : multipart shape with holes
    - `d0_`    : star-like
    - `A`, `B`, `C`, `D`, `E`, `F`, `Io`, `K` : letters
    - `edgy1` : two large singlepart polygons
    - `eclip  : to clip `edgy`
    - `sq`    : four connected shapes, first with a triangular hole.
    - `sq2`   : like `sq` but with 7 shapes, some multipart, some with holes.
    - b0, b1, c0, c1 parts of sq2 used for testing
    """
    print(dedent(msg).format(script))

    # -- load geo
    # pth = r"c:\arcpro_npg\data\npz_npy\ontario_sp.npz"
    # g, arrs, names  = npg.npg_io.load_geo(pth, suppress_extras=False)
    # ---- my shapes

    # #
    # fc0 = r"C:\arcpro_npg\Project_npg\tests.gdb\multi"
    # p_nts = FeatureClassToNumPyArray(
    #     fc0, ["SHAPE@X", "SHAPE@Y"], spatial_reference=SR,
    #     explode_to_points=True)
    # multi = fc_to_Geo(
    #     fc0, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
    #
    # -- save geo
    # f_name =  r"multi.npz"
    # folder = r"c:\arcpro_npg\data\npz_npy"
    # npg.npg_io.save_geo(multi, f_name, folder)
    #
    # -- load geo
    # pth = r"C:\arcpro_npg\data\npz_npy\multi.npz"
    # multi = npg.npg_io.load_geo(pth, extras=False, prn_info=False)
    # multi, arrs, names  = npg.npg_io.load_geo(pth, True, True)
    #

    # #
    # fc2 = r"C:\arcpro_npg\Project_npg\tests.gdb\sq"
    # sq = fc_to_Geo(
    #     fc2, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
    # sq = npg.roll_coords(sq)
    # #
    # fc3 = r"C:\arcpro_npg\Project_npg\tests.gdb\v0"  # Voronoi2
    # v = fc_to_Geo(
    #     fc3, geom_kind=2, minX=300000, minY=5000000, sp_ref=SR, info="")
    # v = npg.roll_coords(v)
    #
    # fc4 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\Polylines"
    # fc5 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\dissolved"
    # #
    j0 = r"C:\arcpro_npg\data\json\multi.geojson"
    # j1 = r"C:\arcpro_npg\data\json\multi.json"
    # j2 = r"C:\arcpro_npg\data\json\sq2.geojson"
    # j3 = r"C:\arcpro_npg\data\json\sq.json"
    # c = np.array([[0, 10.], [11., 13.], [12., 7.], [8., 2], [0., 10]])
    #
    # fc7 = r"C:\arcpro_npg\Project_npg\tests.gdb\hex_shift"
    # fc8 = r"C:\arcpro_npg\Project_npg\tests.gdb\rectangles"
    # h = fc_to_Geo(
    #     fc7, geom_kind=2, minX=300000, minY=5029990, sp_ref=SR, info="")
    # h = npg.roll_coords(h)
    # r = fc_to_Geo(
    #     fc8, geom_kind=2, minX=300000, minY=5029990, sp_ref=SR, info="")
    # r = npg.roll_coords(r)

    # fc0 = r"C:\arcpro_npg\Project_npg\tests.gdb\sq2"
    # sq2 = fc_to_Geo(
    #     fc0, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
    # sq2 = npg.roll_coords(sq2)
    # #
    # cf = r"C:\arcpro_npg\Project_npg\tests.gdb\c"
    # c = fc_to_Geo(
    #     cf, geom_kind=2, minX=300000, minY=5000000, sp_ref=SR, info="")
    # c = npg.roll_coords(c)
    # #
    # # intersection testing shapes
    # #
    # b0, b1, b2, b3, b4, b5, b6 = sq2.outer_rings(False)
    # # b_1 = np.concatenate((b1[:-1], b1[1:]), axis=1).reshape(-1, 2, 2)
    # c0, c1 = c.outer_rings(False)
    # c2 = c1 - (0., 7.5)

    # c_0 = np.concatenate((c0[:-1], c0[1:]), axis=1).reshape(-1, 2, 2)
    #
    # -- rectangles and hexagons
    # -- from npg.npg_plots import plot_polygons

    # fc9 = r"C:\arcpro_npg\Project_npg\tests.gdb\CC"
    # CC = fc_to_Geo(
    #     fc9, geom_kind=2, minX=300000, minY=5000000, sp_ref=SR, info="")
    # CC = CC.XY
    # #

    # A = np.array([[1., 0], [4, 10.], [6, 10], [9, 0], [7, 0], [6, 4], [4, 4],
    #               [3, 0], [1, 0]])
    # B = np.array([[1., 1], [3, 9.], [7, 9], [8, 8], [8, 6], [5, 5], [8, 4],
    #               [8, 2], [7, 1], [1, 1]])
    # # B0 = np.array([[0., 0], [10, 10], [10, 0], [0, 0]])
    # # B = np.array([[0., 0], [5., 5.], [10, 10], [10, 0], [0, 0]])
    # C = np.array([[0., 0], [0, 10.], [10, 10], [10, 8], [2, 8], [2, 2],
    #               [10, 2], [10, 0], [0, 0]])
    # D0 = np.array([[0., 5.], [5., 10.], [10., 5.], [5., 0.], [0., 5.]])
    # D1 = npg.roll_coords(d0)

    # F = np.array([[0., 0], [1, 10.], [10, 10], [9, 8], [3, 8], [2, 5.5],
    #               [5., 5.5], [5., 4], [3, 4], [2, 0], [0, 0]])
    # Io = np.array([[0.0, 0.0], [2., 10.], [4., 10.], [2., 0.0], [0., 0.]])
    # K = np.array([[0.0, 0.0], [2., 10.], [4., 10.], [3., 5], [6., 10],
    #               [8., 10], [5., 4.], [6., 0], [4., 0], [3., 3.], [2., 0.0],
    #               [0., 0.]])
    # a = np.array([[2., 2], [2., 11.], [11., 10.], [11., 2.], [2., 2]])
    # dups = [[[0., 0.], [0., 10.], [0., 10.],
    #          [10., 10.], [10., 0.], [10., 0.],[0., 0.]],
    #         [[0., 0.], [5., 5.], [5., 5.], [5., 0.], [5., 0.], [0., 0.]]]
    # dups = npg.arrays_to_Geo(dups)
    # c_1 = C
    #
    # For concave hull
    # ps = [(207, 184), (393, 60), (197, 158), (197, 114), (128, 261),
    #       (442, 40), (237, 159), (338, 75), (194, 93), (33, 159),
    #       (393, 152), (433, 267), (324, 141), (384, 183), (273, 165),
    #       (250, 257), (423, 198), (227, 68), (120, 184), (214, 49),
    #       (256, 75), (379, 93), (312, 49), (471, 187), (366, 122)]
    # z = np.array(ps) + (300000., 5000000.)
    # dt = np.dtype([('Xs', 'f8'), ('Ys', 'f8')])
    # z0 = z.view(dt).squeeze()

    # # line crosses both outside
    # s00 = np.array([[1., 1], [1., 9.], [9., 9.], [9., 1], [1., 1.]])
    # s01 = np.array([[1., 1], [1., 2], [1., 8.], [1., 9.], [2., 9.],
    #                 [8., 9], [9., 9.], [9., 8.], [9., 2.], [9., 1],
    #                 [8., 1.], [2., 1.], [1., 1.]])
    # s02 = np.array([[1., 2.], [1., 8.], [2., 9.], [8., 9.], [9., 8.],
    #                 [9., 2.], [8., 1.], [2., 1.], [1., 2.]])

    # t00 = np.array([[0., 0], [0., 3], [3., 0.], [0., 0]])
    # t01 = np.array([[0., 7], [0., 10.], [3., 10.], [0., 7.]])
    # t02 = np.array([[7., 10], [10., 10], [10., 7.], [7, 10.]])
    # t03 = np.array([[7., 0.], [10., 3], [10., 0.], [7., 0]])
    # # 2 intersections at points on clipping line
    # t00a = np.array([[0., 0], [0., 3], [1., 2.], [2., 1], [3., 0], [0., 0]])
    # t01a = np.array([[0., 7], [0., 10], [3., 10.],
    #                  [2., 9.], [1., 8.], [0., 7.]])
    # t02a = np.array([[7., 10], [10., 10], [10., 7.],
    #                  [9., 8.], [8., 9.], [7, 10.]])
    # t03a = np.array([[7., 0.], [8., 1.], [9., 2.],
    #                  [10., 3], [10., 0.], [7., 0]])
    # # 1 intersections at closest point to start of clipping line
    # t00b = np.array([[0., 0], [0., 3], [1., 2.], [3., 0.], [0., 0]])
    # t01b = np.array([[0., 7], [0., 10], [3., 10.], [2., 9.], [0., 7.]])
    # t02b = np.array([[7., 10], [10., 10], [10., 7.], [9., 8.], [7, 10.]])
    # t03b = np.array([[7., 0.], [8., 1.], [10., 3], [10., 0.], [7., 0]])
    # # 1 intersections at farthest point to start of clipping line
    # t00c = np.array([[0., 0], [0., 3], [2., 1.], [3., 0.], [0., 0]])
    # t01c = np.array([[0., 7], [0., 10], [3., 10.], [1., 8.], [0., 7.]])
    # t02c = np.array([[7., 10], [10., 10], [10., 7.], [8., 9.], [7, 10.]])
    # t03c = np.array([[7., 0.], [9., 2.], [10., 3], [10., 0.], [7., 0]])
    # # square 1 clip point in
    # t00d = np.array([[0., 0], [0., 3], [3., 3.], [3., 0.], [0., 0]])
    # t01d = np.array([[0., 7], [0., 10.], [3., 10.], [3., 7.], [0., 7.]])
    # t02d = np.array([[7., 7.], [7., 10], [10., 10], [10., 7.], [7, 7.]])
    # t03d = np.array([[7., 0.], [7., 3.], [10., 3], [10., 0.], [7., 0]])
    # # square 1 intersection node first, then second
    # t00e = np.array([[0., 0], [0., 3], [1., 3], [3., 3], [3., 0.], [0., 0]])
    # t01e = np.array([[0., 7], [0., 10.], [3., 10.], [3., 9.],
    #                  [3., 7.], [0., 7.]])
    # t02e = np.array([[7., 7.], [7., 9.], [7., 10], [10., 10],
    #                  [10., 7.], [7, 7.]])
    # t03e = np.array([[7., 0.], [7., 3.], [9., 3.], [10., 3],
    #                  [10., 0.], [7., 0]])
    # # second on
    # t00f = np.array([[0., 0], [0., 3], [3., 3.], [3., 1], [3., 0], [0., 0]])
    # t01f = np.array([[0., 7], [0., 10.], [3., 10.], [3., 7.],
    #                  [1., 7.], [0., 7.]])
    # t02f = np.array([[7., 7.], [7., 10], [10., 10], [10., 7.],
    #                  [9., 7.], [7, 7.]])
    # t03f = np.array([[7., 0.], [7., 3.], [9., 3.], [10., 3],
    #                  [10., 0.], [7., 0]])
    # # both on
    # t00g = np.array([[0., 0], [0., 3], [1., 3.], [3., 3.], [3., 1.],
    #                  [3., 0.], [0., 0]])
    # t01g = np.array([[0., 7], [0., 10.], [3., 10.], [3., 9.],
    #                  [3., 7.], [1., 7.], [0., 7.]])
    # t02g = np.array([[7., 10], [10., 10], [10., 7.], [9., 7.],
    #                  [7., 7.], [7., 9.], [7, 10.]])
    # t03g = np.array([[7., 0.], [7., 1.], [7., 3.], [9., 3.],
    #                  [10., 3], [10., 0.], [7., 0]])
    # # flipped triangles t00, t01, t02, t03
    # t00h = np.array([[0., 0], [0., 3], [3., 0.], [0., 0]])
    # t01 = np.array([[0., 7], [0., 10.], [3., 10.], [0., 7.]])
    # t02 = np.array([[7., 10], [10., 10], [10., 7.], [7, 10.]])
    # t03 = np.array([[7., 0.], [10., 3], [10., 0.], [7., 0]])

    # plot_polygons([s00, t00d, t01d, t02d, t03d])

    # -- lines

    # z0 = np.array([[0., 0], [8., 8.], [2., 2.], [10., 10]])
    # z1 = np.array([[10., 0], [2., 8.], [8., 2.], [0., 10]])
    # z2 = np.array([[2., 0], [2, 7.], [2., 3.], [2., 10]])
    # z3 = np.array([[8., 10], [8, 3.], [8., 7.], [8., 0]])
    # z4 = z1[::-1]
    # z5 = z2[::-1]
