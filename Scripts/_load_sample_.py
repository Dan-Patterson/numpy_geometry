# -*- coding: utf-8 -*-
r"""
-------------
_load_sample_
-------------

General description.

----

Script :
  _load_sample_.py

Author :
    Dan_Patterson

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2020-09-06

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
# pycodestyle D205 gets rid of that one blank line thing
# pylint: disable=C0103,C0302,C0415
# pylint: disable=E0402,E0611,E1136,E1121, R0904,R0914,
# pylint: disable=W0201,W0212,W0221,W0612,W0621,W0105
# pylint: disable=R0902

import sys
from textwrap import dedent
import numpy as np

# ---- optional imports
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts

if 'npg' not in list(locals().keys()):
    import npgeom as npg
    from npg_arc_npg import fc_to_Geo  # get_shapes

from arcpy.da import FeatureClassToNumPyArray

# def _object_format(a):
#     """Object array formatter"""
#     fmt = ("{!r:}\n"*len(a))
#     return fmt.format(*a)

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 7.2f}'.format}  # "object":  lambda: _object_format}
#  --- alter or use below

np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft)  # , legacy='1.13')

script = sys.argv[0]  # print this should you need to locate the script

__all__ = []


# ---- Final main section ----------------------------------------------------
#
if __name__ == "__main__":
    """optional location for parameters"""
    msg = """
    Running... {}
    Producing :
    - `big`   : numpy array for the following
    - `bg`    : ontario, singlepart shapes, 750K points, 580 polygons
    - `multi` : multipart shape with holes
    - `edgy`  : two large singlepart polygons
    - `sq`    : four connected shapes, first with a triangular hole.
    - `sq2`   : same as `sq` but with 3 separate shapes with holes.
    """
    print(dedent(msg).format(script))

    fc00 = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Ontario_singlepart"
    SR = "NAD 1983 Statistics Canada Lambert"
    big = FeatureClassToNumPyArray(
        fc00, ['OID@', 'SHAPE@X', 'SHAPE@Y'], spatial_reference=SR,
        explode_to_points=True)
    minX = np.min(big["SHAPE@X"])
    minY = np.min(big["SHAPE@Y"])
    bg = fc_to_Geo(fc00, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")

    # ---- my shapes
    minX = 300000
    minY = 5000000
    SR = "NAD 1983 CSRS MTM  9"
    #
    fc0 = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\multi"
    p_nts = FeatureClassToNumPyArray(
        fc0, ["SHAPE@X", "SHAPE@Y"], spatial_reference=SR,
        explode_to_points=True)
    multi = fc_to_Geo(fc0, geom_kind=2, minX=minX, minY=minY, info="")
    #
    fc1 = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\edgy"
    ed = fc_to_Geo(fc1, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
    edgy = npg.arrays_to_Geo(ed.bits, 2)  # polygon Geo array
    #
    fc2 = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"
    sq = fc_to_Geo(fc2, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
    sq = npg.roll_coords(sq)
    #
    fc3 = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq2"
    sq2 = fc_to_Geo(fc3, geom_kind=2, minX=minX, minY=minY, sp_ref=SR, info="")
    sq2 = npg.roll_coords(sq2)
    fc4 = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polylines"
    fc5 = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\dissolved"
    #
    j0 = r"C:\Git_Dan\npgeom\Project_npg\Data_files\multi.geojson"
    j1 = r"C:\Git_Dan\npgeom\Project_npg\Data_files\multi.json"
    j2 = r"C:\Git_Dan\npgeom\Project_npg\Data_files\sq.geojson"
    j3 = r"C:\Git_Dan\npgeom\Project_npg\Data_files\sq.json"

    c = np.array([[0, 10.], [11., 13.], [12., 7.], [8., 2], [0., 10]])
    # fc0 = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\multi"
    # fc2 = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"

# ============================================================================

# # ---- sq tests

# # from npgeom.npg_plots import plot_polygons
# script = sys.argv[0]
# u, ids, inv, c = np.unique(sq2, True, True, True, axis=0)
# u_s = sq2[sorted(ids)]  # unique values sorted
# dups = u[sorted(inv)]   # sorted list showing all the duplicates, `c` list
# s_inv = np.sort(inv)    # sort indices
# gt_1 = np.where(c[inv] > 1)[0]  # indices where cnt
# eq_1 = np.where(c[inv] == 1)[0]
# z = sq[s_inv]  # rearranged showing the duplicates based on sort count
# # ids position of first occurence of a unique value

# s = sq.get_shapes([1, 2])  # new geo array with just shapes 1 and 2
# u0, ids0, inv0, c0 = np.unique(s, True, True, True, axis=0)

# b0, b1 = sq.bits[:2]
# a = np.concatenate((b0[:-1], b0[1:]), axis=1)
# b = np.concatenate((b1[:-1], b1[1:]), axis=1)
# # ft0 = np.concatenate((b0[:-1], b0[1:]), axis=1)
# # ft1 = np.concatenate((b1[1:], b1[:-1]), axis=1)
# # ft2 = np.concatenate((b1[:-1], b1[1:]), axis=1)
# ab = a[~np.all(np.isin(a, b), axis=1)]
# ba = b[~np.all(np.isin(b, a), axis=1)]

# b0_ = b0[np.all(np.isin(b0, b1), axis=1)]
# # b0[:-1][np.all(np.isin(b0[:-1], b1[:-1]), axis=1)]  # remove dups
# b1_ = b1[np.all(np.isin(b1, b0), axis=1)]
# # b1[:-1][np.all(np.isin(b1[:-1], b0[:-1]), axis=1)]  # remove dups
# range = Reclass(!BP_sqm!)
