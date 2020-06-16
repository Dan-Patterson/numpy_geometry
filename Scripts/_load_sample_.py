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
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2020-02-22

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
# from textwrap import dedent
import numpy as np
# import npgeom as npg

if 'npg' not in list(locals().keys()):
    import npgeom as npg
    # from npg.npGeo import (roll_coords, arrays_to_Geo, dirr, _svg)
    # Geo, array_IFT
    from npg_arc_npg import fc_to_Geo  # get_shapes

# ---- optional imports
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
from arcpy.da import FeatureClassToNumPyArray

# noqa: E501
ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.1f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=1, suppress=True, threshold=200,
    formatter=ft
)

script = sys.argv[0]  # print this should you need to locate the script

__all__ = []


# ---- Final main section ----------------------------------------------------
#
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))


    f = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
    SR = "NAD 1983 CSRS MTM  9"
    p_nts = FeatureClassToNumPyArray(f, ["SHAPE@X", "SHAPE@Y"],
                                     spatial_reference=SR)
    g = fc_to_Geo(f, geom_kind=2, minX=300000, minY=5000000, info="")

    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\edgy"
    SR = "NAD 1983 CSRS MTM  9"
    ed = fc_to_Geo(in_fc, geom_kind=2, minX=300000, minY=5000000, info="")
    e0, e1 = ed.bits
    eds = [e0, e1]
    edgy = npg.arrays_to_Geo(eds, 2)  # polygon Geo array

    in_fc1 = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"
    SR = "NAD 1983 CSRS MTM  9"
    sq = fc_to_Geo(in_fc1, geom_kind=2, minX=300000, minY=5000000, info="")
    sq2 = npg.roll_coords(sq)

# ============================================================================
"""
import arcpy
pnts = [arcpy.Point(i[0], i[1]) for i in g_uni]
poly0 = arcpy.Polygon(arcpy.Array(arcpy.Point(i[0], i[1]) for i in e0))
poly1 = arcpy.Polygon(arcpy.Array(arcpy.Point(i[0], i[1]) for i in e1))
# ---- to display, uncomment
# npg._svg(ed)
# poly0
# poly1
# ----
# ---- check the arcpy approach

in_0 = [p.within(poly0) for p in pnts]
n0 = len(in_0)
[pnts[i] for i in range(n0) if in_0[i]]

[<Point (9.0, 13.0, #, #)>,
 <Point (11.0, 12.0, #, #)>,
 <Point (12.0, 14.0, #, #)>,
 <Point (13.0, 12.0, #, #)>]

%%timeit
in_0 = [p.within(poly0) for p in pnts]
n0 = len(in_0)
[pnts[i] for i in range(n0) if in_0[i]]
2.27 ms ± 12.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


in_1 = [p.within(poly1) for p in pnts]
n1 = len(in_1)
[pnts[i] for i in range(n1) if in_1[i]]

[<Point (20.0, 1.0, #, #)>]

%%timeit
in_1 = [p.within(poly1) for p in pnts]
n1 = len(in_1)
[pnts[i] for i in range(n1) if in_1[i]]
1.25 ms ± 40.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


# ---- pure python timing
 g_uni[np.nonzero([w_num(p, e0) for p in g_uni])]

array([[ 9.00,  13.00],
       [ 11.00,  12.00],
       [ 12.00,  14.00],
       [ 13.00,  12.00]])

%timeit g_uni[np.nonzero([w_num(p, e0) for p in g_uni])]
3.58 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

g_uni[np.nonzero([w_num(p, e1) for p in g_uni])]

array([[ 20.00,  1.00],
       [ 21.00,  0.00]])

%timeit g_uni[np.nonzero([w_num(p, e1) for p in g_uni])]
1.48 ms ± 40.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# ---- numpy implementation
%timeit np_wn(g_uni, e0)
116 µs ± 4.03 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit np_wn(g_uni, e1)
62.9 µs ± 2.92 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

np_wn(g_uni, e0)
array([[ 9.00,  13.00],
       [ 11.00,  12.00],
       [ 12.00,  14.00],
       [ 13.00,  12.00]])

np_wn(g_uni, e1)
array([[ 20.00,  1.00],
       [ 21.00,  0.00]])

# ---- inclusion test
n0 = len(in_0)
[pnts[i] for i in range(n0) if in_0[i]]  # 5.03 µs ± 1.27 µs
[<Point (9.0, 13.0, #, #)>, <Point (11.0, 12.0, #, #)>,
 <Point (12.0, 14.0, #, #)>, <Point (13.0, 12.0, #, #)>]

n1 = len(in_1)
[pnts[i] for i in range(n1) if in_1[i]]  # 3.71 µs ± 878 ns
[<Point (20.0, 1.0, #, #)>]

# ---- try a multipoint

mp = arcpy.Multipoint(arcpy.Array(pnts))

%%timeit
mp = arcpy.Multipoint(arcpy.Array(pnts))
int_0 = mp.intersect(poly0, 1)  # return points as the output
1.35 ms ± 35.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%%timeit
mp = arcpy.Multipoint(arcpy.Array(pnts))
int_1 = mp.intersect(poly1, 1)
862 µs ± 42.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

[p for p in int_1]
[<Point (20.0, 1.0, #, #)>, <Point (21.0, 0.0, #, #)>]

[p for p in int_0]
[<Point (9.0, 13.0, #, #)>, <Point (11.0, 12.0, #, #)>,
 <Point (12.0, 14.0, #, #)>, <Point (13.0, 12.0, #, #)>]



Dissolve tests

srted = sub.sort_by_extent(0)
b1, b2, b3, b4, b5 = s_bits = srted.bits
idx = srted.point_indices(as_structured=False)
u, idx_, inv_, cnts = np.unique(srted, True, True, True, axis=0)
w = np.where(cnts>1)[0]

b1b2 = np.isin(b2, b1)
b1b3 = np.isin(b3, b1)

"""


# ---- sq tests

# # from npgeom.npg_plots import plot_polygons
# script = sys.argv[0]
# u, ids, inv, c = np.unique(sq2, True, True, True, axis=0)
# u_s = sq2[sorted(ids)]  # unique values sorted
# dups = u[sorted(inv)]   # sorted list showing all the duplicates, the `c` list
# s_inv = np.sort(inv)    # sort indices
# gt_1 = np.where(c[inv] > 1)[0]  # indices where cnt
# eq_1 = np.where(c[inv] == 1)[0]
# z = sq2[s_inv]  # rearranged showing the duplicates based on sort count
# # ids position of first occurence of a unique value

# s = sq2.pull_shapes([1, 2])  # new geo array with just shapes 1 and 2
# u0, ids0, inv0, c0 = np.unique(s, True, True, True, axis=0)

# b0, b1 = sq2.bits[:2]
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
