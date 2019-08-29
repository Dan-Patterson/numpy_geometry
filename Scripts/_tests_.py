# -*- coding: utf-8 -*-
"""
=======
_tests_
=======

Script :
    _tests_.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-08-11

Purpose :
    Tests for the Geo class

Notes
-----
Results from _test_ using the following in_fc

>>> in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_LCConic"

Final array shape
>>> tmp.shape    # ---- (1053725, 2), 1,053,725 points
>>> len(shapes)  # ---- 589

%timeit npg.getSR(in_fc)
201 ms ± 2.29 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit npg.fc_shapes(in_fc)
824 ms ± 11.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit npg.poly2array(shapes)
17.5 s ± 189 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit tmp - m
12.3 ms ± 82.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# ---- this is what is used ----
%timeit npg.fc_geometry(in_fc)
18.6 s ± 109 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit [(i - m) for p in poly_arr for i in p]
9.23 ms ± 135 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit npg.Geo(a, IFT, kind, info)
21.9 µs ± 231 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


References:

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
from textwrap import dedent
import numpy as np

import arcpy

import npgeom as npg


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ===========================================================================
# ---- demo
def _test_(in_fc=None, full=False):
    """Demo files listed in __main__ section

    Usage
    -----
    in_fc, g = npg._tests_._test_()
    """
    if in_fc is None:
        in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
    kind = 2
    info = None
    SR = npg.getSR(in_fc)
    shapes = npg.fc_shapes(in_fc)
    # ---- Do the work ----
    poly_arr = npg.poly2array(shapes)
    tmp, IFT = npg.fc_geometry(in_fc)
    m = np.nanmin(tmp, axis=0)
#    m = [300000., 5000000.]
    a = tmp - m
    poly_arr = [(i - m) for p in poly_arr for i in p]
    g = npg.Geo(a, IFT, kind, info)
    frmt = """
    Type :  {}
    IFT  :
    {}
    """
    k_dict = {0: 'Points', 1: 'Polylines/lines', 2: 'Polygons'}
    print(dedent(frmt).format(k_dict[kind], IFT))
#    arr_poly_fc(a, p_type='POLYGON', gdb=gdb, fname='a_test', sr=SR, ids=ids)
    if full:
        return SR, shapes, poly_arr, a, IFT, g
    return in_fc, g


deg = 5.


def _ptypes_(in_fc, SR):
    """Convert polylines/polygons geometry to array

    p0.__geo_interface__['coordinates']

    """
    def _densify_curves_(geom, deg=deg):
        """Densify geometry for circle and ellipse (geom) at ``deg`` degree
        increments. deg, angle = (1, 361), (2, 181), (5, 73).  If it is a
        triangle, return the poly's array
        """
        if 'curve' in geom.JSON:
            return geom.densify('ANGLE', 1, np.deg2rad(deg))
        return geom

    # ----
    def _geom_(geom):
        """ """
        pnts = [[((p.X, p.Y) if p else null_pnt) for p in part]
                for part in geom]
        return pnts
    # ----
    null_pnt = (np.nan, np.nan)
    out = []
    with arcpy.da.SearchCursor(
            in_fc, ('OID@', 'SHAPE@'), None, SR) as cur:
        for i, r in enumerate(cur):
            sub = [r[0], r[1].partCount, r[1].pointCount, _geom_(r[1])]
            out.append(sub)
    return out


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
    # pth = r"C:/Git_Dan/npgeom/data/Polygons.geojson"
    # pth = r"C:/Git_Dan/npgeom/data/Oddities.geojson"
    # pth = r"C:/Git_Dan/npgeom/data/Ontario_LCConic.geojson"
    # in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/sample_10k"
#    testing = True
#    if testing:
#        in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
#        # in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Oddities"
#        # in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_LCConic"
#        returned = _test_(in_fc)
    in_fc, g = _test_()

"""
import npgeom as npg
import arcpy
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_LCConic"
SR = npg.getSR(in_fc)
c = arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR)

prts = [row[1].partCount for row in c]
c.reset()
pnts = [row[1].pointCount for row in c]
c.reset()
np.sum(pnts)   # 1,053,261

%timeit [row[1].partCount for row in c]
The slowest run took 9.50 times longer than the fastest.
This could mean that an intermediate result is being cached.
471 ns ± 585 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit [row[1].pointCount for row in c]
The slowest run took 8.50 times longer than the fastest.
This could mean that an intermediate result is being cached.
429 ns ± 520 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)


c0 = arcpy.da.SearchCursor(in_fc, ('SHAPE@XY'), None, SR)
c0 = arcpy.da.SearchCursor(in_fc, ('SHAPE@XY'), None, SR,
                           explode_to_points=True)
%timeit arcpy.da.SearchCursor(in_fc, ('SHAPE@XY'), None, SR)
5.06 ms ± 316 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

c1 = arcpy.da.SearchCursor(in_fc, ('SHAPE@X', 'SHAPE@Y'), None, SR)
%timeit arcpy.da.SearchCursor(in_fc, ('SHAPE@X', 'SHAPE@Y'), None, SR)
5.93 ms ± 1.12 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

------------------------
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
SR = npg.getSR(in_fc)
c1 = arcpy.da.SearchCursor(in_fc, ('SHAPE@X', 'SHAPE@Y'), None, SR)
g1 = c1._as_narray()
gxy = stu(g1)

----------------------------------
---- small set 12 shapes with 2 multipart shapes
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
SR = npg.getSR(in_fc)

%timeit _ptypes_(in_fc, SR)
8.89 ms ± 298 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
----
---- big set
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_LCConic"
SR = npg.getSR(in_fc)

%timeit _ptypes_(in_fc, SR)
20.9 s ± 198 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

589 shapes
 82 multipart shapes
507 singlepart shapes
1,053,261 points
1,197 parts

---- subset of singlepart shapes from Ontario_LCConic
in_fc = r"C:/Git_Dan/npgeom/npgeom.gdb/Ontario_singlepart"
SR = npg.getSR(in_fc)

%timeit _ptypes_(in_fc, SR)
15 s ± 247 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

507 singlepart shapes
748,874 points
"""
