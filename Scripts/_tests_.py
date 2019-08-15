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

References:

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
from textwrap import dedent
import numpy as np

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
    tmp, IFT, IFT_2 = npg.fc_geometry(in_fc)
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
        return SR, shapes, poly_arr, a, IFT, IFT_2, g
    return in_fc, g


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
    # pth = r"C:\Git_Dan\npgeom\data\Polygons.geojson"
    # pth = r"C:\Git_Dan\npgeom\data\Oddities.geojson"
    # pth = r"C:\Git_Dan\npgeom\data\Ontario_LCConic.geojson"
    #in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\sample_10k"
#    testing = True
#    if testing:
#        in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Polygons"
#        # in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Oddities"
#        # in_fc = r"C:\Git_Dan\npgeom\npgeom.gdb\Ontario_LCConic"
#        returned = _test_(in_fc)
