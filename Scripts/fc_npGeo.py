# -*- coding: utf-8 -*-
"""
========
fc_npGeo
========

Script : 
    fc_npGeo.py

Author : 
    Dan_Patterson@carleton.ca

Modified : 2019-06-13
    Creation date during 2019 as part of ``arraytools``.

Purpose : Tools for working with poly features as an array class
    Requires npGeo to implement the array geometry class.

See Also : npGeo
    A fuller description of the class, its methods and properties is given
    there.  This script focuses on getting arcpy geometry into numpy arrays.

References
----------
**General**

`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.

**Advanced license tools**

Some of the functions that you can replicate using this data class would
include:

`Feature Envelope to Polygon
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-envelope-to-polygon.htm>`_.

`Feature to Line
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-to-line.htm>`_.

`Feature to Point
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-to-point.htm>`_.

`Feature Vertices to Points
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-vertices-to-points.htm>`_.

`Minimum Bounding Geometry: circle, MABR, Extent Polygon, Convex Hull
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum
-bounding-geometry.htm>`_.

`Polygon to Line
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/polygon
-to-line.htm>`_.

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=E0611  # stifle the arcgisscripting
# pylint: disable=E1101  # ditto for arcpy
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=W0621  # redefining name

import sys
from textwrap import dedent  #, indent
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts

import arcpy
from npGeo import Geo

__all__ = ['FLOATS', 'INTS', 'NUMS',    # constants
           'arcpy', 'np', 'stu',        # imports
           'sys', 'dedent', 'Geo',
           '_check', '_demo_',
           '_make_nulls_', 'getSR',     # featureclass methods
           'fc_data', 'fc_geometry',
           'fc_shapes', 'poly2array',
           'array_poly', 'arrays_Geo',  # array methods
           'geometry_fc',
           'prn_q', 'prn_tbl'           # printing
           ]
# ---- Constants -------------------------------------------------------------
#
script = sys.argv[0]

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS

null_pnt = (np.nan, np.nan)  # ---- a null point

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=100, precision=2, suppress=True,
                    threshold=100, formatter=ft)

# ==== arc geometry =========================================================
# These are the main geometry to array conversions
#
# ---- for polyline/polygon features
#
def poly2array(polys):
    """Convert polyline or polygon shapes to arrays for use in the Geo class.

    Parameters
    ----------
    polys : tuple, list
        Polyline or polygons in a list/tuple
    """
    def _p2p_(poly):
        """Convert a single ``poly`` shape to numpy arrays or object"""
        sub = []
        for arr in poly:
            pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
            sub.append(np.asarray(pnts))
        return sub
    # ----
    if not isinstance(polys, (list, tuple)):
        polys = [polys]
    out = []
    for poly in polys:
        out.append(_p2p_(poly))
    return out

# ---- produce the Geo class object from the output of poly2arrays
#
def arrays_Geo(in_arrays):
    """Produce a Geo class object from a list/tuple of arrays.

    Parameters
    ----------
    in_arrays : list
        ``in_arrays`` can be created by adding existing 2D arrays to the list
         or produced from the conversion of poly features to arrays using
        ``poly2arrays``.
    Kind : integer
        MultiPoints (0), polylines (1) or polygons (2)

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
        if len(p) == 1:
            id_too.append([i, len(p[0])])
            a_2d.extend(p[0])
        else:
            id_too.extend([[i, len(k)] for k in p])
            a_2d.append([j for i in p for j in i])
    a_2d = np.vstack(a_2d)
    id_too = np.array(id_too)
    I = id_too[:, 0]
    too = np.cumsum(id_too[:, 1])
    frum = np.concatenate(([0], too))
    IFT = np.array(list(zip(I, frum, too)))
    return a_2d, IFT

# ===========================================================================
# ---- featureclass section, arcpy dependent via arcgisscripting
#
def _make_nulls_(in_fc, int_null=-999):
    """Return null values for a list of fields objects, excluding objectid
    and geometry related fields.  Throw in whatever else you want.

    Parameters
    ----------
    in_flds : list of arcpy field objects
        Use arcpy.ListFields to get a list of featureclass fields.
    int_null : integer
        A default to use for integer nulls since there is no ``nan`` equivalent
        Other options include

    >>> np.iinfo(np.int32).min # -2147483648
    >>> np.iinfo(np.int16).min # -32768
    >>> np.iinfo(np.int8).min  # -128

    >>> [i for i in cur.__iter__()]
    >>> [[j if j else -999 for j in i] for i in cur.__iter__() ]
    """
    nulls = {'Double': np.nan, 'Single': np.nan, 'Float': np.nan,
             'Short': int_null, 'SmallInteger': int_null, 'Long': int_null,
             'Integer': int_null, 'String':str(None), 'Text':str(None),
             'Date': np.datetime64('NaT')}
    #
    desc = arcpy.da.Describe(in_fc)
    if desc['dataType'] != 'FeatureClass':
        print("Only Featureclasses are supported")
        return None, None
    in_flds = desc['fields']
    shp = desc['shapeFieldName']
    good = [f for f in in_flds if f.editable and f.name != shp]
    fld_dict = {f.name: f.type for f in good}
    fld_names = list(fld_dict.keys())
    null_dict = {f: nulls[fld_dict[f]] for f in fld_names}
    # ---- insert the OBJECTID field
    return null_dict, fld_names


def getSR(in_fc, verbose=False):
    """Return the spatial reference of a featureclass"""
    desc = arcpy.da.Describe(in_fc)
    SR = desc['spatialReference']
    if verbose:
        print("SR name: {}  factory code: {}".format(SR.name, SR.factoryCode))
    return SR


def fc_composition(in_fc, SR=None):
    """Featureclass geometry composition in terms of shapes, shape parts, and
    point counts for each part.
    """
    if SR is None:
        SR = getSR(in_fc)    
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@', spatial_reference=SR) as cur:
        len_lst = []
        for p_id, row in enumerate(cur):
            p = row[0]
            parts = p.partCount
            num_pnts = np.asarray([p[i].count for i in range(parts)])
            IDs = np.repeat(p_id, parts)
            part_count = np.arange(parts)
            too = np.cumsum(num_pnts)
            result = np.stack((IDs, part_count, num_pnts, too), axis=-1)
            len_lst.append(result)
    fc_comp = np.vstack(len_lst)
    return fc_comp

# ---- Used to create the inputs for the Geo class
#
def fc_geometry(in_fc, SR=None):
    """Derive, arcpy geometry objects from a featureClass searchcursor.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass.  Points not supported.
    SR : spatial reference
       Spatial reference object, name or id

    Returns
    -------
    ``a_2d, IFT`` (ids_from_to), where ``a_2d`` are the points as a 2D array,
    ``IFT``represent the id numbers (which are repeated for multipart shapes),
    and the from-to pairs of the feature parts

    See Also
    --------
    Use **arrays_Geo** to produce ``Geo`` objects directly pre-existing arrays,
     or arrays derived form existing arcpy poly objects which originated from
     esri featureclasses.

    Notes
    -----
    Multipoint, polylines and polygons and its variants are supported.

    **Point and Multipoint featureclasses**

    >>> cent = arcpy.da.FeatureClassToNumPyArray(pnt_fc,
                                             ['OID@', 'SHAPE@X', 'SHAPE@Y'])

    For multipoints, use

    >>> allpnts = arcpy.da.FeatureClassToNumPyArray(multipnt_fc,
                                                ['OID@', 'SHAPE@X', 'SHAPE@Y']
                                                explode_to_points=True)

    **IFT array structure**

    To see the ``IFT`` output as a structured array, use the following.

    >>> dt = np.dtype({'names': ['ID', 'From', 'To'], 'formats': ['<i4']*3})
    >>> z = IFT.view(dtype=dt).squeeze()
    >>> prn_tbl(z)  To see the output in tabular form

    **Flatten geometry tests**

    >>> %timeit fc_geometry(in_fc2, SR)
    105 ms ± 1.04 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    ...
    >>> %%timeit
    ... cur = arcpy.da.SearchCursor(in_fc2, 'SHAPE@', None, SR)
    ... p = [row[0] for row in cur]
    ... sh = [[i for i in itertools.chain.from_iterable(shp)] for shp in p]
    ... pnts = [[[pt.X, pt.Y] if pt else null_pnt for pt in lst] for lst in sh]
    4.4 ms ± 21.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    msg = """
    Use arcpy.FeatureClassToNumPyArray for Point files.
    MultiPoint, Polyline and Polygons and its variants are supported.
    """
    # ----
    def _multipnt_(in_fc, SR):
        """Convert multipoint geometry to array"""
        pnts = arcpy.da.FeatureClassToNumPyArray(in_fc,
                   ['OID@', 'SHAPE@X', 'SHAPE@Y'],
                   spatial_reference=SR,
                   explode_to_points=True
                   )
        id_len = np.vstack(np.unique(pnts['OID@'], return_counts=True)).T
        a_2d = stu(pnts[['SHAPE@X', 'SHAPE@Y']])  # ---- use ``stu`` to convert
        return id_len, a_2d
    # ----
    def _polytypes_(in_fc, SR):
        """Convert polylines/polygons geomeetry to array"""
        null_pnt = (np.nan, np.nan)
        id_len = []
        a_2d = []
        with arcpy.da.SearchCursor(in_fc, 'SHAPE@', None, SR) as cursor:
            for p_id, row in enumerate(cursor):
                sub = []
                IDs =[]
                num_pnts = []
                parts = row[0].partCount
                for arr in row[0]:                    
                    pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
                    sub.append(np.asarray(pnts))
                    IDs.append(p_id)                   
                    num_pnts.append(len(pnts))
                part_count = np.arange(parts)
                #too = np.cumsum(num_pnts)
                result = np.stack((IDs, part_count, num_pnts), axis=-1)
                id_len.append(result)
                a_2d.extend([j for i in sub for j in i])
        # ----
        id_len = np.vstack(id_len)  #np.array(id_len)
        a_2d = np.asarray(a_2d)
        return id_len, a_2d
    #
    # ---- Check and process section ----------------------------------------
    desc = arcpy.da.Describe(in_fc)
    fc_kind = desc['shapeType']
    SR = desc['spatialReference']
    if fc_kind == "Point":
        print(dedent(msg))
        return None
    elif fc_kind == "Multipoint":
        id_len, a_2d = _multipnt_(in_fc, SR)
    else:
        id_len, a_2d = _polytypes_(in_fc, SR)
    # ---- Return and send out
    ids = id_len[:, 0]
    too = np.cumsum(id_len[:, 2])
    frum = np.concatenate(([0], too))
    from_to = np.array(list(zip(frum, too)))
    IFT = np.c_[ids, from_to]
    id_len2 = np.hstack((id_len, IFT[:, 1:]))
    dt = np.dtype({'names':['IDs', 'Part', 'Points', 'From_ID', 'To_ID'],
                   'formats': ['i4', 'i4','i4','i4','i4']})
    IFT_2 = uts(id_len2, dtype=dt)
    return a_2d, IFT, IFT_2


def fc_g2(in_fc, SR=None):
    """variant"""
    desc = arcpy.da.Describe(in_fc)
    SR = desc['spatialReference']
    null_pnt = (np.nan, np.nan)
    id_len = []
    a_2d = []
    with arcpy.da.SearchCursor(in_fc, ['OID@', 'SHAPE@'], None, SR) as cursor:
        for row in cursor:
            sub = []
            for arr in row[1]:
                pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
                sub.append(np.asarray(pnts))
                id_len.extend([(row[0], len(pnts))])
            a_2d.extend([j for i in sub for j in i])
    # ----
    id_len = np.array(id_len)
    a_2d = np.asarray(a_2d)
    ids = id_len[:, 0]
    too = np.cumsum(id_len[:, 1])
    frum = np.concatenate(([0], too))
    from_to = np.array(list(zip(frum, too)))
    IFT = np.c_[ids, from_to] # np.array(list(zip(ids, frum, too)))
    return a_2d, IFT

def fc_data(in_fc):
    """Pull all editable attributes from a featureclass tables.  During the
    process, <null> values are changed to an appropriate type.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass

    Notes
    -----
    The output objectid and geometry fields are renamed to
    `OID_`, `X_cent`, `Y_cent`, where the latter two are the centroid values.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    null_dict, fld_names = _make_nulls_(in_fc, int_null=-999)
    fld_names = flds + fld_names
    new_names = ['OID_', 'X_cent', 'Y_cent']
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, fld_names,
                                          skip_nulls=False,
                                          null_value=null_dict)
    a.dtype.names = new_names + fld_names[3:]
    return np.asarray(a)

# ===========================================================================
# ---- back to featureclass
#
def array_poly(a, p_type=None, sr=None, IFT=None):
    """
    Used by ``geometry_fc`` to assemble the poly features from array(s).
    This can be used separately.

    Parameters
    ----------
    a : array
        Points array
    p_type : text
        Polygon or Polyline
    sr : spatial reference
        Spatial reference object, name or id
    IFT : array
        An Nx3 array consisting of I(d)F(rom)T(o) points

    Notes
    -----
    Polyline or polygon features can be created from the array data.  The
    features can be multipart with or without interior rings.

    Outer rings are ordered clockwise, inner rings
    (holes) are ordered counterclockwise.  For polylines, there is no concept
    of order. Splitting is modelled after _nan_split_(arr)
    """
    def _arr_poly_(arr, SR, as_type):
        """Slices the array where nan values appear, splitting them off during
        the process.
        """
        subs = []
        s = np.isnan(arr[:, 0])
        if np.any(s):
            w = np.where(s)[0]
            ss = np.split(arr, w)
            subs = [ss[0]]
            subs.extend(i[1:] for i in ss[1:])
        else:
            subs.append(arr)
        aa = []
        for sub in subs:
            aa.append([arcpy.Point(*pairs) for pairs in sub])
        if as_type == 'POLYGON':
            poly = arcpy.Polygon(arcpy.Array(aa), SR)
        elif as_type == 'POLYLINE':
            poly = arcpy.Polyline(arcpy.Array(aa), SR)
        return poly
    # ----
    if not np.all([check is None for check in [p_type, sr, IFT]]):
        msg = """
        Missing or incorrect parameters...
        {}"""
        print(dedent(msg).format(dedent(array_poly.__doc__)))
    ids = IFT[:, 0]
    from_to = IFT[:, 1:]
    chunks = [a[f:t] for f, t in from_to]  # ---- _poly_pieces_ chunks input
    polys = []
    for i in chunks:
        p = _arr_poly_(i, sr, p_type)  # ---- _arr_poly_ makes parts of chunks
        polys.append(p)
    out = list(zip(polys, ids))
    return out


def geometry_fc(a, IFT, p='POLYGON', gdb=None, fname=None, sr=None):
    """Reform poly features from the list of arrays created by ``fc_geometry``.

    Parameters
    ----------
    a : array or list of arrays
        Some can be object arrays, normally created by ``pnts_arr``
    ids : list/array
        Identifies which feature each input belongs to.  This enables one to
        account for multipart shapes
    from_to : list/array
        See ids above, denotes the actual splice elements for each feature.
    p : string
        Uppercase geometry type
    gdb : text
        Geodatabase name
    fname : text
        Featureclass name
    sr : spatial reference
        name or object

    Returns
    -------
    Singlepart and multipart featureclasses.

    Notes
    -----
    The work is done by ``array_poly``.
    """
    out = array_poly(a, p, sr, IFT)   # call array_poly and ist sub
    name = gdb + "\\" + fname
    wkspace = arcpy.env.workspace = 'memory'  # legacy is in_memory
    arcpy.management.CreateFeatureclass(wkspace, fname, p,
                                        spatial_reference=sr)
    arcpy.management.AddField(fname, 'ID_arr', 'LONG')
    with arcpy.da.InsertCursor(fname, ['SHAPE@', 'ID_arr']) as cur:
        for row in out:
            cur.insertRow(row)
    out_fname = fname + "_mp"
    arcpy.management.Dissolve(fname, out_fname, "ID_arr",
                              multi_part="MULTI_PART",
                              unsplit_lines="DISSOLVE_LINES")
    arcpy.management.CopyFeatures(out_fname, name)

#
# ============================================================================
# ---- array dependent
def prn_q(a, edges=3, max_lines=25, width=120, decimals=2):
    """Format a structured array by setting the width so it hopefully wraps.
    """
    width = min(len(str(a[0])), width)
    with np.printoptions(edgeitems=edges, threshold=max_lines, linewidth=width,
                         precision=decimals, suppress=True, nanstr='-n-'):
        print("\nArray fields/values...:")
        print("  ".join([n for n in a.dtype.names]))
        print(a)

#
# ---- printing based on arraytools.frmts.py using prn_rec and dependencies
#
def _check(a):
    """Check dtype and max value for formatting information"""
    return a.shape, a.ndim, a.dtype.kind, np.nanmin(a), np.nanmax(a)


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
            elif c_kind in FLOATS:  # and np.isscalar(c[0]):  # float rounded
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


# ---- from fc_geo.py -------------------------------------------------------
#
def fc_shapes(in_fc, SR=None):
    """Featureclass to arcpy shapes.  Returns polygon, polyline, multipoint,
    or points.
    """
    if SR is None:
        SR = getSR(in_fc)
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@', spatial_reference=SR) as cur:
        out = [row[0] for row in cur]
    return out

# ===========================================================================
# ---- demo
def _demo_(in_fc, kind, info=None):
    """Demo files listed in __main__ section"""
    SR = getSR(in_fc)
    shapes = fc_shapes(in_fc)
    # ---- Do the work ----
    tmp, IFT, IFT_2 = fc_geometry(in_fc)
    m = np.nanmin(tmp, axis=0)
#    m = [300000., 5000000.]
    a = tmp  - m
    g = Geo(a, IFT, kind, info)
    frmt = """
    Type :  {}
    ids-from-to:
    {}
    """
    k_dict = {0:'Points', 1:'Polylines/lines', 2:'Polygons'}
    print(dedent(frmt).format(k_dict[kind], IFT))
    #arr_poly_fc(a1, p_type='POLYGON', gdb=gdb, fname='a1_test', sr=SR, ids=ids)
    return SR, shapes, IFT, IFT_2, g


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""

#    # All polygon shapes
#    in_fc0 = r"C:/Arc_projects/CoordGeom/CoordGeom.gdb/Polygons"
#    SR, sh0, IFT0, s0 = _demo_(in_fc0, 2, 's0')
#    # Single multipart polygon shape
#    in_fc1 = r"C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape1"
#    SR1, sh1, IFT1, s1 = _demo_(in_fc1, 2, False)  # multipart
#    # Above plus one shape to the right
    in_fc2 = r"C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2"
#    in_fc2 = r"C:/Arc_projects/CoordGeom/CoordGeom.gdb/Shape2_multipnts"
    SR, sh2, IFT2, IFT_2, s2 = _demo_(in_fc2, 2, 's2')
#    # Ontario large file
    in_fc = r"C:\Arc_projects\Canada\Canada.gdb\Ontario_LCConic"
#    SR, sh, IFT, s = _demo_(in_fc, 2, 's')
    #
    # ---- Get the shapes that you want by changing s0
    #shps = [s0.get(i) for i in range(5)]
