# -*- coding: utf-8 -*-
r"""
-------------------------------------------
  npg_arc: Functions that require `arcpy`
-------------------------------------------

The functions include those related to featureclasses, their properties and
conversion to Geo arrays and other data formats.

----

Script :
    npg_arc.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-12-30

Purpose
-------
Working with numpy and arcpy.

Notes
-----
See `_npgeom_notes_.py` for extra notes.

References
----------
None (yet).
"""
# pylint: disable=C0103, C0302, C0415, E0611, E1136, E1121, R0904, R0914,
# pylint: disable=W0201, W0212, W0221, W0612, W0621, W0105
# pylint: disable=R0902

# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import copy
from textwrap import dedent, indent

import numpy as np

from numpy.lib.recfunctions import repack_fields
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts

if 'npg' not in list(locals().keys()):
    import npgeom as npg

import json
from arcpy import (
    Array, Exists, ListFields, Multipoint, Point, Polygon, Polyline
)

from arcpy.da import (
    Describe, InsertCursor, SearchCursor, FeatureClassToNumPyArray,
    TableToNumPyArray
)  # ExtendTable, NumPyArrayToTable,  UpdateCursor

# from arcpy.geoprocessing import gp
from arcpy.geoprocessing import env
from arcpy.management import (
    AddField, CopyFeatures, CreateFeatureclass, Delete
)  # DeleteFeatures


script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    '_area_part_', '_cw_', 'poly_shape_finder',
    'get_SR', 'get_shapes', 'get_geo_interface', 'get_shape_K',
    'get_fc_composition', 'get_fc_field_properties',
    'get_shape_properties',
    'fc_nparray_Geo', 'id_fr_to',            # option 1, use this ***
    'Geo_to_fc', 'Geo_to_poly', 'view_poly'     # back to fc
    'make_nulls', 'fc_data', 'tbl_data',     # get attributes
    'fc_geo_Geo', 'flat',                    # option 2
    'fc_gi_Geo',
    'fc_sc_Geo', 'fc_parser', 'ift_maker',   # option 3
    'fc2na', 'array_poly', 'geometry_fc',
    'poly2array', '_flat',                   # geometry to array
    '_del_none', '_poly_arr_',               # extras
    '__geo_interface__'
]


# ---- (1) General helpers ---------------------------------------------------
# ---- clockwise checker
def _area_part_(a):
    """Mini e_area, used by areas and centroids."""
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.sum((e0 - e1)*0.5)  # can use nansum too


def _cw_(a):
    """Clockwise test."""
    return 1 if _area_part_(a) > 0. else 0


def poly_shape_finder(polys):  # -> array
    """Determine the structure of arcobject geometry objects.

    Information is returned with respect to parts and points etcetera.

    Parameters
    ----------
    polys : arcobject object
        This can be a list or a single object.

    See Also
    --------
    ``get_geo_interface(in_fc, SR=None)``  produces the array to use

    Notes
    -----
    np.diff(np.cumsum([i is None for i in z]), prepend=0)
    """
    def psf(poly, cnt):
        """Polgon shape finder.

        Find out where `None` if any is located (`w`) and subtract its
        length from the `arr` length.  Adjust the `w` values accordingly
        to derive the endpoints of each bit.
        """
        out = []
        for i, arr in enumerate(poly):
            if hasattr(arr, '__len__'):
                w = np.where([p is None for p in arr])[0].tolist()
                out.append([cnt, i, len(arr) - len(w), w - np.arange(len(w))])
        return out
    # ----
    main = []
    cnt = 0
    if hasattr(polys, '__type_string__'):
        polys = [polys]
    for poly in polys:
        main.append(psf(poly, cnt))
        cnt += 1
    return main


# ============================================================================
# ---- (2) helpers, using featureclasses, arcpy dependent --------------------
# arcpy dependent via arcgisscripting
#
# Spatial reference object
def get_SR(in_fc, verbose=False):
    """Return the spatial reference of a featureclass."""
    desc = Describe(in_fc)
    SR = desc['spatialReference']
    if verbose:
        print("SR name: {}  factory code: {}".format(SR.name, SR.factoryCode))
    return SR


# Featureclass to shapes, using a searchcursor
def get_shapes(in_fc, SR=None):
    """Return arcpy shapes from a featureclass.

    Returns polygon, polyline, multipoint, or points.
    """
    if SR is None:
        SR = get_SR(in_fc)
    with SearchCursor(in_fc, "SHAPE@", spatial_reference=SR) as cur:
        out = [row[0] for row in cur]
    return out


# Featureclass to geo_interface
def get_geo_interface(in_fc, SR=None):
    """Featureclass to arcpy shapes.

    Returns polygon, polyline, multipoint, or points.

    See Also
    --------
    npg.shape_finder(arr)  to derive the structure of ``out``.
    """
    if SR is None:
        SR = get_SR(in_fc)
    with SearchCursor(in_fc, "SHAPE@", spatial_reference=SR) as cur:
        out = []
        for i, r in enumerate(cur):
            shp = r[0]
            out.append(shp.__geo_interface__['coordinates'])
    return out


# geometry shape class to Geo shape class
def get_shape_K(in_fc):
    """Return shape type for a featureclass.  Returns (kind, k)"""
    desc = Describe(in_fc)
    kind = desc['shapeType']
    if kind in ('Polygon', 'PolygonM', 'PolygonZ'):
        return (kind, 2)
    if kind in ('Polyline', 'PolylineM', 'PolylineZ'):
        return (kind, 1)
    if kind in ('Point', 'Multipoint'):
        return (kind, 0)


# Featureclass composition
def get_fc_composition(in_fc, SR=None, prn=True, start=0, end=50):
    """Return featureclass geometry composition.

    The information returned is includes the shapes, shape parts, and
    point counts for each part.
    """
    if SR is None:
        SR = get_SR(in_fc)
    with SearchCursor(in_fc, ['OID@', 'SHAPE@'], spatial_reference=SR) as cur:
        len_lst = []
        for _, row in enumerate(cur):
            p_id = row[0]
            p = row[1]
            parts = p.partCount
            num_pnts = np.asarray([p[i].count for i in range(parts)])
            IDs = np.repeat(p_id, parts)
            part_count = np.arange(parts)
            too = np.cumsum(num_pnts)
            result = np.stack((IDs, part_count, num_pnts, too), axis=-1)
            len_lst.append(result)
    tmp = np.concatenate(len_lst, axis=0)  # np.vstack(len_lst)
    too = np.cumsum(tmp[:, 2])
    frum = np.concatenate(([0], too))
    frum_too = np.array(list(zip(frum, too)))
    fc_comp = np.hstack((tmp[:, :3], frum_too))  # axis=0)
    dt = np.dtype({'names': ['IDs', 'Part', 'Points', 'From_pnt', 'To_pnt'],
                   'formats': ['i4', 'i4', 'i4', 'i4', 'i4']})
    fc = uts(fc_comp, dtype=dt)
    frmt = "\nFeatureclass...  {}" + \
        "\nShapes :{:>5.0f}\nParts  :{:>5.0f}\n  max  :{:>5.0f}" + \
        "\nPoints :{:>5.0f}\n  min  :{:>5.0f}\n  med  :{:>5.0f}" + \
        "\n  max  :{:>5.0f}"
    if prn:  # ':>{}.0f
        uni, cnts = np.unique(fc['IDs'], return_counts=True)
        a0, a1 = [fc['Part'] + 1, fc['Points']]
        args = [in_fc, len(uni), np.sum(cnts), np.max(a0),
                np.sum(a1), np.min(a1), int(np.median(a1)), np.max(a1)]
        msg = dedent(frmt).format(*args)
        print(msg)
        # ---- to structured and print
        frmt = "{:>8} "*5
        start, end = sorted([abs(int(i)) if isinstance(i, (int, float))
                             else 0 for i in [start, end]])
        end = min([fc.shape[0], end])
        print(frmt.format(*fc.dtype.names))
        for i in range(start, end):
            print(frmt.format(*fc[i]))
        return None
    return fc


# fields information
def get_fc_field_properties(in_fc, verbose=True):
    """Return field properties for featureclasses."""
    flds = ListFields(in_fc)   # arcpy.ListFields
    names = [f.name for f in flds]
    types = [f.type for f in flds]
    edit_ = [str(f.editable) for f in flds]
    req_ = [str(f.required) for f in flds]
    headers = ['Names', 'Types', 'Editable', 'Required']
    out = np.empty((len(names),), dtype=[("", "U12")]*len(headers))
    out.dtype.names = headers
    data = [names, types, edit_, req_]
    for i, j in enumerate(headers):
        out[j] = data[i]
    if verbose:
        npg.prn_tbl(out)
    else:
        return out


def get_shape_properties(a_shape, prn=True):
    """Get some basic shape geometry properties"""
    coords = a_shape.__geo_interface__['coordinates']
    sr = a_shape.spatialReference
    props = ['type', 'isMultipart', 'partCount', 'pointCount', 'area',
             'length', 'length3D', 'centroid', 'trueCentroid', 'firstPoint',
             'lastPoint', 'labelPoint']
    props2 = [['Name', sr.name], ['Factory code', sr.factoryCode]]
    t = "\n".join(["{!s:<12}: {}".format(i, a_shape.__getattribute__(i))
                   for i in props])
    t = t + "\n" + "\n".join(["{!s:<12}: {}".format(*i) for i in props2])
    tc = '{!r:}'.format(np.array(coords))
    tt = t + "\nCoordinates\n" + indent(tc, '....')
    if prn:
        print(tt)
    else:
        return tt


# ============================================================================
# ---- (3a) fc_nparray_Geo section
# ---- ** use this one **
# fc -> nparray -> Geo  uses FeatureClassToNumpyArray ---------------
#
# -- main function --
def fc_nparray_Geo(in_fc, geom_kind=2, info=""):
    """Convert a FeatureClassToNumPyArray to a Geo array.

    This works with the geometry only.  Skip the attributes for later.  The
    processing requirements are listed below.  Just copy and paste.

    Parameters
    ----------
    in_fc : featureclass
        Featureclass in a file geodatabase.
    geom_kind : integer
        Points (0), Polylines (1) and Polygons (2)

    Notes
    -----
    >>> arr = FeatureClassToNumPyArray(
    ...         in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'], explode_to_points=True
    ...         )
    """
    def _area_part_(a):
        """Mini e_area, used by areas and centroids"""
        x0, y1 = (a.T)[:, 1:]
        x1, y0 = (a.T)[:, :-1]
        e0 = np.einsum('...i,...i->...i', x0, y0)
        e1 = np.einsum('...i,...i->...i', x1, y1)
        return np.sum((e0 - e1)*0.5)

    def cw(a):
        """Clockwise check."""
        return 1 if _area_part_(a) > 0. else 0

    def cw2(b):
        """Clockwise check."""
        return 0 if np.sum(np.cross(b[:-1], b[1:])) > 0. else 1
    # ---- (1) Foundational steps
    # Create the array, extract the object id values.
    # To avoid floating point issues, extract the coordinates, round them to a
    # finite precision and shift them to the x-y origin
    #
    kind = geom_kind
    sp_ref = get_SR(in_fc)
    a = FeatureClassToNumPyArray(
        in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'],
        spatial_reference=sp_ref, explode_to_points=True
    )
    oids = a['OID@']
    xy = a[['SHAPE@X', 'SHAPE@Y']]
    mn = [np.min(xy['SHAPE@X']), np.min(xy['SHAPE@Y'])]
    mx = [np.max(xy['SHAPE@X']), np.max(xy['SHAPE@Y'])]
    extent = np.array([mn, mx])
    xy['SHAPE@X'] = np.round(xy['SHAPE@X'] - mn[0], 3)
    xy['SHAPE@Y'] = np.round(xy['SHAPE@Y'] - mn[1], 3)
    xy.dtype.names = ['X', 'Y']
    xy = repack_fields(xy)
    #
    # ---- (2) Prepare the oid data for use in identifying from-to points.
    uniq, indx, cnts = np.unique(oids, True, return_counts=True)
    id_vals = oids[indx]
    indx = np.concatenate((indx, [a.shape[0]]))
    #
    # ---- (3) Construct the IFT data using ``id_fr_to`` to carry the load.
    IFT_ = np.asarray(id_fr_to(xy, oids))
    cols = IFT_.shape[0]
    IFT = np.full((cols, 6), -1, dtype=np.int32)
    IFT[:, :3] = IFT_
    #
    # ---- (4) clockwise check for polygon parts to identify outer/inner rings
    if kind == 2:  # polygons
        xy_arr = stu(xy)   # View the data as an unstructured array
        cl_wise = np.array([cw(xy_arr[i[1]:i[2]]) for i in IFT_])
    else:  # not relevant for polylines or points
        cl_wise = np.full_like(oids, -1)
    IFT[:, 3] = cl_wise
    #
    # ---- (5) construct part_ids and pnt_nums
    if kind == 2:
        parts = [np.cumsum(IFT[:, 3][IFT[:, 0] == i]) for i in id_vals]
        part_ids = np.concatenate(parts)
        ar = np.where(IFT[:, 3] == 1)[0]
        ar0 = np.stack((ar[:-1], ar[1:])).T
        pnt_nums = np.zeros(IFT.shape[0], dtype=np.int32)
        for (i, j) in ar0:  # now provide the point numbers per part per shape
            pnt_nums[i:j] = np.arange((j - i))  # smooth!!!
    else:
        part_ids = np.ones_like(oids)
        pnt_nums = np.ones_like(oids)
    IFT[:, 4] = part_ids
    IFT[:, 5] = pnt_nums
    #
    # ---- (6) Create the output array... as easy as ``a`` to ``z``
    z = npg.Geo(xy_arr, IFT, kind, Extent=extent, Info="test", SR=sp_ref.name)
    out = copy.deepcopy(z)
    return out


# helper function
def id_fr_to(a, oids):
    """Produce the ``id, from, to`` point indices used by fc_nparray_Geo.

    NOTE : no error checking.

    Parameters
    ----------
    a : structured array
        The required format is dtype([('X', '<f8'), ('Y', '<f8')])
    oids : array
        An array of object ids derived from the feature class.  There is no
        guarantee that they will be as simple as a sequential list, so do not
        substitute with an np.arange(...) option
    """
    sze = a.shape[0]
    val = 0
    idx = []
    key = 0
    while (val < sze):
        w = np.where(a == a[val])[0]  # fast even with large arrays
        n = len(w)
        sub = [val]
        id_val = [oids[val]]  # a list for concatenation
        if n == 1:            # one found, use the next one
            val = w[0] + 1
        elif n == 2:          # two found, use the last one + 1
            key = w[-1]
            val = key + 1
            sub.append(val)
            idx.append(id_val + sub)
        elif n > 2:           # multiple found, identify where val fits in
            key = w[np.where(val < w)[0][0]]
            val = key + 1
            sub.append(val)
            idx.append(id_val + sub)
    return idx


# ============================================================================
# ---- (3b) Back to featureclass ----------------------------------------------
def Geo_to_fc(geo, gdb=None, name=None, kind=None, SR=None):
    """Return a FeatureClass from a Geo array."""
    SR = SR
    if kind in (None, 0, 1, 2):
        print("\n ``kind`` must be one of Polygon, Polyline or Point.")
        return None
    #
    dx, dy = geo.LL
    geo = geo.shift(dx, dy)
    polys = Geo_to_poly(geo, SR, kind)
    out_name = gdb.replace("\\", "/") + "/" + name
    wkspace = env.workspace = 'memory'  # legacy is in_memory
    tmp_name = "{}\\{}".format(wkspace, "tmp")
    if Exists(tmp_name):
        Delete(tmp_name)
    CreateFeatureclass(wkspace, "tmp", kind, spatial_reference=SR)
    AddField("tmp", 'ID_arr', 'LONG')
    with InsertCursor("tmp", ['SHAPE@', 'ID_arr']) as cur:
        for row in polys:
            cur.insertRow(row)
    CopyFeatures("tmp", out_name)
    return


# ---- Use this to convert Geo to arcpy shapes
def Geo_to_poly(geo, sr, kind="Polygon", as_singlepart=True):
    """Create poly features from a Geo array.

    Parameters
    ----------
    geo : geo array
    sr : spatial reference object
    kind : str
        Output geometry type.
    Geo_to_poly is called by Geo_to_fc.
    arr2poly does the actual poly construction

    >>> ps = Geo_to_poly(g, sr=g.SR, kind="Polygon", as_singlepart=True)
    >>> ps  # returns the single part representation of the polygon
    [(<Polygon object at 0x25f07f4d7f0[0x25f09232968]>, 1),
     (<Polygon object at 0x25f07f4d668[0x25f092327d8]>, 1),
     (<Polygon object at 0x25f07f4d160[0x25f09232828]>, 2),
     (<Polygon object at 0x25f07f4d208[0x25f092324b8]>, 2),
     (<Polygon object at 0x25f07f4d4a8[0x25f0af15558]>, 3)]
    """
    # ---- helper
    def arr2poly(a, SR):
        """Construct the poly feature from lists or arrays.

        The inputs may be nested and mixed in content.
        """
        aa = []
        for pairs in a:
            sub = pairs[0]
            oid = pairs[1]
            aa.append([Point(*pairs) for pairs in sub])
        if p_type.upper() == 'POLYGON':
            poly = Polygon(Array(aa), SR)
        elif p_type.upper() == 'POLYLINE':
            poly = Polyline(Array(aa), SR)
        return (poly, oid)
    #
    SR = sr
    if "gon" in kind.lower():
        p_type = "POLYGON"
    elif "line" in kind.lower():
        p_type = "POLYLINE"
    if as_singlepart:
        b_ift = geo.bit_IFT
        uni, idx = np.unique(b_ift[:, [0, 4]], True, axis=0)
    else:
        uni, idx = np.unique(geo.IDs, True, axis=0)
    ifts = np.split(geo.IFT, idx[1:])
    out = []
    for ift in ifts:
        sl = []
        for s in ift:
            oid = s[0]
            f, t = s[1:3]
            sl.append((geo.XY[f:t], oid))
        out.append(sl)
    polys = []
    for arr in out:
        polys.append(arr2poly(arr, SR))
    return polys


def view_poly(geo, id_num=1, view_as=2):
    """View a poly feature as an SVG in the console.

    Parameters
    ----------
    geo : Geo array
        The Geo array part to view.
    id_num : integer
        The shape in the Geo array to view.
    view_as : integer
        Polygon = 2, Polygon = 1, Multipoint = 0

    Notes
    -----
    These provide information on the content of the svg representation.

    >>> p0.__getSVG__()
    >>> p0._repr_svg_()
    f = [" M {},{} " + "L {},{} "*(len(b) - 1) for b in g0.bits]
    ln = [f[i].format(*b.ravel()) for i, b in enumerate(g0.bits)]
    st = "".join(ln) + "z"
    """
    if id_num not in (geo.IDs):
        msg = "Id ... {} ... not found.\n Use geo.IDs to see their values"
        print(msg.format(id_num))
        return
    shp = geo.get_shape(id_num)
    z = [Array([Point(*i) for i in b]) for b in shp.bits]
    if view_as == 2:
        return Polygon(Array(z))
    elif view_as == 1:
        return Polyline(Array(z))
    else:
        zz = []
        for i in z:
            zz.extend(i)
        return Multipoint(Array(zz))


# ============================================================================
# ---- (3c) attribute data
# Change FC <null> to a useful nodata value
def make_nulls(in_fc, include_oid=True, int_null=-999):
    """Return null values for a list of fields objects.

    Thes excludes objectid and geometry related fields.
    Throw in whatever else you want.

    Parameters
    ----------
    in_fc : featureclass or featureclass table
        Uses arcpy.ListFields to get a list of featureclass/table fields.
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
             'Integer': int_null, 'String': str(None), 'Text': str(None),
             'Date': np.datetime64('NaT'), 'Geometry': np.nan}
    #
    desc = Describe(in_fc)
    if desc['dataType'] not in ('FeatureClass', 'Table'):
        print("Only Featureclasses and tables are supported")
        return None, None
    in_flds = desc['fields']
    good = [f for f in in_flds if f.editable and f.type != 'Geometry']
    fld_dict = {f.name: f.type for f in good}
    fld_names = list(fld_dict.keys())
    null_dict = {f: nulls[fld_dict[f]] for f in fld_names}
    # ---- insert the OBJECTID field
    if include_oid and desc['hasOID']:
        oid_name = 'OID@'  # desc['OIDFieldName']
        oi = {oid_name: -999}
        null_dict = dict(list(oi.items()) + list(null_dict.items()))
        fld_names.insert(0, oid_name)
    return null_dict, fld_names


# Featureclass attribute data
def fc_data(in_fc):
    """Pull all editable attributes from a featureclass tables.

    During the process, <null> values are changed to an appropriate type.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass.

    Notes
    -----
    The output objectid and geometry fields are renamed to
    `OID_`, `X_cent`, `Y_cent`, where the latter two are the centroid values.
    """
    null_dict, fld_names = make_nulls(in_fc, include_oid=True, int_null=-999)
    flds = ["OID@", "SHAPE@", "SHAPE@X", "SHAPE@Y", "SHAPE@XY"]
    new_names = [[i, i.replace("@", "_")][i in flds] for i in fld_names]
    a = FeatureClassToNumPyArray(
        in_fc, fld_names, skip_nulls=False, null_value=null_dict
    )
    a.dtype.names = new_names
    return np.asarray(a)


# Featureclass table attribute data
def tbl_data(in_tbl):
    """Pull all editable attributes from a featureclass tables.

    During the process, <null> values are changed to an appropriate type.

    Parameters
    ----------
    in_tbl : text
        Path to the input featureclass.

    Notes
    -----
    The output objectid and geometry fields are renamed to
    `OID_`, `X_cent`, `Y_cent`, where the latter two are the centroid values.
    """
    flds = ['OID@']
    null_dict, fld_names = make_nulls(in_tbl, include_oid=True, int_null=-999)
    if flds not in fld_names:
        new_names = out_flds = fld_names
    if fld_names[0] == 'OID@':
        out_flds = flds + fld_names[1:]
        new_names = ['OID_', 'X_cent', 'Y_cent'] + out_flds[3:]
    a = TableToNumPyArray(
        in_tbl, out_flds, skip_nulls=False, null_value=null_dict
    )
    a.dtype.names = new_names
    return np.asarray(a)


# =========================================================================
# ---- (4) fc_geo_interface_Geo section
#  fc -> __geo_interface__ -> to Geo
#
def fc_geo_Geo(in_fc):
    """Shell to run `fc_parser`.

    subs, xys, polys = fc_to_Geo(in_fc)
    """
    ift, xys, polys = fc_gi_Geo(in_fc)
    name, K = get_shape_K(in_fc)
    g = npg.Geo(xys, IFT=ift, Kind=K, Info="geo_interface_Geo")
    return g


# def shape_finder(arr, ids=None):
#     """Provide the structure of an array/list which may be uneven and nested.

#     Parameters
#     ----------
#     arr : array-like
#         An array of objects. In this case points.
#     ids : integer
#         The object ID values for each shape. If ``None``, then values will be
#         returned as a sequence from zero to the length of ``arr``.
#     """
#     main = []
#     if ids is None:
#         ids = np.arange(len(arr))
#     arr = np.asarray(arr).squeeze()
#     cnt = 0
#     for i, a in enumerate(arr):
#         info = []
#         if hasattr(a, '__len__'):
#             a0 = np.asarray(a)
#             for j, a1 in enumerate(a0):
#                 if hasattr(a1, '__len__'):
#                     a1 = np.asarray(a1)
#                     if len(a1.shape) >= 2:
#                         info.append([ids[i], cnt, j, *a1.shape])
#         main.append(np.asarray(info))
#         cnt += 1
#     return main  # np.vstack(main)


def flat(l):
    """Flatten input. Basic flattening but doesn't yield where things are"""
    def _flat(l, r):
        """Recursive flattener."""
        if not isinstance(l[0], (list, np.ndarray, tuple)):  # added [0]
            r.append(l)
        else:
            for i in l:
                r = r + flat(i)
        return r
    return _flat(l, [])


# ----
def fc_gi_Geo(in_fc):
    """Convert FeatureClass __geo_interface__ to Geo array.

    Requires
    --------
    ``npg.shape_finder`` and ``flat`` are needed.
    """
    # ---- Gather the geometry objects and run ``npg.shape_finder``.
    SR = get_SR(in_fc)
    polys = []  # to store the arcpy polygons
    data = []   # coordinates for the geometry
    with SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as cur:
        for i, r in enumerate(cur):
            oid, shp = r[0], r[1]
            polys.append(shp)
            coords = shp.__geo_interface__['coordinates']  # ---- shape - array
            shps = npg.shape_finder(coords, oid)  # ---- in npGeo
            data.append([oid, shps, coords])
    # ---- construct ift
    ift_inf = np.vstack([d[1] for d in data])
    xys = np.vstack([flat(d[2]) for d in data])
    fr_to = np.cumsum(np.concatenate(([0], ift_inf[:, 3])))
    fr = fr_to[:-1]                          # from
    too = fr_to[1:]                          # to
    cw = np.where(ift_inf[:, 2] == 0, 1, 0)  # clockwise check
    c0 = ift_inf[:, 0]                       # shape id
    c1 = ift_inf[:, 1] + 1                   # part id, added 1 to part
    c2 = ift_inf[:, 2]                       # bit id
    ift = np.stack((c0, fr, too, cw, c1, c2), axis=1)
    return ift, xys, polys


# =========================================================================
# ---- (5) fc_sc_Geo section
# fc searchcursor to Geo
#
def fc_sc_Geo(in_fc):
    """Shell to run fc_parser

    subs, xys, polys = fc_to_Geo(in_fc)
    """
    subs, xys, polys = fc_parser(in_fc)
    name, K = get_shape_K(in_fc)
    ift = ift_maker(subs)
    g = npg.Geo(xys, IFT=ift, Kind=K, Info="fc_to_Geo")
    return g  # subs, xys, polys


def fc_parser(in_fc):
    """Examine polygon featureclass (in_fc) geometry.

    Returns
    -------
    subs : list
        A nested list enabling on to identify the components of polygon shapes.
        A shape can be multipart (parts), and each part can have holes (bits).
        The output from subs can be used in ``ift_make`` to produce a
        sequential list of points identified by their point number, shape,
        part and bit.  The results can be used in the Geo class in the
        ``npgeom`` package.
    xys : array
        The in_fc class geometry devolved into an array of N points of shape,
        (N, 2) (X, Y)
    polys : arcpy Polygon objects
        Just in case you want to use them for testing. Remove that return
        object from this function if you don't need them.
    """
    def _geom_(geom):
        """Gather the required information for each shape.

        It examines the Polygon geometry, determines whether there are
        multiple parts and/or holes in the parts.
        """
        out = [len(geom)]
        coords = []
        for prt, part in enumerate(geom):
            xys = []
            # locate ``None`` separators, split on those since they
            # separate inner and outer rings
            w = np.isin(part, None)
            s = np.where(w)[0]
            bits = np.split(part, s)
            psum = [prt, np.sum(~w), len(bits), s.tolist()]
            b_cnt = []
            for arr in bits:
                t = []
                for p in arr:
                    if p:
                        t.append((p.X, p.Y))
                xys.append(t)
                b_cnt.extend([len(t) if t else -1])
            psum.append(b_cnt)
            out.append(psum)
            coords.append(np.vstack(xys))
        coords = np.vstack(coords)
        return out, coords
    # ---- Gather the geometry objects and run them through ``_geom_``.
    SR = get_SR(in_fc)
    polys = []  # to store the arcpy polygons
    subs = []   # the magic box that stores info on shapes, parts and bits
    data = []   # coordinates for the geometry
    with SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as cur:
        for i, r in enumerate(cur):
            polys.append(r[1])
            geom, coords = _geom_(r[1])
            subs.append([r[0], geom])
            data.append(coords)
    xys = np.vstack(data)
    return subs, xys, polys


def ift_maker(subs):
    """Make the ift from the subs."""
    tmp = []
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    for sub in subs:
        id_val = sub[0]  # ---- polygon id value
        parts = sub[1][0]  # number of parts
        tot_pnts = 0
        tot_parts = 0
        for i in range(1, parts + 1):
            part, num_pnts, num_parts, spl, whr = sub[1][i]  # main split
            c1.extend(whr)
            c4.append(np.full(num_parts, fill_value=part+1, dtype='int32'))
            c5.append(np.arange(num_parts))
            tot_pnts += num_pnts
            tot_parts += num_parts
        tp = np.full(tot_parts, fill_value=id_val, dtype='int32')
        tmp.append(tp)  # tot_parts))
    # ----
    c0 = np.concatenate(tmp)
    fr = np.concatenate(([0], c1))
    fr = np.cumsum(fr)
    c1 = fr[:-1]
    c2 = fr[1:]
    c4 = np.concatenate(c4)
    c5 = np.concatenate(c5)
    c3 = np.where(c5 == 0, 1, 0)
    ift = np.stack((c0, c1, c2, c3, c4, c5), axis=1)  # c0
    return ift  # , tmp, c0, c1, c2, c3, c4, c5


# ============================================================================
# -- helpers, or standalone --
def fc2na(in_fc):
    """Return FeatureClassToNumPyArray.  Shorthand interface.

    Get the geometry from a featureclass and clean it up.  This involves
    shifting the coordinates to the 0,0 origin and rounding them.

    Returns
    -------
    oids : array
        The object id values as derived from the featureclass
    a : structured array
        The coordinates with named fields ('X', 'Y').  These are useful for
        sorting and/or finding duplicates.
    xy : ndarray
        The coordinates in ``a`` as an ndarray.
    """
    arr = FeatureClassToNumPyArray(
        in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'], explode_to_points=True
    )
    oids, x, y = [arr[name] for name in ['OID@', 'SHAPE@X', 'SHAPE@Y']]
    m = [np.min(x), np.min(y)]
    a = np.empty((len(x), ), dtype=np.dtype([('X', 'f8'), ('Y', 'f8')]))
    a['X'] = np.round(x - m[0], 3)
    a['Y'] = np.round(y - m[1], 3)
    xy = stu(a)
    return oids, a, xy


# Array to poly features
def array_poly(a, p_type=None, sr=None, IFT=None):
    """Assemble poly features from arrays.

    Used by `geometry_fc` or it can be used separately.

    Parameters
    ----------
    a : array
        Points array.
    p_type : text
        POLYGON or POLYLINE
    sr : spatial reference
        Spatial reference object, name or id.
    IFT : array
        An Nx6 array consisting of I(d)F(rom)T(o) points.

    Notes
    -----
    Polyline or polygon features can be created from the array data.  The
    features can be multipart with or without interior rings.

    Outer rings are ordered clockwise, inner rings (holes) are ordered
    counterclockwise.  For polylines, there is no concept of order.
    Splitting is modelled after _nan_split_(arr).
    """
    def _arr_poly_(arr, SR, as_type):
        """Slice the array where nan values appear, splitting them off."""
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
            aa.append([Point(*pairs) for pairs in sub])
        if as_type.upper() == 'POLYGON':
            poly = Polygon(Array(aa), SR)
        elif as_type.upper() == 'POLYLINE':
            poly = Polyline(Array(aa), SR)
        return poly
    # ----
    ids = IFT[:, 0]
    from_to = IFT[:, 1:3]
    chunks = [a.XY[f:t] for f, t in from_to]  # ---- _poly_pieces_ chunks input
    polys = []
    for i in chunks:
        p = _arr_poly_(i, sr, p_type)  # ---- _arr_poly_ makes parts of chunks
        polys.append(p)
    out = list(zip(polys, ids))
    return out


def geometry_fc(a, IFT, p_type=None, gdb=None, fname=None, sr=None):
    """Form poly features from the list of arrays created by `fc_geometry`.

    Parameters
    ----------
    a : array or list of arrays
        Some can be object arrays, normally created by ``pnts_arr``
    IFT : list/array
        Identifies which feature each input belongs to.  This enables one to
        account for multipart shapes
    p_type : string
        Uppercase geometry type eg POLYGON.
    gdb : text
        Geodatabase path and name.
    fname : text
        Featureclass name.
    sr : spatial reference
        name or object

    Returns
    -------
    Singlepart and/or multipart featureclasses.

    Notes
    -----
    The work is done by ``array_poly``.
    """
    if p_type is None:
        p_type = "POLYGON"
    out = array_poly(a, p_type.upper(), sr=sr, IFT=IFT)   # call array_poly
    name = gdb + "/" + fname
    wkspace = env.workspace = 'memory'  # legacy is in_memory
    CreateFeatureclass(wkspace, fname, p_type, spatial_reference=sr)
    AddField(fname, 'ID_arr', 'LONG')
    with InsertCursor(fname, ['SHAPE@', 'ID_arr']) as cur:
        for row in out:
            cur.insertRow(row)
    CopyFeatures(fname, name)
    return


# ============================================================================
# ---- (6) Geometry to array -------------------------------------------------
#
def poly2array(polys):
    """Convert polyline or polygon shapes to arrays for use in the Geo class.

    Parameters
    ----------
    polys : tuple, list
        Polyline or polygons in a list/tuple
    """
    null_pnt = [[np.nan, np.nan]]

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


def _flat(l, r):
    """Flatten iterable."""
    if not isinstance(l[0], (list, np.ndarray, tuple)):  # added [0]
        r.append(l)
    else:
        for i in l:
            r = r + flat(i)
    return r


# ---- EXTRAS and TESTS

def _del_none(a, cond=None):
    """Delete equivalent"""
    def _xy_(part):
        return [(p.X, p.Y) for p in part]
    # ----
    idx = np.where(np.isin(a, None))[0]
    xys = np.delete(a, idx)
    splitter = idx - np.arange(len(idx))
    too = np.array([0, len(xys)])
    fr = np.sort(np.concatenate((too, splitter)))
    fr_to = np.array([fr[:-1], fr[1:]]).T
    out = [xys[i:j] for i, j in fr_to]
    final = [np.array(_xy_(part)) for part in out]
    return final, out  # xys, splitter, final


# ---- (7) JSON, GeoJSON section ---------------------------------------------
#
def fc_json(in_fc, SR=None):
    """Produce arrays from the json representation of get_shapes shapes."""
    shapes = get_shapes(in_fc, SR=SR)
    if SR is None:
        SR = get_SR(in_fc)
    arr = []
    json_keys = [i for i in json.loads(shapes[0].JSON).keys()]
    geom_key = json_keys[0]
    for s in shapes:
        arr.append(json.loads(s.JSON)[geom_key])
    return arr


"""
Notes
p0 .... a polygon
p0[0] . its array
np.where(np.isin(p0[0], None))[0]
array([ 5, 11, 16], dtype=int32)

# now
z = np.where(np.isin(p0[0], None), 1, 0)
 array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
# or
z = np.where(np.isin(p0[0], None), 0, 1)
z = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])

# use
z = np.isin(p0[0], None, 0, 1)  # see above
w0 = np.where(z == 0)[0]        # array([ 5, 11, 16], dtype=int32)
w1 = w0 + 1                     # array([ 6, 12, 17], dtype=int32)
w01 = np.sort(np.concatenate((w0, w1)))
                                #  array([ 5,  6, 11, 12, 16, 17], dtype=int32)
s = np.split(z, w01)
[array([1, 1, 1, 1, 1]), array([0]), array([1, 1, 1, 1, 1]),
 array([0]), array([1, 1, 1, 1]), array([0]), array([1, 1, 1, 1])]


s_sum = [sum(i) for i in s]

***  good
out1
[array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]),
 array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1])]

def _z(a):
    z = [x.sum() for x in np.split(a, np.where(np.diff(a) != 0)[0] + 1)
         if x[0] == 1]
    return z

[_z(a) for a in out1]
[[5, 5, 4, 4], [5, 4]]
"""


def _poly_arr_(poly):
    """Return coordinates of nested objects.

    w = np.isin(part, None)
    s = np.where(w)[0]
    bits = np.split(part, s)
    """
    def _split_(part):
        yield [(p.X, p.Y) for p in part if p]

    out = []
    out1 = []
    for part in poly:
        w = np.where(np.isin(part, None, invert=True), 1, 0)
        tmp = (list(_split_(part)))
        out1.append(w)
        for i in tmp:
            out.append(np.array(i).squeeze())
    return out, out1


def __geo_interface__(self):
    """Geo interface function."""
    def split_part(a_part):
        part_list = []
        for item in a_part:
            if item is None:
                if part_list:
                    yield part_list
                part_list = []
            else:
                part_list.append((item.X, item.Y))
        if part_list:
            yield part_list
    part_json = [list(split_part(part))
                 for part in self]
    return {'type': 'MultiPolygon', 'coordinates': part_json}


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
