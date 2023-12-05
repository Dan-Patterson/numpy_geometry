# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
---------------------------------------------
  npg_arc_npg: Functions that require `arcpy`
---------------------------------------------

Functions for reading/writing Geo array and arcpy polygon/polyline shapes.

----

Script :
    npg_arc_npg.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2022-11-23

Purpose
-------
Geo array and arcpy geometry support functions.

Notes
-----
See `_npgeom_notes_.py` for extra notes.

Using da.SearchCursor to project data::

    fc2 = r'C:/Git_Dan/npgeom/Project_npg/tests.gdb/sq'
    import arcpy
    SR0 = arcpy.da.Describe(fc2)['spatialReference']
    SR1 = arcpy.SpatialReference(4326)
    # --   in its native projected coordinate system
    with arcpy.da.SearchCursor(
            fc2, ['SHAPE@X', 'SHAPE@Y'], spatial_reference=SR0,
            explode_to_points=False) as cur:
        a = cur._as_narray()
    # --   projected to a GCS
    with arcpy.da.SearchCursor(
            fc2, ['SHAPE@X', 'SHAPE@Y'], spatial_reference=SR1,
            explode_to_points=False) as cur:
        b = cur._as_narray()
    a
    array([( 300005.48,  5000004.88), ( 300010.33,  5000010.33),
           ( 300006.33,  5000011.22), ( 300005.75,  5000013.42)],
          dtype=[('SHAPE@X', '<f8'), ('SHAPE@Y', '<f8')])
    b
    array([(-76.56,  45.14), (-76.56,  45.14),
           (-76.56,  45.14), (-76.56,  45.14)],
          dtype=[('SHAPE@X', '<f8'), ('SHAPE@Y', '<f8')])

Convert point array to point featureclass::

    # pl_n is a Nx2 numpy array
    pl_ns = npg.npg_geom_ops._view_as_struct_(pl_n + [300000, 5000000])
    SR = r'NAD 1983 CSRS MTM  9'
    fc = r"C:\arcpro_npg\Project_npg\npgeom.gdb\edgy1_allpnts"
    NumPyArrayToFeatureClass(pl_ns, fc, ["f0", "f1"], SR)


References
----------
None (yet).
"""
# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121


import sys
import copy
# from textwrap import dedent, indent

import numpy as np

from numpy.lib.recfunctions import repack_fields
from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts

# if 'npg' not in list(locals().keys()):
#    import npgeom as npg

from npg import npGeo  # noqa
from npg.npGeo import Geo
# from npGeo import *

import arcpy
from arcpy import Array, Exists, Multipoint, Point, Polygon, Polyline

from arcpy.da import (
    Describe, InsertCursor, SearchCursor, FeatureClassToNumPyArray,
    TableToNumPyArray)  # ExtendTable, NumPyArrayToTable,  UpdateCursor

from arcpy.management import (
    AddField, CopyFeatures, CreateFeatureclass, Delete)  # DeleteFeatures


script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    'get_SR', 'get_shape_K',
    '_fc_shapes_', '_fc_as_narray_', '_fc_geo_interface_',
    '_json_geom_',
    'fc_to_Geo', 'id_fr_to',                        # option 1, use this ***
    'Geo_to_arc_shapes', 'Geo_to_fc', 'view_poly',  # back to fc
    'make_nulls', 'fc_data', 'tbl_data',            # get attributes
    'fc2na', 'attr_to_npz', 'array_poly', 'geometry_fc',
    '_array_to_poly_', '_poly_to_array_',           # geometry to array
    '_poly_arr_', 'poly2array',                     # extras
    'fc_union', 'shp_dissolve', 'fc_dissolve'       # arcpy functions
]   # '_del_none', '__geo_interface__', '_flat',


# ============================================================================
# ---- Helpers
# Spatial reference object
def get_SR(in_fc, verbose=False):
    """Return the spatial reference of a featureclass."""
    desc = Describe(in_fc)
    SR = desc['spatialReference']
    if verbose:
        print("SR name: {}  factory code: {}".format(SR.name, SR.factoryCode))
    return SR


def get_shape_K(in_fc):
    """Return shape type for a featureclass.  Returns (kind, k)."""
    desc = Describe(in_fc)
    kind = desc['shapeType']
    if kind in ('Polygon', 'PolygonM', 'PolygonZ'):
        return (kind, 2)
    if kind in ('Polyline', 'PolylineM', 'PolylineZ'):
        return (kind, 1)
    if kind in ('Point', 'Multipoint'):
        return (kind, 0)


# ---- (1) featureclass geometry using...
def _fc_shapes_(in_fc, with_id=True):
    """Return geometry from a featureclass as geometry objects."""
    flds = ["SHAPE@"]
    if with_id:
        flds = ["OID@", "SHAPE@"]
    with SearchCursor(in_fc, flds) as cursor:
        if with_id:
            a = [(row[0], row[1]) for row in cursor]
        else:
            a = [row[0] for row in cursor]
    return a


def _fc_as_narray_(in_fc, with_id=True):
    """Return geometry from a featureclass using `as_narray`."""
    flds = ["SHAPE@X", "SHAPE@Y"]
    if with_id:
        flds = ["OID@", "SHAPE@X", "SHAPE@Y"]
    with SearchCursor(in_fc, flds, explode_to_points=True) as cursor:
        a = cursor._as_narray()
    del cursor
    return a


def _fc_geo_interface_(in_fc, with_id=True):
    """Return geometry from a featureclass using `__geo_interface__`."""
    flds = ["SHAPE@"]
    if with_id:
        flds = ["OID@", "SHAPE@"]
    with SearchCursor(in_fc, flds) as cursor:
        if with_id:
            a = [(row[0], row[1].__geo_interface__['coordinates'])
                 for row in cursor]
        else:
            a = [row[0].__geo_interface__['coordinates'] for row in cursor]
    del cursor
    return a


# ---- (2) geojson and json geometry...
def _json_geom_(pth):
    """Return polygon/polyline geometry from a geoJSON or JSON file.

    Parameters
    ----------
    pth : text
        The file path to the json or geojson file.  The file extension is key.

    Notes
    -----
    No error checking is done to ensure that the file provided to the function
    complies with the structure required.  It is a convenience function.
    """
    import json
    json_type = pth.split(".")[-1]
    with open(pth) as f:
        data = json.load(f)  # kys = data.keys()
    if json_type == 'geojson':
        a = [i['geometry']['coordinates'] for i in data['features']]
    elif json_type == 'json':
        as_type = 'rings' if "Polygon" in data['geometryType'] else 'paths'
        a = [i['geometry'][as_type] for i in data['features']]
    else:
        print("json file must end in `geojson` or `json` file extension.")
    return a


# ============================================================================
# ---- (3) fc_to_Geo section
# -- fc -> nparray -> Geo  uses FeatureClassToNumpyArray
#
# -- main function --
#
def fc_to_Geo(in_fc, geom_kind=2, minX=0, minY=0, sp_ref=None, info=""):
    """Convert a FeatureClassToNumPyArray to a Geo array.

    This works with the geometry only.  Skip the attributes for later.  The
    processing requirements are listed below.  Just copy and paste.

    Parameters
    ----------
    in_fc : featureclass
        Featureclass in a file geodatabase.
    geom_kind : integer
        Points (0), Polylines (1) and Polygons (2)

    minX, minY : numbers
        If these values are 0, then the minimum values will be determined and
        used to shift the data towards the origin.
    sp_ref : text
        Spatial reference name.  eg `'NAD_1983_CSRS_MTM_9'`

    Notes
    -----
    The `arcpy.da.Describe` method takes a substantial amount of time.
    >>> %timeit Describe(fc2)
    >>> 355 ms ± 17.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    def _area_part_(a):
        """Mini e_area, used by areas and centroids."""
        x0, y1 = (a.T)[:, 1:]
        x1, y0 = (a.T)[:, :-1]
        e0 = np.einsum('...i,...i->...i', x0, y0)
        e1 = np.einsum('...i,...i->...i', x1, y1)
        return np.sum((e0 - e1) * 0.5)

    def _cw_(a):
        """Clockwise check."""
        return 1 if _area_part_(a) > 0. else 0

    # -- (1) Foundational steps
    # Create the array, extract the object id values.
    # To avoid floating point issues, extract the coordinates, round them to a
    # finite precision and shift them to the x-y origin
    #
    kind = geom_kind
    if sp_ref is None:      # sp_ref = get_SR(in_fc, verbose=False)
        sp_ref = "undefined"
    a = FeatureClassToNumPyArray(
        in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'],
        explode_to_points=True)  # spatial_reference=sp_ref
    oids = a['OID@']
    xy = a[['SHAPE@X', 'SHAPE@Y']]
    mn = [np.min(xy['SHAPE@X']), np.min(xy['SHAPE@Y'])]
    mx = [np.max(xy['SHAPE@X']), np.max(xy['SHAPE@Y'])]
    extent = np.array([mn, mx])
    # -- shift if needed
    dx, dy = mn
    if minX != 0.:
        dx = minX  # mn[0] - minX
    if minY != 0.:
        dy = minY  # mn[1] - minY
    xy['SHAPE@X'] = np.round(xy['SHAPE@X'] - dx, 3)
    xy['SHAPE@Y'] = np.round(xy['SHAPE@Y'] - dy, 3)
    xy.dtype.names = ['X', 'Y']
    xy = repack_fields(xy)
    #
    # -- (2) Prepare the oid data for use in identifying from-to points.
    uniq, indx, cnts = np.unique(oids, True, return_counts=True)
    id_vals = oids[indx]
    indx = np.concatenate((indx, [a.shape[0]]))
    #
    # -- (3) Just return points and extent for point data

    # -- (4) Construct the IFT data using `id_fr_to` to carry the load.
    if kind == 2:
        IFT_ = np.asarray(id_fr_to(xy, oids))
    elif kind == 1:
        IFT_ = np.vstack((id_vals, indx[:-1], indx[1:])).T
    cols = IFT_.shape[0]
    IFT = np.full((cols, 6), -1, dtype=np.int32)
    IFT[:, :3] = IFT_
    #
    # -- (5) clockwise check for polygon parts to identify outer/inner rings
    if kind == 2:  # polygons
        xy_arr = stu(xy)   # View the data as an unstructured array
        cl_wise = np.array([_cw_(xy_arr[i[1]:i[2]]) for i in IFT_])
    else:  # not relevant for polylines or points
        xy_arr = stu(xy)
        cl_wise = np.full(IFT.shape[0], 1)   # np.full_like(oids, -1)
    IFT[:, 3] = cl_wise
    #
    # -- (6) construct part_ids and pnt_nums
    if kind == 2:
        parts = [np.cumsum(IFT[:, 3][IFT[:, 0] == i]) for i in id_vals]
        part_ids = np.concatenate(parts)
        ar = np.where(IFT[:, 3] == 1)[0]
        ar0 = np.stack((ar[:-1], ar[1:])).T
        pnt_nums = np.zeros(IFT.shape[0], dtype=np.int32)
        for (i, j) in ar0:  # now provide the point numbers per part per shape
            pnt_nums[i:j] = np.arange((j - i))  # smooth!!!
    elif kind == 1:  # single part polylines assumed
        part_ids = np.full(IFT.shape[0], 1)
        pnt_nums = np.full(IFT.shape[0], 1)
        ar = np.where(IFT[:, 3] == 1)[0]
        ar0 = np.stack((ar[:-1], ar[1:])).T
        pnt_nums = np.zeros(IFT.shape[0], dtype=np.int32)
        for (i, j) in ar0:  # now provide the point numbers per part per shape
            pnt_nums[i:j] = np.arange((j - i))  # smooth!!!
    #
    IFT[:, 4] = part_ids
    IFT[:, 5] = pnt_nums
    #
    # -- (7) Create the output array... as easy as ``a`` to ``z``
    z = Geo(xy_arr, IFT, kind, Extent=extent, Info="test", SR=sp_ref)
    out = copy.deepcopy(z)
    return out


# helper function
def id_fr_to(a, oids):
    """Produce the `id, from, to` point indices used by fc_to_Geo.

    .. note::
       No error checking.  This function is used by `fc_to_Geo` .

    Parameters
    ----------
    a : structured array
        The required format is dtype([('X', '<f8'), ('Y', '<f8')])
    oids : array
        An array of object ids derived from the feature class.  There is no
        guarantee that they will be as simple as a sequential list, so do not
        substitute with an np.arange(...) option.
    """
    sze = a.shape[0]
    val = 0
    idx = []
    key = 0
    while val < sze:
        w = np.where(a == a[val])[0]  # fast even with large arrays
        n = len(w)
        sub = [val]
        id_val = [oids[val]]  # a list for concatenation
        if n < 1:
            continue
        if n == 1:          # one found, use the next one
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
# ---- (4) Geo to arcpy shapes ----------------------------------------
#
def Geo_to_arc_shapes(geo, as_singlepart=True):
    """Create poly features from a Geo array.

    Parameters
    ----------
    geo : Geo array
        Properties of the Geo array are used to derive the remaining
        parameters.
    as_singlepart : boolean
        True, turns multipart shapes into singlepart.  False, retains the
        multipart nature.  Holes remain preserved.

    Notes
    -----
    SR, Kind :
        Spatial reference object and output geometry are derived from input
        Geo array.

    - `Geo_to_arc_shapes` is called by `Geo_to_fc`.
    - `arr2poly` in the function, does the actual poly construction.

    Example
    -------
    >>> ps = Geo_to_arc_shapes(g, as_singlepart=True)
    >>> ps  # returns the single part representation of the polygons
    [(<Polygon object at 0x25f07f4d7f0[0x25f09232968]>, 1),
     (<Polygon object at 0x25f07f4d668[0x25f092327d8]>, 1),
     (<Polygon object at 0x25f07f4d160[0x25f09232828]>, 2),
     (<Polygon object at 0x25f07f4d208[0x25f092324b8]>, 2),
     (<Polygon object at 0x25f07f4d4a8[0x25f0af15558]>, 3)]
    """
    # -- helper
    def arr2poly(a, SR):
        """Construct the poly feature from lists or arrays.

        The inputs may be nested and mixed in content.
        """
        aa = []
        poly = None
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
    SR = geo.SR
    if geo.K == 2:
        p_type = "POLYGON"
    elif geo.K == 1:
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


# ============================================================================
# ---- (5) Geo to featureclass ----------------------------------------------
def Geo_to_fc(geo, gdb=None, name=None, kind=None, SR=None):
    """Return a FeatureClass from a Geo array."""
    SR = SR
    if kind in (None, 0, 1, 2):
        print("\n ``kind`` must be one of Polygon, Polyline or Point.")
        return None
    #
    # dx, dy = geo.LL
    # geo = geo.shift(dx, dy)
    polys = Geo_to_arc_shapes(geo, as_singlepart=True)
    out_name = gdb.replace("\\", "/") + "/" + name
    wkspace = arcpy.env.workspace = r'memory'  # legacy is in_memory
    tmp_name = r"memory\tmp"  # r"{}\{}".format(wkspace, "tmp")
    if Exists(tmp_name):
        Delete(tmp_name)
    CreateFeatureclass(wkspace, "tmp", kind, spatial_reference=SR)
    AddField("tmp", 'ID_arr', 'LONG')
    with InsertCursor("tmp", ['SHAPE@', 'ID_arr']) as cur:
        for row in polys:
            cur.insertRow(row)
    CopyFeatures("tmp", out_name)
    return None


def view_poly(geo, id_num=1, view_as=2):
    """View a single poly feature as an SVG in the console.

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
        return None
    shp = geo.get_shapes(id_num)
    z = [Array([Point(*i) for i in b]) for b in shp.bits]
    if view_as == 2:
        return Polygon(Array(z))
    if view_as == 1:
        return Polyline(Array(z))
    zz = []
    for i in z:
        zz.extend(i)
    return Multipoint(Array(zz))


# ============================================================================
# ---- (6) attribute data
# Change FC <null> to a useful nodata value
def make_nulls(in_fc, include_oid=True, include_editable=True, int_null=-999):
    """Return null values for a list of fields objects.

    This excludes objectid and geometry related fields.
    Throw in whatever else you want.

    Parameters
    ----------
    in_fc : featureclass or featureclass table
        Uses arcpy.ListFields to get a list of featureclass/table fields.
    include_oid : boolean
        Include the `object id` field to denote unique records and geometry
        in featureclasses or geodatabase tables.  This is recommended, if you
        wish to join attributes back to geometry.
    include_editable : boolean
        Include `editable fields` but not the `geometry` field.  eg Shape_Area
    int_null : integer
        A default to use for integer nulls since there is no ``nan`` equivalent
        Other options include

    >>> np.iinfo(np.int32).min # -2147483648
    >>> np.iinfo(np.int16).min # -32768
    >>> np.iinfo(np.int8).min  # -128

    >>> [i for i in cur.__iter__()]
    >>> [[j if j else -999 for j in i] for i in cur.__iter__() ]

    Notes
    -----
    The output objectid and geometry fields are renamed to
    `OID_`, `X_cent`, `Y_cent`, where the latter two are the centroid values.
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
    if include_editable:
        good = [f for f in in_flds if f.type not in ['OID', 'Geometry']]
    else:
        good = [f for f in in_flds if f.editable and f.type != 'Geometry']
    #
    fld_dict = {f.name: f.type for f in good}
    fld_names = list(fld_dict.keys())
    null_dict = {f: nulls[fld_dict[f]] for f in fld_names}
    # -- insert the OBJECTID field
    if include_oid and desc['hasOID']:
        oid_name = desc['OIDFieldName']
        oi = {oid_name: -999}
        null_dict = dict(list(oi.items()) + list(null_dict.items()))
        fld_names.insert(0, oid_name)
    return null_dict, fld_names


# Featureclass attribute data
def fc_data(in_fc, include_oid=True, int_null=-999):
    """Return geometry, text and numeric attributes from a featureclass table.

    During the process, <null> values are changed to an appropriate type.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass.
    include_oid, int_null : boolean
        See `make_nulls` for description
    """
    null_dict, fld_names = make_nulls(in_fc, include_oid, int_null)
    flds = ["Shape", "SHAPE@", "SHAPE@X", "SHAPE@Y", "SHAPE@Z",
            "SHAPE@XY", "SHAPE@TRUECENTROID", "SHAPE@JSON",
            "SHAPE@AREA", "SHAPE@LENGTH"]  # "OID@"
    new_names = []
    if fld_names is not None:
        new_names = [[i, i.replace("@", "__")][i in flds] for i in fld_names]
    else:
        fld_names = "*"
    a = TableToNumPyArray(
        in_fc, fld_names, skip_nulls=False, null_value=null_dict
    )
    if new_names:
        a.dtype.names = new_names
    return np.asarray(a)


# Featureclass table attribute data
def tbl_data(in_tbl, int_null=-999):
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
    null_dict, fld_names = make_nulls(in_tbl, include_oid=True,
                                      int_null=int_null)
    if flds not in fld_names:
        new_names = out_flds = fld_names
    if fld_names[0] == 'OID@':
        out_flds = flds + fld_names[1:]
        new_names = ['OID_'] + out_flds[1:]  # 'X_c', 'Y_c'] + out_flds[3:]
    a = TableToNumPyArray(
        in_tbl, out_flds, skip_nulls=False, null_value=null_dict
    )
    a.dtype.names = new_names
    return np.asarray(a)


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

    Notes
    -----
    Projected/planar coordinates are assumed and they are rounded to the
    nearest millimeter, change if you like.
    """
    arr = FeatureClassToNumPyArray(
        in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'], explode_to_points=True
    )
    oids, x, y = [arr[name] for name in ['OID@', 'SHAPE@X', 'SHAPE@Y']]
    m = [np.min(x), np.min(y)]
    a = np.empty((len(x), ), dtype=np.dtype([('X', 'f8'), ('Y', 'f8')]))
    a['X'] = np.round(x - m[0], 3)  # round `X` and `Y` values
    a['Y'] = np.round(y - m[1], 3)
    xy = stu(a)
    return oids, a, xy


def attr_to_npz(fc_name, out_name):
    r"""Save the attributes associated with a geo array.

    Parameters
    ----------
    fc_name : text
        The complete featureclass name in the geodatabase to save.
    out_name : text
        The full path, filename and extension for the output `*.npz`.
    array_name : text
        The name to assign to the output array.  Use `raw` format.

        >>> out_name = r"C:\arcpro_npg\data\ontario_attr.npz"

    Returns
    -------
        Saves the structured array containing the attribute data, the field
        names and type of the data.

    Example
    -------
    >>> f_name = "C:/arcpro_npg/data/ontario_attr.npz"
    >>> attr_to_npz(f_name, "data")

    See Also
    --------
    `npg_io.load_geo_attr` retrieves the array information from the .npz file.
    """
    #
    # Run tbl_data and make_nulls to create the attribute array
    a = tbl_data(fc_name, int_null=-999)
    b = a.dtype.names
    out_name = out_name.replace("\\", "/")
    np.savez(out_name, arr=a, fields=b)
    print("\nGeo array saved to ... {} ...".format(out_name))


# ---- (7) Geo or ndarrays to poly features
#
def array_poly(arrs, p_type=None, sr=None, IFT=None):
    """Assemble poly features from Geo or ndarrays.

    Used by `geometry_fc` or it can be used separately.

    Parameters
    ----------
    arr : arrays
        Points array(s).
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
    counterclockwise.
    """
    def _arr_poly_(arr, SR, as_type):
        """Slice the array where nan values appear, splitting them off."""
        aa = [Point(*pairs) for pairs in arr]
        if as_type.upper() == 'POLYGON':
            poly = Polygon(Array(aa), SR)
        elif as_type.upper() == 'POLYLINE':
            poly = Polyline(Array(aa), SR)
        return poly
    # --
    polys = []
    if hasattr(arrs, "IFT"):
        from_to = arrs.IFT[:, 1:3]
        arrs = [arrs.XY[f:t] for f, t in from_to]  # --- _poly_pieces_ chunks
    elif isinstance(arrs, np.ndarray):
        if len(arrs.shape) == 2:
            arrs = [arrs]
        elif arrs.dtype == 'O':
            arrs = arrs
    for a in arrs:
        p = _arr_poly_(a, sr, p_type)
        polys.append(p)
    return polys


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
    wkspace = arcpy.env.workspace = r'memory'  # legacy is in_memory
    CreateFeatureclass(wkspace, fname, p_type, spatial_reference=sr)
    AddField(fname, 'ID_arr', 'LONG')
    with InsertCursor(fname, ['SHAPE@', 'ID_arr']) as cur:
        for row in out:
            cur.insertRow(row)
    CopyFeatures(fname, name)
    return None


# ============================================================================
# ---- (8) mini helpers -------------------------------------------------
#
# -- array to polygon/polyline
#
def _array_to_poly_(arr, SR=None, as_type="Polygon"):
    """Convert array-like objects to arcpy geometry.

    This can include an `ndarray`, an `object array` or a `list of lists`
    which represent polygon or polyline geometry.

    Parameters
    ----------
    arr : list-like
        A list of arrays representing the poly parts, or an object array.
    SR : spatial reference
        Leave as None if not known.  Enclose in quotes. eg. "4326"
    as_type : text
        Polygon or Polyline.

    Notes
    -----
    Polygon geometries require a duplicate first and last point.
    Outer rings are ordered clockwise with inner rings (holes)
    counterclockwise.
    No check are made to the integrety of the geometry in this function.
    """
    subs = np.asarray(arr, dtype='O')
    aa = []
    for sub in subs:
        aa.append([Point(*pairs) for pairs in sub])
    if as_type.upper() == 'POLYGON':
        poly = Polygon(Array(aa), SR)
    elif as_type.upper() == 'POLYLINE':
        poly = Polyline(Array(aa), SR)
    return poly


def _poly_to_array_(polys):
    """Convert polyline or polygon shapes to arrays for use in numpy.

    Parameters
    ----------
    polys : tuple, list
        Polyline or polygons in a list/tuple
    """
    def _p2p_(poly):
        """Convert a single ``poly`` shape to numpy arrays or object."""
        sub = []
        pt = Point()  # arcpy.Point()
        for arr in poly:
            pnts = [[p.X, p.Y] for p in arr if pt]
            sub.append(np.asarray(pnts, dtype='O'))
        return sub
    # ----
    if not isinstance(polys, (list, tuple)):
        polys = [polys]
    out = []
    for poly in polys:
        out.extend(_p2p_(poly))  # or append, extend it is
    return out


def _poly_arr_(poly):
    """Return coordinates of nested objects.

    >>> w = np.isin(part, None)
    >>> s = np.where(w)[0]
    >>> bits = np.split(part, s)
    """
    def _split_(part):
        yield [(p.X, p.Y) for p in part if p]
    # ----
    arrs = []
    for part in poly:
        out = []
        w = np.where(np.isin(part, None, invert=False))[0]
        bits = np.split(part, w)
        for i in bits:
            sub = _split_(i)
            out.append(np.array(*sub).squeeze())
            # out.append(*sub)
        arrs.append(np.asarray(out, dtype='O').squeeze())
        # arrs.append(out)
    return np.asarray(arrs, dtype='O')


# ============================================================================
# ---- extras
def poly2array(polys):
    """Convert polyline or polygon shapes to arrays for use in the Geo class.

    Parameters
    ----------
    polys : tuple, list
        Polyline or polygons in a list/tuple
    """
    null_pnt = [[np.nan, np.nan]]

    def _p2p_(poly):
        """Convert a single ``poly`` shape to numpy arrays or object."""
        sub = []
        for arr in poly:
            pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
            sub.append(np.asarray(pnts, dtype='O'))
        return sub
    # ----
    if len(polys) == 1:  # not isinstance(polys, (list, tuple)):
        polys = [polys]
    out = []
    for poly in polys:
        out.append(_p2p_(poly))
    return out


def _to_ndarray(in_fc, to_pnts=True):
    """Convert searchcursor shapes an ndarray quickly.

    Parameters
    ----------
    in_fc : featureclass
    to_pnts : boolean
        True, returns all points in the geometry.  False, returns the shape
        centroid.

    See Also
    --------
    `_geom_as_narray_` returns a simplified version.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    with SearchCursor(in_fc, flds, explode_to_points=to_pnts) as cur:
        flds = cur.fields
        dt = cur._dtype
        a = cur._as_narray()
    return a, flds, dt


# ============================================================================
# ---- (9) arcpy functions
def fc_union(in_fc, poly_type="polygon"):
    """Union features in a featureclass.

    The output shape is built from its individual parts.
    Shared boundaries will be dissolved.

    Parameters
    ----------
    in_fc : featureclass
    poly_type : text
        Either `polygon` or `polyline`
    fc = "C:/Git_Dan/npgeom/Project_npg/tests.gdb/sq"
    """
    arr = []
    SR = get_SR(in_fc, verbose=False)
    with SearchCursor(in_fc, ['SHAPE@']) as cursor:
        for row in cursor:
            poly = row[0]
            for cnt in range(poly.partCount):
                part = poly.getPart(cnt)
                arr.append(part)
    a = Array(arr)
    if poly_type == "polygon":
        return Polygon(a, SR)
    if poly_type == "polyline":
        return Polyline(a, SR)
    else:
        print("Not polygon or polyline")
        return None


def shp_dissolve(polys):
    """Dissolve polygon boundaries."""
    poly = polys[0]
    for i, p in enumerate(polys[1:]):
        poly = poly.union(p)
    return poly


def fc_dissolve(in_fc, poly_type="polygon"):
    r"""Union features in a featureclass.

    The output shape is built from its individual parts.
    Shared boundaries will be dissolved.

    Parameters
    ----------
    in_fc : featureclass
    poly_type : text
        Either `polygon` or `polyline`
    fc = r'C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq'
    """
    with SearchCursor(in_fc, ['SHAPE@']) as cursor:
        shps = [row[0] for row in cursor]
    return shp_dissolve(shps)


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    # in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
    # in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
