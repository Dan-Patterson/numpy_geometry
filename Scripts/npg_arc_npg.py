# -*- coding: utf-8 -*-
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
    2020-02-14

Purpose
-------
Geo array and arcpy geometry support functions.

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


import sys
import copy
# from textwrap import dedent, indent

import numpy as np

from numpy.lib.recfunctions import repack_fields
from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts

if 'npg' not in list(locals().keys()):
    import npgeom as npg

# import arcpy

from arcpy import (
    Array, Exists, Multipoint, Point, Polygon, Polyline
)  # ListFields

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
    'get_SR', 'fc_to_Geo', 'id_fr_to',          # option 1, use this ***
    'Geo_to_shapes', 'Geo_to_fc', 'view_poly',  # back to fc
    'make_nulls', 'fc_data', 'tbl_data',        # get attributes
    'fc2na', 'array_poly', 'geometry_fc',
    '_array_to_poly_', '_poly_to_array_',       # geometry to array
    '_poly_arr_', 'poly2array'                    # extras
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


# ============================================================================
# ---- (1) fc_to_Geo section
# -- fc -> nparray -> Geo  uses FeatureClassToNumpyArray
#
# -- main function --
#
def fc_to_Geo(in_fc, geom_kind=2, minX=0, minY=0, info=""):
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

    def _cw_(a):
        """Clockwise check."""
        return 1 if _area_part_(a) > 0. else 0

    def _SR_(in_fc, verbose=False):
        """Return the spatial reference of a featureclass."""
        desc = Describe(in_fc)
        SR = desc['spatialReference']
        return SR

    # ---- (1) Foundational steps
    # Create the array, extract the object id values.
    # To avoid floating point issues, extract the coordinates, round them to a
    # finite precision and shift them to the x-y origin
    #
    kind = geom_kind
    sp_ref = _SR_(in_fc)
    a = FeatureClassToNumPyArray(
        in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'],
        spatial_reference=sp_ref, explode_to_points=True
    )
    oids = a['OID@']
    xy = a[['SHAPE@X', 'SHAPE@Y']]
    mn = [np.min(xy['SHAPE@X']), np.min(xy['SHAPE@Y'])]
    mx = [np.max(xy['SHAPE@X']), np.max(xy['SHAPE@Y'])]
    extent = np.array([mn, mx])
    # ---- shift if needed
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
        cl_wise = np.array([_cw_(xy_arr[i[1]:i[2]]) for i in IFT_])
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
    """Produce the ``id, from, to`` point indices used by fc_to_Geo.

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
        if n < 1:
            continue
        elif n == 1:            # one found, use the next one
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
# ---- (2) Geo to arcpy shapes ----------------------------------------
#
def Geo_to_shapes(geo, as_singlepart=True):
    """Create poly features from a Geo array.

    Parameters
    ----------
    geo : Geo array
        Properties of the Geo array are used to derive the remaining
        parameters.
    as_singlepart : boolean
        True, turns multipart shapes into singlepart.  False retains the
        multipart nature.  Holes remain preserved.

    Notes
    -----
    SR, Kind :
        Spatial reference object and output geometry are derived from input
        Geo array.

    - `Geo_to_shapes` is called by `Geo_to_fc`.
    - `arr2poly` in the function, does the actual poly construction

    Example
    -------
    >>> ps = Geo_to_shapes(g, as_singlepart=True)
    >>> ps  # returns the single part representation of the polygons
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
# ---- (3) Geo to featureclass ----------------------------------------------
def Geo_to_fc(geo, gdb=None, name=None, kind=None, SR=None):
    """Return a FeatureClass from a Geo array."""
    SR = SR
    if kind in (None, 0, 1, 2):
        print("\n ``kind`` must be one of Polygon, Polyline or Point.")
        return None
    #
    # dx, dy = geo.LL
    # geo = geo.shift(dx, dy)
    polys = Geo_to_shapes(geo, as_singlepart=True)
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
# ---- (4) attribute data
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
    flds = ["Shape", "OID@", "SHAPE@", "SHAPE@X", "SHAPE@Y", "SHAPE@XY"]
    new_names = [[i, i.replace("@", "__")][i in flds] for i in fld_names]
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


# ---- Array to poly features
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


def array_poly(arr, p_type=None, sr=None, IFT=None):
    """Assemble poly features from arrays.

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

    def is_Geo_(obj):
        """From : Function of npgeom.npGeo module"""
        if ('Geo' in str(type(obj))) & (issubclass(obj.__class__, np.ndarray)):
            return True
        return False
    # ----
    if is_Geo_(arr):
        # ids = arr.IFT[:, 0]
        from_to = arr.IFT[:, 1:3]
        arr = [arr.XY[f:t] for f, t in from_to]  # --- _poly_pieces_ chunks
        polys = []
    for a in arr:
        # p = _array_to_poly_(a, sr, p_type)  # makes parts of chunks
        p = _arr_poly_(a, sr, p_type)
        polys.append(p)
    out = list(zip(polys))  # , ids))
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
# ---- (4) mini helpers -------------------------------------------------
#
# ---- array to polygon/polyline
#
def _array_to_poly_(arr, SR=None, as_type="Polygon"):
    """Convert an `ndarray`, an `object array` or a `list of lists`
    to an arcpy polygon or polyline geometry.

    Parameters
    ----------
    arr : list-like
        A list of arrays representing the poly parts, or an object array.
    SR : spatial reference
        Leave as None if not known.
    as_type : text
        Polygon or Polyline.

    Notes
    -----
    Polygon geometries require a duplicate first and last point.
    Outer rings are ordered clockwise with inner rings (holes)
    counterclockwise.
    No check are made to the integrety of the geometry in this function.
    """
    subs = np.asarray(arr)
    aa = []
    for sub in subs:
        aa.append([Point(*pairs) for pairs in sub])
    if as_type.upper() == 'POLYGON':
        poly = Polygon(Array(aa), SR)
    elif as_type.upper() == 'POLYLINE':
        poly = Polyline(Array(aa), SR)
    return poly


# ---- polygon/polyline to array
def _poly_to_array_(polys):
    """Convert polyline or polygon shapes to arrays for use in numpy.

    Parameters
    ----------
    polys : tuple, list
        Polyline or polygons in a list/tuple
    """
    def _p2p_(poly):
        """Convert a single ``poly`` shape to numpy arrays or object"""
        sub = []
        pt = Point()  # arcpy.Point()
        for arr in poly:
            pnts = [[pt.X, pt.Y] for pt in arr if pt]
            sub.append(np.asarray(pnts))
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

    w = np.isin(part, None)
    s = np.where(w)[0]
    bits = np.split(part, s)
    """
    def _split_(part):
        yield [(p.X, p.Y) for p in part if p]
    # ----
    arrs =[]
    for part in poly:
        out = []
        w = np.where(np.isin(part, None, invert=False))[0]
        bits = np.split(part, w)
        for i in bits:
            sub = _split_(i)
            out.append(np.array(*sub).squeeze())
            # out.append(*sub)
        arrs.append(np.asarray(out).squeeze())
        # arrs.append(out)
    return np.asarray(arrs)


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
        """Convert a single ``poly`` shape to numpy arrays or object"""
        sub = []
        for arr in poly:
            pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
            sub.append(np.asarray(pnts))
        return sub
    # ----
    if len(polys) == 1:  # not isinstance(polys, (list, tuple)):
        polys = [polys]
    out = []
    for poly in polys:
        out.append(_p2p_(poly))
    return out


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons"
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\Polygons2"
