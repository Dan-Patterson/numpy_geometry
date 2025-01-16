# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
tbx_tools
---------

**Toolbox tools for Free tools.**

Script :
    'C:/arcpro_npg/tbx_tools.py'
Author :
    Dan_Patterson@carleton.ca
Modified :
    2023-10-13

Purpose
-------
Tools to provide ``free`` advanced license functionality for ArcGIS Pro,
amongst other things.

Notes
-----
None

References
----------
**Advanced license tools**

Some of the functions that you can replicate using this data class would
include:

**Attribute Tools**

AttributeSort

Crosstabulate

**Containers**

`1 Bounding circles
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum
-bounding-geometry.htm>`_.  minimum area bounding circle

`2 Convex hulls
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum
-bounding-geometry.htm>`_.

`3 Feature Envelope to Polygon
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-envelope-to-polygon.htm>`_.  axis oriented envelope

**Conversion**

`1 Feature to Point
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-to-point.htm>`_.  centroid for point clusters, polylines or polygons

`2 Polygons to Polylines
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
feature-to-polygon.htm>`_.  Simple conversion.

`3 Feature Vertices to Points
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-vertices-to-points.htm>`_.

**Alter geometry**

`Shift, move, translate features
<https://pro.arcgis.com/en/pro-app/tool-reference/editing/
transform-features.htm>`_.

`Sort Geometry
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/sort.htm>`_.

`Shift features
<https://pro.arcgis.com/en/pro-app/tool-reference/editing/
transform-features.htm>`_.

`Split Line at Vertices
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
split-line-at-vertices.htm>`_.

`Feature to Line
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-to-line.htm>`_.

`Frequency
<https://pro.arcgis.com/en/pro-app/tool-reference/analysis/frequency.htm>`_.

`Feature Envelope to Polygon
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-envelope-to-polygon.htm>`_.

`Densify examples
<https://stackoverflow.com/questions/64995977/generating-equidistance-points
-along-the-boundary-of-a-polygon-but-cw-ccw>`_.


**To do**

`Delete Identical
<https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/
delete-identical.htm>`_.

`Eliminate Polygon Part
<https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/
eliminate-polygon-part.htm>`_.

`Erase
<https://pro.arcgis.com/en/pro-app/latest/tool-reference/analysis/erase.htm>`_.

`Find Identical
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
find-identical.htm>`_.

`Split
<https://pro.arcgis.com/en/pro-app/latest/tool-reference/analysis/split.htm>`_.

`Unsplit line
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
unsplit-line.htm>`_.
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable-E0611  # arcpy.da or arcgisscripting.da issue
# pylint: disable=E1101  # arcpy.da issue
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import os
from textwrap import dedent

import numpy as np
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import append_fields

# from importlib import reload
# import npg

# from npg import npGeo, npg_arc_npg, npg_create, npg_overlay

from npg.npGeo import arrays_to_Geo  # Geo
from npg.npg_arc_npg import (get_SR, get_shape_K, fc_to_Geo, Geo_to_fc,
                             Geo_to_arc_shapes, fc_data)
from npg.npg_create import circle, hex_flat, hex_pointy, rectangle, triangle
from npg.npg_bool_ops import merge_  # Changed from npg_overlay

from scipy.spatial import Voronoi  # Delaunay

# import arcpy
import arcgisscripting as ags
from arcpy import (da, env, AddMessage, Exists, GetMessages, gp)  # noqa
from arcpy.management import (
    AddField, CopyFeatures, CreateFeatureclass, Delete, MakeFeatureLayer,
    MultipartToSinglepart, SelectLayerByLocation, XYToLine)
from arcpy.analysis import Clip


env.overwriteOutput = True

np.set_printoptions(
    edgeitems=10, linewidth=80, precision=3, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 7.3f}'.format})
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]
pth = os.path.split(script)[0]

# pth = "/".join(script.split("/")[:-1])


# ===========================================================================

tool_list = [
    'Attribute sort', 'Frequency and Stats',
    'Features to Points', 'Polygons to Polylines', 'Split at Vertices',
    'Vertices to Points',
    'Extent Sort', 'Geometry Sort',
    'Area Sort', 'Length Sort',
    'Fill Holes', 'Keep Holes',
    'Rotate Features', 'Shift Features', 'Dissolve Boundaries',
    'Delaunay', 'Voronoi']
cont_list = ['Bounding Circles', 'Convex Hulls', 'Extent Polys',
             'Minimum area bounding rectangle']
dens_list = ['Densify by Distance', 'Densify by Percent', 'Densify by Factor']
tile_list = ['rectangle', 'hex_pointy', 'hex_flat', 'triangle']

all_tools = sorted(tool_list + tile_list + cont_list + dens_list)
# ============================================================================
# ---- messages --------------------------------------------------------------
msg0 = """
----
Either you failed to specify the geodatabase location and filename properly
or you had flotsam, including spaces, in the path, like...\n
...  {}\n
Create a safe path and try again...\n
`Filenames and paths in Python`
<https://community.esri.com/blogs/dan_patterson/2016/08/14/filenames-and
-file-paths-in-python>`_.
----
"""

msg_mp_sp = """
----
Multipart shapes have been converted to singlepart, so view any data
carried over during the extendtable join as representing those from
the original data.  Recalculate values where appropriate.
----
"""

msg_pgon_pline = """
----
Multipart shapes have been converted to singlepart.  Inner and outer rings
(holes) have been retained, but the holes have been appended to the end.

If the featureclass is not added to the map, refresh the gdb in Catalog and
add the result to the map.
----
"""


def tweet(msg):
    """Print a message for both arcpy and python."""
    m = "\n{}\n".format(msg)
    AddMessage(m)
    print(m)


# ============================================================================
# ---- io tools --------------------------------------------------------------
#
def check_path(fc):
    r"""
    Check file name and file path for compliance.

    ---- check_path ----

    Checks for a file geodatabase and a filename. Files and/or paths containing
    `flotsam` are flagged as being invalid.

    Check the geodatabase location and filename properly.  Flotsam, in the
    path or name, consists of one or more of these characters...::

       \'!"#$%&\'()*+,-;<=>?@[]^`{|}~  including the `space`

    Create a safe path and try again...

    References
    ----------
    `Lexical analysis
    <https://docs.python.org/3/reference/lexical_analysis.html>`_.

    `Filenames and paths in Python
    <https://community.esri.com/blogs/dan_patterson/2016/08/14/filenames-and
    -file-paths-in-python>`_.
    """
    msg = dedent(check_path.__doc__)
    flotsam = '!"#$%&\'()*+,-;<=>?@[]^`~}{ '  # " ... plus the `space`"
    fail = False
    fc = (r"{}").format(fc)
    if (".gdb" not in fc) or np.any([i in fc for i in flotsam]):
        fail = True
    pth = fc.replace("\\", "/").split("/")
    name = pth[-1]
    if (len(pth) == 1) or (name[-4:] == ".gdb"):
        fail = True
    if fail:
        # tweet(msg)
        return (None, None)
    gdb = "/".join(pth[:-1])
    return gdb, name


def _in_(in_fc, info):
    """Produce the Geo array.

    Parameters
    ----------
    in_fc : text
        Path to the featureclass in the geodatabase.
    info : text
        Option information to provide to the inputs.

    Notes
    -----
    syntax :
        fc_to_Geo(in_fc, geom_kind=2, minX=0, minY=0, sp_ref=None, info="")
    - in_fc : featureclass name
    - geom_kind : polygon (2), polyline (1)
    - minX, minY : coordinates of the lower left corner of the feature extent
    - sp_ref : spatial reference name, code or equivalent
    - info : optional text
    """
    SR = get_SR(in_fc)
    shp_kind, k = get_shape_K(in_fc)
    g = fc_to_Geo(in_fc, geom_kind=k, sp_ref=SR.name, info=info)
    m = g.LL  # the lower left of the Geo array, which has been shifted
    oids = g.shp_ids
    return g, oids, shp_kind, k, m, SR


def _out_(shps, gdb, name, out_kind, SR):
    """Output the FeatureClass from the Geo array.

    shps : output geometry
        Poly features as esri geometry objects.
    gdb, name : string
        Geodatabase and featureclass name.
    out_kind : string
        Polygon, Polyline
    SR : spatial referencnce
       Spatial reference object name or kind.
    """
    Geo_to_fc(shps, gdb=gdb, name=name, kind=out_kind, SR=SR)


def _extend_table_(shps, gdb, name, in_fc):
    """Return joined data."""
    out = gdb + "\\" + name
    if Exists(out):
        d = fc_data(in_fc)
        import time
        time.sleep(1.0)
        da.ExtendTable(out, 'OBJECTID', d, 'OID_', append_only=False)
    return None


def temp_fc(geo, name, kind, SR):
    """Similar to `_out_` but creates a `memory` featureclass."""
    polys = Geo_to_arc_shapes(geo, as_singlepart=True)
    wkspace = env.workspace = 'memory'  # legacy is in_memory
    tmp_name = "{}\\{}".format(wkspace, name)
    # tmp = MultipartToSinglepart(in_fc, r"memory\in_fc_temp")
    if Exists(tmp_name):
        Delete(tmp_name)
    CreateFeatureclass(wkspace, name, kind, spatial_reference=SR)
    AddField(tmp_name, 'ID_arr', 'LONG')
    with ags.da.InsertCursor(name, ['SHAPE@', 'ID_arr']) as cur:
        for row in polys:
            cur.insertRow(row)
    return tmp_name


# ============================================================================
# ---- Attribute Tools -------------------------------------------------------
# attribute sort
def attr_sort(a, oid_fld=None, sort_flds=None, out_fld=None):
    """Return old and new id values for the sorted array."""
    idx = np.argsort(a, order=sort_flds)
    srted = a[idx]
    dt = [(oid_fld, '<i4'), (out_fld, '<i4')]
    out = np.zeros_like(srted, dtype=np.dtype(dt))  # create the new array
    out[oid_fld] = srted[oid_fld]
    out[out_fld] = np.arange(0, out.shape[0])
    return out


# frequency and statistics
def freq(a, cls_flds, stat_fld):
    """Frequency and crosstabulation.

    Parameters
    ----------
    a : array
        A structured array.
    cls_flds : field name
        Fields to use in the analysis, their combination representing the case.
    stat_fld : field name
        The field to provide summary statistics for.

    Notes
    -----
    1. Slice the input array by the classification fields.
    2. Sort the sliced array using the flds as sorting keys.
    3. Use unique on the sorted array to return the results and the counts.

    >>> np.unique(ar, return_index=False, return_inverse=False,
    ...           return_counts=True, axis=None)
    """
    if stat_fld is None:
        a = a[cls_flds]  # (1) It is actually faster to slice the whole table
    else:
        all_flds = cls_flds + [stat_fld]
        a = a[all_flds]
    idx = np.argsort(a, axis=0, order=cls_flds)  # (2)
    a_sort = a[idx]
    uni, inv, cnts = np.unique(a_sort[cls_flds], False,
                               True, return_counts=True)  # (3)
    out_flds = "Counts"
    out_data = cnts
    if stat_fld is not None:
        splitter = np.where(np.diff(inv) == 1)[0] + 1
        a0 = a_sort[stat_fld]
        splits = np.split(a0, splitter)
        sums = np.asarray([np.nansum(i.tolist()) for i in splits])
        nans = np.asarray([np.sum(np.isnan(i.tolist())) for i in splits])
        mins = np.asarray([np.nanmin(i.tolist()) for i in splits])
        means = np.asarray([np.nanmean(i.tolist()) for i in splits])
        maxs = np.asarray([np.nanmax(i.tolist()) for i in splits])
        out_flds = [out_flds, stat_fld + "_sums", stat_fld + "_NaN",
                    stat_fld + "_min", stat_fld + "_mean", stat_fld + "_max"]
        out_data = [out_data, sums, nans, mins, means, maxs]
    out = append_fields(uni, names=out_flds, data=out_data, usemask=False)
    return out


# ============================================================================
# ---- Container Tools -------------------------------------------------------
#
# bounding circles
def circles(in_fc, gdb, name, out_kind):
    """Minimum area bounding circles.

    Change `angle=2` to a smaller value for denser points on circle perimeter.

    `getSR`, `get_shape_K` and `fc_geometry` are from `npg_io`.
    """
    info = "bounding circles"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    shps = g.bounding_circles(angle=2, shift_back=True, return_xyr=False)
    _out_(shps, gdb, name, out_kind, SR)
    return "{} completed".format("Circles")


# convex hulls
def convex_hull_polys(in_fc, gdb, name, out_kind):
    """Determine the convex hulls on a shape basis."""
    info = "convex hulls to polygons"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    shps = g.convex_hulls(by_bit=False, shift_back=True, threshold=50)
    _out_(shps, gdb, name, out_kind, SR)
    return "{} completed".format("Convex Hulls")


# extent_poly section
def extent_poly(in_fc, gdb, name, out_kind):
    """Feature envelope to polygon demo."""
    info = "extent to polygons"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    shps = g.extent_rectangles(shift_back=True, asGeo=True)
    _out_(shps, gdb, name, out_kind, SR)
    return "{} completed".format("Extents")


def mabr(in_fc, gdb, name, out_kind):
    """Return Minumum Area Bounding Rectangle."""
    info = "minimum area bounding rectangle"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    shps = g.min_area_rect(shift_back=True, as_structured=False)
    # LBRT = shps[:, 1:]
    _out_(shps, gdb, name, out_kind, SR)
    return "{} completed".format("Minimum area bounding rectangle")


# ============================================================================
# ---- Conversion Tools ------------------------------------------------------
#
# features to point
def f2pnts(in_fc):
    """Features to points.

    `getSR`, `get_shape_K` and `fc_geometry` are from `npg_io`.
    """
    info = "feature to points"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    cent = g.centroids() + m
    dt = np.dtype([('OID_', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')])
    out = np.empty((len(cent), ), dtype=dt)
    out['OID_'] = oids
    out['Xs'] = cent[:, 0]
    out['Ys'] = cent[:, 1]
    return out, SR  # featureclass creation handled elsewhere


# polygon to polyline
def pgon_to_pline(in_fc, gdb, name):
    """Polygon to polyline conversion.

    Multipart shapes are converted to singlepart.  Holes are re-oriented and
    appended to the end of the outer-rings. The combined geometry is used to
    produce the polylines.
    """
    info = "pgon to pline"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    dx, dy = LL = g.LL
    z = g.outer_rings()
    z1 = g.holes_to_shape()
    z2 = merge_(z1, z)
    z2.XT = g.XT - LL
    z2 = z2.translate(dx, dy)
    z2.K = 1
    out_kind = "Polyline"
    _out_(z2, gdb, name, out_kind, SR)
    # extend_table_(shps, gdb, name, in_fc)
    # if Exists(out):
    #     d = fc_data(in_fc)
    #     import time
    #     time.sleep(1.0)
    #     da.ExtendTable(out, 'OBJECTID', d, 'OID_', append_only=False)
    tweet(dedent(msg_pgon_pline))
    return None


# split line at vertices
def split_at_vertices(in_fc, out_fc):
    """Split at vertices.  Unique segments retained."""
    info = "split at vertices"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    od = g.segment_polys(as_basic=False, shift_back=True, as_3d=False)
    tmp = "memory/tmp"
    if Exists(tmp):
        Delete(tmp)
    ags.da.NumPyArrayToTable(od, tmp)
    xyxy = list(od.dtype.names[:4])
    args = [tmp, out_fc] + xyxy + ["GEODESIC", "Orig_id", SR]
    XYToLine(*args)
    return None


# vertices to points
def p_uni_pnts(in_fc):
    """Implement `_polys_to_unique_pnts_` in `npg_helpers`."""
    info = "unique points"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    out = g.polys_to_points(keep_order=True, as_structured=True)
    return out, SR


# ============================================================================
# ---- Sort Geometry ---------------------------------------------------------
#
# sort by area, length
def sort_geom(in_fc, gdb, name, sort_kind):
    """Sort features by area, length for full shape."""
#    tmp = MultipartToSinglepart(in_fc, r"memory\in_fc_temp")
    info = "sort features"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)  # tmp, info)
    if sort_kind == 'area':
        srt = g.sort_by_area(ascending=True, just_indices=False)
    elif sort_kind == 'length':
        srt = g.sort_by_length(ascending=True, just_indices=False)
    out_kind = shp_kind
    x, y = g.LL
    srt = srt.translate(dx=x, dy=y)
    _out_(srt, gdb, name, out_kind, SR)
    return None


# sort by extent
def sort_extent(in_fc, gdb, name, key):
    """Sort features by extent."""
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="sort features")
    srt = g.sort_by_extent(extent_pnt='LB', key=key, just_indices=False)
    out_kind = shp_kind
    x, y = g.LL
    srt = srt.translate(dx=x, dy=y)
    _out_(srt, gdb, name, out_kind, SR)
    return None


# ============================================================================
# ---- Alter Geometry --------------------------------------------------------
#

def dens_dist(in_fc, gdb, name, dist):
    """Densify by distance."""
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="densify by distance")
    if dist is None:
        dist = 1.
    elif dist < 1. or dist > 100.:  # limit of 1 to 100%
        dist = np.abs(min(dist, 100.))
    dens = g.densify_by_distance(spacing=dist)
    out_kind = shp_kind
    x, y = g.LL
    dens = dens.translate(dx=x, dy=y)
    _out_(dens, gdb, name, out_kind, SR)
    return None


def dens_fact(in_fc, gdb, name, dist):
    """Densify by factor."""
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="densify by distance")
    if dist is None:
        dist = 1.
    dens = g.densify_by_factor(factor=dist)
    out_kind = shp_kind
    x, y = g.LL
    dens = dens.translate(dx=x, dy=y)
    _out_(dens, gdb, name, out_kind, SR)
    return None


def dens_perc(in_fc, gdb, name, dist):
    """Densify by percent."""
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="densify by percent")
    if dist is None:
        dist = 100.
    dens = g.densify_by_percent(percent=dist, is_percent=True)
    out_kind = shp_kind
    x, y = g.LL
    dens = dens.translate(dx=x, dy=y)
    _out_(dens, gdb, name, out_kind, SR)
    return None


def fill_holes(in_fc, gdb, name):
    """Fill holes in a featureclass.  See the Eliminate part tool."""
    SR = get_SR(in_fc)
    # tmp = MultipartToSinglepart(in_fc, r"memory\in_fc_temp")
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="fill holes")  # in_fc = tmp
    if g.is_multipart():
        g = g.multipart_to_singlepart(info="")
    o_rings = g.outer_rings(True)
    out_kind = shp_kind
    x, y = g.LL
    o_rings = o_rings.translate(dx=x, dy=y)
    _out_(o_rings, gdb, name, out_kind, SR)
    # _extend_table_(shps, gdb, name, in_fc)
    # out = "{}/{}".format(gdb, name)
    # if Exists(out):
    #     import time
    #     time.sleep(1.0)
    #     d = fc_data(tmp)
    #     da.ExtendTable(out, 'OBJECTID', d, 'OID@')
    return None


def keep_holes(in_fc, gdb, name):
    """Fill holes in a featureclass.  See the Eliminate part tool."""
    SR = get_SR(in_fc)
    # tmp = MultipartToSinglepart(in_fc, r"memory\in_fc_temp")
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="keep holes")  # in_fc = tmp
    if g.is_multipart():
        g = g.multipart_to_singlepart(info="")
    i_rings = g.inner_rings(True)
    out_kind = shp_kind
    x, y = g.LL
    i_rings = i_rings.translate(dx=x, dy=y)
    _out_(i_rings, gdb, name, out_kind, SR)
    # out = "{}/{}".format(gdb, name)
    # if Exists(out):
    #     import time
    #     time.sleep(1.0)
    #     d = fc_data(tmp)
    #     da.ExtendTable(out, 'OBJECTID', d, 'OID@')
    return None


def rotater(in_fc, gdb, name, as_group, angle, clockwise):
    """Rotate features separately or as a group."""
    SR = get_SR(in_fc)
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="rotate features")
    if g.is_multipart():
        g = g.multipart_to_singlepart(info="")
    g0 = g.rotate(as_group=as_group, angle=angle, clockwise=clockwise)
    out_kind = shp_kind
    x, y = g.LL
    g0 = g0.translate(dx=x, dy=y)
    _out_(g0, gdb, name, out_kind, SR)
    # out = "{}/{}".format(gdb, name)
    # if Exists(out):
    #     import time
    #     time.sleep(1.0)
    #     d = fc_data(tmp)
    #     da.ExtendTable(out, 'OBJECTID', d, 'OID_')
    return None


def shifter(in_fc, gdb, name, dX, dY):
    """Shift features to a new location by delta X and Y values.

    Multipart shapes are converted to singlepart shapes.
    """
    # tmp = MultipartToSinglepart(in_fc, r"memory\in_fc_temp")  # in_fc = tmp
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info="shift features")
    if g.is_multipart():
        g = g.multipart_to_singlepart(info="")
    out_kind = shp_kind
    x, y = g.LL
    g0 = g.translate(x + dX, y + dY)
    _out_(g0, gdb, name, out_kind, SR)
    # out = "{}/{}".format(gdb, name)
    # if Exists(out):
    #     import time
    #     time.sleep(1.0)
    #     d = fc_data(tmp)
    #     da.ExtendTable(out, 'OBJECTID', d, 'OID_', append_only=False)
    return None


def dissolve_boundaries(in_fc, gdb, name):
    """Dissolve shared boundaries between shapes."""
    info = "boundary dissolve"
    g, oids, shp_kind, k, m, SR = _in_(in_fc, info)
    out_kind = shp_kind
    x, y = g.LL
    g0 = dissolve(g, asGeo=True)
    tweet("g0")
    # g0 = arrays_to_Geo(g0, kind=2, info="extent")
    g0 = g0.translate(dx=x, dy=y)
    _out_(g0, gdb, name, out_kind, SR)


# ============================================================================
# ---- Triangulation Tools ---------------------------------------------------
#
def tri_poly(in_fc, gdb, name, out_kind, constrained=True):
    """Return the Delaunay triangulation of the poly* features."""
    tmp = MultipartToSinglepart(in_fc, r"memory\in_fc_temp")  # in_fc = tmp
    g, oids, shp_kind, k, m, SR = _in_(tmp, info="triangulate")
    out = g.triangulate(as_one=False, as_polygon=True)
    x, y = g.LL
    g0 = out.translate(dx=x, dy=y)
    if constrained:
        tmp_name = temp_fc(g0, "tmp", shp_kind, SR)  # make temp featureclass
        tmp_lyr = MakeFeatureLayer(tmp_name)
        SelectLayerByLocation(
            tmp_lyr, "WITHIN", tmp, None, "NEW_SELECTION", "NOT_INVERT")
        out_name = gdb.replace("\\", "/") + "/" + name
        CopyFeatures(tmp_lyr, out_name)
    else:
        _out_(g0, gdb, name, shp_kind, SR)
    return None


def vor_poly(in_fc, gdb, name, out_kind):
    """Return the Voronoi/Theissen poly* features."""
    tmp = MultipartToSinglepart(in_fc, r"memory\in_fc_temp")
    g, oids, shp_kind, k, m, SR = _in_(tmp, info="voronoi")
    out = []
    L, B, R, T = g.aoi_extent()  # full extent for infinity circle
    xc, yc = np.array([(R - L) / 2., (T - B) / 2.])
    radius = max((R - L), (T - B)) * 10
    inf_circ = circle(radius, xc=xc, yc=yc)
    pnts = [np.vstack((g.XY, inf_circ))]
    for ps in pnts:
        if len(ps) > 2:
            ps = np.unique(ps, axis=0)
        avg = np.mean(ps, axis=0)
        p = ps - avg
        tri = Voronoi(p)
        for region in tri.regions:
            if -1 not in region:
                polygon = np.array([tri.vertices[i] + avg for i in region])
                if len(polygon) >= 3:
                    out.append(polygon)
    x, y = g.LL
    out = arrays_to_Geo(out, kind=2, info="voronoi")
    g0 = out.translate(dx=x, dy=y)
    #
    tmp = temp_fc(g0, "tmp", shp_kind, SR)
    ext_poly = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
    ext_poly = arrays_to_Geo([ext_poly], kind=2, info="extent")
    ext_poly = ext_poly.translate(dx=x, dy=y)
    tmp1 = temp_fc(ext_poly, "tmp1", shp_kind, SR)
    final = gdb.replace("\\", "/") + "/" + name
    Clip(tmp, tmp1, final)  # arcpy.analysis.Clip
    return None


# ============================================================================
# ---- Tile Creation ---------------------------------------------------------
#
# from npg.npg_create import circle, hex_flat, hex_pointy, rectangle, triangle
# tile_list = ['rectangle', 'hex_pointy', 'hex_flat', 'triangle']

# ---- =======================================================================
# ---- pick tool section
# : testing or tool run
# tbx = pth + "/npGeom.tbx"
# tbx = ImportToolbox(tbx)

f0 = """
-------------------
Source script... {}
Using :
    tool   : {}
    input  : {}
    output : {}
-------------------
"""

f1 = """
------------------
Sorting      : {}
Using fields : {}
Output field : {}
-----------------
"""


def pick_tool(tool, in_fc, out_fc, gdb, name):
    """Pick the tool and run the option."""
    # --
    #
    tweet(dedent(f0).format(script, tool, in_fc, out_fc))
    #
    # ---- Attribute tools
    if tool == 'Attribute sort':                  # ---- (1) attribute sort
        sort_flds = str(sys.argv[4])
        out_fld = str(sys.argv[5])
        sort_flds = sort_flds.split(";")
        tweet(dedent(f1).format(in_fc, sort_flds, out_fld))
        oid_fld = ags.da.Describe(in_fc)['OIDFieldName']
        flds = [oid_fld] + sort_flds
        a = ags.da.TableToNumPyArray(in_fc, flds)
        out = attr_sort(a, oid_fld, sort_flds, out_fld)  # run... attr_sort
        ags.da.ExtendTable(in_fc, oid_fld, out, oid_fld, append_only=False)
    elif tool == 'Frequency and Stats':           # ---- (2) freq and stats
        cls_flds = sys.argv[4]
        stat_fld = sys.argv[5]
        cls_flds = cls_flds.split(";")  # multiple to list, singleton a list
        if stat_fld in (None, 'NoneType', ""):
            stat_fld = None
        # use the whole array and skip nulls
        if stat_fld is not None:
            all_flds = cls_flds + [stat_fld]
        a = ags.da.TableToNumPyArray(
            in_fc, field_names=all_flds, skip_nulls=True)
        out = freq(a, cls_flds, stat_fld)         # do freq analysis
        if Exists(out_fc) and env.overwriteOutput:
            Delete(out_fc)
        ags.da.NumPyArrayToTable(out, out_fc)
    #
    # ---- Containers
    #
    elif tool in cont_list:
        out_kind = sys.argv[4].upper()
        c_d = {
            'Bounding Circles': circles(in_fc, gdb, name, out_kind),
            'Convex Hulls': convex_hull_polys(in_fc, gdb, name, out_kind),
            'Extent Polys': extent_poly(in_fc, gdb, name, out_kind),
            'Minimum area bounding rectangle': mabr(in_fc, gdb, name, out_kind)
            }
        c_d[tool]  # run the tool
    #
    # ---- Conversion
    #
    elif tool in ['Features to Points', 'Vertices to Points']:
        if tool == 'Features to Points':         # ---- (1) features to point
            out, SR = f2pnts(in_fc)
        elif tool == 'Vertices to Points':       # ---- (2) feature to vertices
            out, SR = p_uni_pnts(in_fc)
        ags.da.NumPyArrayToFeatureClass(out, out_fc, ['Xs', 'Ys'], SR)
    elif tool == 'Polygons to Polylines':        # ---- (3) polygon to polyline
        pgon_to_pline(in_fc, gdb, name)
    elif tool == 'Split at Vertices':            # ---- (4) split at vertices
        split_at_vertices(in_fc, out_fc)
    #
    # ---- Sort geometry
    #
    elif tool in ['Area Sort', 'Length Sort', 'Geometry Sort']:
        srt_type = tool.split(" ")[0].lower()
        tweet("...\n{} as {}".format(tool, 'input'))
        sort_geom(in_fc, gdb, name, srt_type)
    elif tool == 'Extent Sort':
        srt_type = int(sys.argv[4][0])
        tweet("...\n{} as {}".format(tool, 'input'))
        sort_extent(in_fc, gdb, name, srt_type)
    #
    # ---- Alter geometry
    #
    elif tool in dens_list:
        dist = float(sys.argv[4])
        d_d = {'Densify by Distance': dens_dist(in_fc, gdb, name, dist),
               'Densify by Percent': dens_perc(in_fc, gdb, name, dist),
               'Densify by Factor': dens_fact(in_fc, gdb, name, dist)
               }
        d_d[tool]  # run the tool
    elif tool == 'Fill Holes':                   # ---- (4) fill holes
        fill_holes(in_fc, gdb, name)
    elif tool == 'Keep Holes':                   # ---- (5) keep holes
        keep_holes(in_fc, gdb, name)
    elif tool == 'Rotate Features':              # ---- (6) rotate
        clockwise = False
        as_group = False
        rot_type = str(sys.argv[4])  # True: extent center. False: shape center
        angle = float(sys.argv[5])
        clockwise = str(sys.argv[6])
        if rot_type == "shape center":
            as_group = True
        if clockwise.lower() == "true":
            clockwise = True
        rotater(in_fc, gdb, name, as_group, angle, clockwise)
    elif tool == 'Shift Features':               # ---- (7) shift
        dX = float(sys.argv[4])
        dY = float(sys.argv[5])
        shifter(in_fc, gdb, name, dX=dX, dY=dY)
    elif tool == 'Dissolve Boundaries':
        dissolve_boundaries(in_fc, gdb, name)
    #
    # ---- Triangulation
    elif tool == 'Delaunay':                     # ---- (1) Delaunay
        out_kind = sys.argv[4].upper()
        constrained = sys.argv[5]
        if constrained == "True":
            constrained = True
        else:
            constrained = False
        tri_poly(in_fc, gdb, name, out_kind, constrained)
    elif tool == 'Voronoi':                      # ---- (2) Voronoi
        out_kind = sys.argv[4].upper()
        vor_poly(in_fc, gdb, name, out_kind)
    #
    # ---- Tiling
    elif tool in tile_list:
        out_fc = sys.argv[2]      # full featureclass path and name
        SR = sys.argv[3]          # spatial reference
        out_kind = sys.argv[4]         # 1 Polyline, 2 Polygon
        dx = float(sys.argv[5])        # x distance/increment
        dy = float(sys.argv[6])        # y distance/increment
        x_c = int(sys.argv[7])         # x columns
        y_r = int(sys.argv[8])         # y rows
        o_x = float(sys.argv[9])       # x origin
        o_y = float(sys.argv[10])      # y origin
        # rot_angle = sys.argv[11]
        k = 2 if out_kind == 'Polygon' else 1
        tool = str(sys.argv[11])
        t_d = {
            'hex_flat': hex_flat(dx, -dy, x_c, y_r, o_x, o_y, True, k),
            'hex_pointy': hex_pointy(dx, -dy, x_c, y_r, o_x, o_y, True, k),
            'rectangle': rectangle(dx, -dy, x_c, y_r, o_x, o_y, True, k),
            'triangle': triangle(dx, -dy, x_c, y_r, o_x, o_y, True, k)
            }
        a = t_d[tool]  # pick and use the appropriate function
        tweet("type {}\n{}".format(type(a), a[:10]))
        msg = "gdb   : {}\name : {}\nSR   : {}\ntype : {}"
        tweet(msg.format(gdb, name, SR, out_kind))
        _out_(a, gdb, name, k, SR)  # produce the output
        return None
    else:
        tweet("Tool {} not found".format(tool))
        return None
    # --


# ---- == testing or tool run ================================================
#
def _testing_():
    """Run in spyder."""
    pth = "C:/arcpro_npg"
    in_fc = "C:/arcpro_npg/Project_npg/npgeom.gdb/sq2"
    out_fc = "C:/arcpro_npg/Project_npg/tests.gdb/x"
    # tbx = pth + "/npGeom_30.atbx"  # can't import a *.atbx
    # tbx = ImportToolbox(tbx)
    # tool = 'ShiftFeatures'  # None  #
    # info_ = gp.getParameterInfo(tool)
    # for param in info_:
    #     print("Name: {}, Type: {}, Value: {}".format(
    #         param.name, param.parameterType, param.value))
    print("tbx    : {}\nInput  : {}\nOutput : {} ".format(pth, in_fc, out_fc))
    return in_fc, out_fc  # in_fc, out_fc, tool, kind


def _tool_(all_tools):
    """Run from a tool in arctoolbox in ArcGIS Pro.

    The tool checks to ensure that the path to the output complies and that
    the desired tool actually exists, so it can be parsed based on type.
    """
    tool = sys.argv[1]
    if tool not in tile_list:  # -- tile_list doesn't require an input FC
        in_fc = sys.argv[2]
        out_fc = sys.argv[3]
    else:
        in_fc = None
        out_fc = sys.argv[2]
    # tweet("out_fc  {}".format(out_fc))
    if out_fc not in (None, 'None'):
        gdb, name = check_path(out_fc)           # ---- check the paths
        if gdb is None:
            tweet(dedent(msg0))
            return None
    else:
        gdb = None
        name = None
    if tool not in all_tools:                    # ---- check the tool
        tweet("Tool {} not implemented".format(tool))
        return None
    msg1 = "Tool   : {}\ninput  : {}\noutput : {}"
    tweet(msg1.format(tool, in_fc, out_fc))
    pick_tool(tool, in_fc, out_fc, gdb, name)    # ---- run the tool
    return tool, in_fc, out_fc


# ===========================================================================
# ---- main section: testing or tool run

if len(sys.argv) == 1:
    testing = True
    result = _testing_()
else:
    testing = False
    AddMessage("=hello1")  # arcpy.AddMessage, AddMessage imported directly
    AddMessage(r"<hello2>")
    AddMessage("<{}>".format("hello3"))
    tool, in_fc, out_fc = _tool_(all_tools)
    msg = "Outside...\nTool   : {}\ninput  : {}\noutput : {}"
    tweet(msg.format(tool, in_fc, out_fc))
    GetMessages()

# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    info_ = _testing_()

    # in_fc = "C:/arcpro_npg/Project_npg/tests.gdb/sq2"
    # out_fc = "C:/arcpro_npg/Project_npg/tests.gdb/x0"
