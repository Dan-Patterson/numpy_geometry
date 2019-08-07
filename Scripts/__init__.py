# -*- coding: utf-8 -*-
"""
==================
npgeom.__init__.py
==================

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-07-29
    Creation date during 2019 as part of ``arraytools``.

Purpose : Tools for working with point and poly features as an array class
    Requires npGeo to implement the array geometry class.

Notes
-----

Import suggestion and package properties and methods.  The Geo class in npGeo
provides the base class for this package.  It is based on the numpy ndarray.

>>> import npgeom as npg

>>> npg.npGeo_io.__all__
... ['poly2array', 'load_geojson', 'Arrays_to_Geo', 'Geo_to_arrays',
... 'array_ift', '_make_nulls_', 'getSR', 'fc_composition', 'fc_data',
... 'fc_geometry', 'fc_shapes', 'getSR', 'array_poly', 'geometry_fc',
... 'prn_q', '_check', 'prn_tbl', 'prn_geo']

>>> npg.npGeo.__all__
... ['Geo', 'Update_Geo', '_angles_', '_area_centroid_', '_area_part_',
... '_ch_', '_ch_scipy', '_ch_simple_', '_nan_split_', '_o_ring_',
... '_pnts_on_line_', '_polys_to_segments_', '_polys_to_unique_pnts_']

>>> npg.npGeo_helpers.__all__
... ['_angles_', '_area_centroid_', '_area_part_', '_ch_', '_ch_scipy',
'_ch_simple_', '_nan_split_', '_o_ring_', '_pnts_on_line_',
'_polys_to_segments_', '_polys_to_unique_pnts_', '_simplify_lines_']

**Import options for arcpy functions**

>>> import arcgisscripting as ags
... ['ContingentFieldValue', 'ContingentValue', 'DatabaseSequence', 'Describe',
... 'Domain', 'Editor', 'ExtendTable', 'FeatureClassToNumPyArray',
... 'InsertCursor', 'ListContingentValues', 'ListDatabaseSequences',
... 'ListDomains', 'ListFieldConflictFilters', 'ListReplicas', 'ListSubtypes',
... 'ListVersions', 'NumPyArrayToFeatureClass', 'NumPyArrayToTable', 'Replica',
... 'SearchCursor', 'TableToNumPyArray', 'UpdateCursor', 'Version', 'Walk'...]

>>> ags.da.FeatureClassToNumPyArray(...)  # useage

Arcpy methods and properties needed::

    arcpy.Point, arcpy.Polyline, arcpy.Polygon, arcpy.Array
    arcpy.ListFields
    arcpy.management.CopyFeatures
    arcpy.da.Describe
    arcpy.da.InsertCursor
    arcpy.da.SearchCursor
    arcpy.da.FeatureClassToNumPyArray
"""

# import numpy as np
from . import npGeo_io  # noqa: F401; pylint: disable=unused-variable
from . import npGeo
from . import npGeo_helpers
"""
from .npGeo_io import (
        poly2array, Arrays_to_Geo, Geo_to_arrays, array_ift,
        _make_nulls_, getSR, fc_composition, fc_data, fc_geometry, fc_shapes,
        array_poly, geometry_fc, prn_q, _check, prn_tbl, prn_geo
        )
from .npGeo import (Geo,  Update_Geo)
from .npGeo_helpers import (
        _angles_, _area_centroid_, _area_part_, _ch_, _o_ring_, _dist_along_,
        _percent_along_, _pnts_on_line_, _polys_to_segments_,
        _polys_to_unique_pnts_, _simplify_lines_
        )
"""
__all__ = npGeo_io.__all__ + npGeo.__all__ + npGeo_helpers.__all__
__all__.sort()
print("\nUsage...\n  import npgeom as npg")
