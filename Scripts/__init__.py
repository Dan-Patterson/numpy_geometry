# -*- coding: utf-8 -*-
r"""
npgeom
======

**npgeom module __init__ file**

Normal usage:

    >>> import npgeom as npg

Script :
    npgeom.__init__.py.

Author :
    Dan Patterson
- Dan_Patterson@carleton.ca
- https://github.com/Dan-Patterson

Modified : 2020-03-28
    Creation date during 2019 as part of ``arraytools``.

Purpose
-------
Tools for working with point and poly features as an array class.
Requires npGeo to implement the array geometry class.

See Also
--------
Many links and references in ``_npgeom_notes_.py``.

.. note::

   This double dot, note thing produces a nice blue colored box with the
   note inside.

Notes
-----
Import suggestion and package properties and methods.  The Geo class in npGeo
provides the base class for this package.  It is based on the numpy ndarray.

>>> import npgeom as npg

>>> npg.npg_io.__all__
... ['poly2array', 'load_geojson', 'arrays_to_Geo', 'Geo_to_arrays',
...  'array_ift', 'make_nulls', 'get_SR', 'fc_composition', 'fc_data',
...  'fc_geometry', 'get_shapes', 'get_SR', 'shape_to_K', 'array_poly',
...  'geometry_fc', 'prn_q', '_check', 'prn_tbl', 'prn_geo']

>>> npg.npGeo.__all__
... ['Geo', 'is_Geo', 'arrays_to_Geo', '_arr_ift_', 'Geo_to_arrays',
...  '_fill_float_array',
...  'dirr', 'geo_info', 'check_geometry', '_pnts_in_geo',
...  '_svg']

>>> npg.npg_helpers.__all__
... ['_area_bit_', '_in_extent_', '_is_ccw_', '_is_clockwise_',
...  '_is_right_side', '_length_bit_', '_pnts_in_extent_',
...  '_rotate_', '_scale_', '_translate_',
...  'compare_geom', 'crossings', 'in_out_crosses', 'interweave',
...  'keep_geom', 'line_crosses',  'poly_cross_product_',
...  'polyline_angles', 'radial_sort', 'remove_geom', 'sort_xy']

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

Spyder and conda
----------------
When using spyder, you can access conda.

Currently use IPython line magics and change your directory to where conda.exe
resides.

>>> cd C:\arc_pro\bin\Python\Scripts
>>> conda list  # will provide a listing of your packages

Note:  Python resides in... (substitute `arc_pro` for your install folder).

>>> C:\arc_pro\bin\Python\envs\arcgispro-py3

"""
# pyflakes: disable=F403
# pylint: disable=unused-import
# pylint: disable=W0611

# ---- sys, np imports
import sys
import numpy as np

# ---- import for npg
from . import (
    npgDocs, npGeo, npg_io, npg_geom, npg_pip, npg_helpers, npg_table,
    npg_create, npg_analysis, npg_overlay, npg_utils, smallest_circle,
)
from . npGeo import *
from . npg_io import *
from . npg_geom import *
from . npg_pip import *
from . npg_helpers import *
from . npg_table import *
from . npg_create import *
from . npg_analysis import *
from . npg_overlay import *
from . npg_utils import *
from . smallest_circle import *

# ---- docstring info for Geo and some methods
from . npgDocs import (
    npGeo_doc, Geo_hlp, array_IFT_doc, dirr_doc,
    outer_rings_doc, inner_rings_doc, pull_shapes_doc, polys_to_segments_doc,
    sort_by_extent_doc, radial_sort_doc
)
npGeo.__doc__ += npGeo_doc
npGeo.Geo.__doc__ += Geo_hlp
npGeo.array_IFT.__doc__ += array_IFT_doc
npGeo.dirr.__doc__ += dirr_doc

npGeo.Geo.outer_rings.__doc__ += outer_rings_doc
npGeo.Geo.inner_rings.__doc__ += inner_rings_doc
npGeo.Geo.pull_shapes.__doc__ += pull_shapes_doc
npGeo.Geo.polys_to_segments.__doc__ += polys_to_segments_doc
npGeo.Geo.radial_sort.__doc__ += radial_sort_doc
npGeo.Geo.sort_by_extent.__doc__ += sort_by_extent_doc

# ---- define __all__
__all__ = [
    'npgDocs', 'npGeo', 'npg_io', 'npg_geom', 'npg_helpers', 'npg_overlay',
    'npg_table', 'npg_create', 'npg_analysis', 'npg_overlay', 'npg_utils',
    'npg_helpers', 'smallest_circle'
]

__all__.extend(npgDocs.__all__)
__all__.extend(npGeo.__all__)
__all__.extend(npg_io.__all__)
__all__.extend(npg_geom.__all__)
__all__.extend(npg_pip.__all__)
__all__.extend(npg_helpers.__all__)
__all__.extend(npg_table.__all__)
__all__.extend(npg_create.__all__)
__all__.extend(npg_analysis.__all__)
__all__.extend(npg_overlay.__all__)
# __all__.extend(smallest_circle.__all__)
__all__.sort()

msg = """
----------------------------------------------
---- ... (n)um(p)y (g)eometry ... npg ... ----
location
... {}
python version and location ...
... {}
... {}
numpy version ...
... {}
Usage...
... import npgeom as npg

Modules not imported by default...
... npg_arc_npg
... npg_plots

----------------------------------------------
"""
pth = __path__[0]
print(msg.format(pth, sys.version, sys.exec_prefix, np.__version__))
del msg
