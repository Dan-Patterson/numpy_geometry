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

Modified : 2020-09-09
    Creation date during 2019 as part of ``arraytools``.

Purpose
-------
Tools for working with point and poly features as an array class.
Requires npGeo to implement the array geometry class.

See Also
--------
Many links and references in ``_npgeom_notes_.py``.

.. note::

   This double dot, .. note:: thing produces a nice blue colored box with the
   note inside.

Notes
-----
Import suggestion and package properties and methods.  The Geo class in npGeo
provides the base class for this package.  It is based on the numpy ndarray.

>>> import npgeom as npg


**Import options for arcpy functions**

>>> import arcgisscripting as ags
... ['ContingentFieldValue', 'ContingentValue', 'DatabaseSequence', 'Describe',
... 'Domain', 'Editor', 'ExtendTable', 'FeatureClassToNumPyArray',
... 'InsertCursor', 'ListContingentValues', 'ListDatabaseSequences',
... 'ListDomains', 'ListFieldConflictFilters', 'ListReplicas', 'ListSubtypes',
... 'ListVersions', 'NumPyArrayToFeatureClass', 'NumPyArrayToTable', 'Replica',
... 'SearchCursor', 'TableToNumPyArray', 'UpdateCursor', 'Version', 'Walk'...]

>>> ags.da.FeatureClassToNumPyArray(...)  # useage

>>> from arcpy import (env, AddMessage, Exists)

>>> from arcpy.management import (
...     AddField, CopyFeatures, CreateFeatureclass, Delete, MakeFeatureLayer,
...     MultipartToSinglepart, SelectLayerByLocation, XYToLine
... )

>>> from arcpy.analysis import Clip


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
# pyflakes: disable=E0401,F403,F401
# pylint: disable=unused-import
# pylint: disable=W0611
# pylint: disable=E0401
# ---- sys, np imports
import sys
import numpy as np

# ---- import for npg
import npgDocs, npGeo, npg_io, npg_geom, npg_pip, npg_helpers, npg_table
import npg_create, npg_analysis, npg_overlay, npg_utils, npg_min_circ

from . npGeo import *  # noqa: F401, 403
from . npg_io import *  # noqa: F401
from . npg_geom import *  # noqa: F401
from . npg_pip import *  # noqa: F401
from . npg_helpers import *  # noqa: F401
from . npg_table import *  # noqa: F401
from . npg_create import *  # noqa: F401
from . npg_analysis import *  # noqa: F401
from . npg_overlay import *  # noqa: F401
from . npg_utils import *  # noqa: F401
from . npg_min_circ import *  # noqa: F401

# ---- docstring info for Geo and some methods
from . npgDocs import (
    npGeo_doc, Geo_hlp, array_IFT_doc, dirr_doc, shapes_doc, parts_doc,
    outer_rings_doc, inner_rings_doc, get_shapes_doc, sort_by_extent_doc,
    radial_sort_doc
)
npGeo.__doc__ += npGeo_doc
npGeo.Geo.__doc__ += Geo_hlp
npGeo.array_IFT.__doc__ += array_IFT_doc
npGeo.dirr.__doc__ += dirr_doc

npGeo.Geo.shapes.__doc__ += shapes_doc
npGeo.Geo.parts.__doc__ += parts_doc
npGeo.Geo.outer_rings.__doc__ += outer_rings_doc
npGeo.Geo.inner_rings.__doc__ += inner_rings_doc
npGeo.Geo.get_shapes.__doc__ += get_shapes_doc
npGeo.Geo.radial_sort.__doc__ += radial_sort_doc
npGeo.Geo.sort_by_extent.__doc__ += sort_by_extent_doc
# npGeo.Geo.IFT.__doc__ = array_IFT_doc

# ---- define __all__
__all__ = [
    'npgDocs', 'npGeo', 'npg_io', 'npg_geom', 'npg_helpers', 'npg_overlay',
    'npg_table', 'npg_create', 'npg_analysis', 'npg_utils',
    'npg_helpers', 'npg_min_circ'
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
# __all__.extend(npg_min_circ.__all__)
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
