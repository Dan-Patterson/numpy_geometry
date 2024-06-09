# -*- coding: utf-8 -*-
# flake8 --per-file-ignores="__init__.py:F401
# noqa: E401, E402, D205, D400, F401, F403
# pylint: disable=C0410
r"""
npg  NumPy Geometry
===================

**npg module __init__ file**

Normal usage:

    >>> import npg

Script :
    __init__.py.

Author :
    Dan Patterson
- Dan_Patterson@carleton.ca
- https://github.com/Dan-Patterson

Modified : 2024-05-04
    Creation date during 2019 as part of ``arraytools``.

Purpose
-------
Tools for working with point and poly features as an array class.
Requires npGeo to implement the array geometry class.

See Also
--------
Many links and references in ``..docs/_npgeom_notes_.txt``.

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
# noqa: E401, F401, F403, C0410
# pyflakes: disable=E0401,F403,F401
# pylint: disable=unused-import
# pylint: disable=E0401

# ---- sys, np imports
import sys
import numpy as np

# ---- import for npg
# import npg

from . import npgDocs, npGeo, npg_geom_hlp, npg_geom_ops
from . import npg_bool_hlp, npg_bool_ops
from . import npg_min_circ, npg_overlay, npg_analysis
from . import npg_setops, npg_io, npg_table, npg_create, npg_utils
from . import npg_pip, npg_prn  # noqa
# --
# requires arcpy
# npg_create, npg_arc_npg, npg_geom_ops, npg_io

from . npGeo import *  # noqa
from . npg_geom_hlp import *  # noqa
from . npg_geom_ops import *  # noqa
from . npg_bool_hlp import *  # noqa
from . npg_bool_ops import *  # noqa
from . npg_clip_split import clip_poly, split_poly  # noqa
from . npg_io import *  # noqa

# from . npg_pip import *
# from . npg_min_circ import *
# from . npg_overlay import *
# from . npg_analysis import *
# from . npg_setops import *
# from . npg_prn import *
# from . npg_table import *
# from . npg_create import *
# from . npg_utils import *

# ---- docstring info for Geo and some methods


# ---- define __all__
__all__ = [
    'npgDocs', 'npGeo', 'npg_io', 'npg_geom_ops', 'npg_geom_hlp',
    'npg_bool_hlp', 'npg_bool_ops',
    'npg_overlay', 'npg_table', 'npg_create', 'npg_analysis', 'npg_utils',
    'npg_setops',  'npg_min_circ'
]

__helpers__ = [
    'npGeo_doc', 'Geo_hlp', 'array_IFT_doc', 'dirr_doc', 'shapes_doc',
    'parts_doc', 'outer_rings_doc', 'inner_rings_doc', 'get_shapes_doc',
    'sort_by_extent_doc', 'radial_sort_doc'
]

__all__.extend(npgDocs.__all__)
__all__.extend(npGeo.__all__)
__all__.extend(npg_geom_ops.__all__)
__all__.extend(npg_geom_hlp.__all__)
__all__.extend(npg_bool_ops.__all__)
__all__.extend(npg_bool_hlp.__all__)
__all__.sort()

__helpers__.extend(npg_geom_hlp.__helpers__)
__helpers__.extend(npg_geom_ops.__helpers__)
__helpers__.extend(npg_bool_ops.__helpers__)
__helpers__.extend(npg_bool_hlp.__helpers__)
__helpers__.sort()

"""
__all__.extend(npg.npg_io.__all__)
__all__.extend(npg_prn.__all__)
__all__.extend(npg_pip.__all__)
__all__.extend(npg_table.__all__)
__all__.extend(npg_create.__all__)
__all__.extend(npg_analysis.__all__)
__all__.extend(npg_overlay.__all__)
__all__.extend(npg_setops.__all__)
# __all__.extend(npg_min_circ.__all__)
__helpers__.extend(npg_analysis.__helpers__)
__helpers__.extend(npg_overlay.__helpers__)

"""
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
... import npg

Modules not imported by default...
... npg_arc_npg
... npg_plots

----------------------------------------------
"""

pth = __path__[0]
print(msg.format(pth, sys.version, sys.exec_prefix, np.__version__))
del pth
del msg
