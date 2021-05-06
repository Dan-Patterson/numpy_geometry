# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
--------
  npg_io
--------

**Input/Output related functions**

  - Load and save Geo arrays like you can with numpy arrays.
  - All required information is saved in the standard `.npz` format for easy
    packing and unpacking.
  - Loading json and GeoJSON are supported.

----

Script :
    .../npg/npg_io.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-12-20

Purpose
-------
Tools for working with point and poly features as an array class.
Requires npGeo to implement the array geometry class.

See Also
--------
__init__ :
    The ``.../npgeom/__init__.py`` script has further information on arcpy
    related functionality.
npGeo :
    A fuller description of the Geo class, its methods and properties is given
    ``.../npgeom/npGeo``.  This script focuses on getting arcpy or geojson
    geometry into numpy arrays.

Example
-------
see : r"C:\Git_Dan\npgeom\docs\json_conversion_notes.txt"

`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.
"""
# pylint: disable=C0103, C0302, C0330, C0415
# pylint: disable=E0611, E1101, E1136, E1121
# pylint: disable=R0902, R0904, R0914
# pylint: disable=W0105, W0201, W0212, W0221, W0612, W0614, W0621, W0105

import sys
import json
import numpy as np

import npGeo

# ---- Keep for now.
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# import npGeo
# from npGeo import *

# ---- Constants -------------------------------------------------------------
#
script = sys.argv[0]

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=160, precision=2, suppress=True,
    threshold=100, formatter=ft)

__all__ = ['dtype_info', 'load_geo', 'save_geo', 'load_txt', 'save_txt',
           'load_geojson', 'geojson_Geo']


# ---- (1) arrays : in and out------------------------------------------------
#
def dtype_info(a, as_string=False):
    """Return dtype information for a structured/recarray.

    Output can be tuples or strings.

    Examples
    --------
    >>> a = np.array([(0,  1,  2), (3,  4,  5), (6,  7,  8), (9, 10, 11)],
                  dtype=[('f0', '<i4'), ('f1', '<i4'), ('f2', '<i4')])
    >>> a.dtype.descr
    [('f0', '<i4'), ('f1', '<i4'), ('f2', '<i4')]
    >>> names, formats = dtype_info(b, as_string=False)
    #   ['f0', 'f1', 'f2'], ['<i4', '<i4', '<i4'])
    >>> names, formats = dtype_info(b, as_string=True)
    #   'f0, f1, f2', '<i4, <i4, <i4'
    >>> list(zip(*dtype_info(b, False)))  # to reconstruct
    [('f0', '<i4'), ('f1', '<i4'), ('f2', '<i4')]
    """
    dt = a.dtype.descr
    names = a.dtype.names
    if names is None:
        return dt
    names = list(a.dtype.names)
    formats = [i[1] for i in dt]
    if as_string and names is not None:
        names = ", ".join(names)
        formats = ", ".join(formats)
    return names, formats


def load_geo(f_name, suppress_extras=True):
    """Load a well formed `npy` file representing a structured array.

    Unpack an npz file containing a Geo array.

    Parameters
    ----------
    f_name : text
        Full path and filename.
    suppress_extras : boolean
        If False, only the Geo array is returned, otherwise, the Geo array,
        the constituent arrays and their names are returned.

    Returns
    -------
    geo : Geo array
        The Geo array is created within this function and returned along with
        the base arrays, (`arrs`) and the list of array names (`names`), if
        ``suppress_extras`` is False.
    arrs : arrays
        The arrays within the npz,
    names : list
        A list of the array names which you use to slice particular arrays..


    Example
    -------
    >>> f_name = "C:/Git_Dan/npgeom/data/g_arr.npz"
    >>> geo, arrs, names = npg.load_geo(f_name)
    >>> arr_names = arrs.files  # returns the list of array names inside
    arr0 = arrs[0]
    An array or arrays. The description, field names and their size of each
    are returned.

    Notes
    -----
    From above: arrs = np.load(f_name)
    arrs : numpy.lib.npyio.NpzFile
        The type of file.
    other properties : dir(arrs)
        'allow_pickle', 'close', 'f', 'fid', 'files', 'get', 'items',
        'iteritems', 'iterkeys', 'keys', 'pickle_kwargs', 'values', 'zip'
    """
    arrs = np.load(f_name)
    names = arrs.files  # array names
    print("\nLoading...{}\nArrays include...{}".format(f_name, names))
    frmt = "({}) name : {}\n  shape : {}\n  descr. : {}"
    for i, name in enumerate(names):
        tmp = arrs[name]
        shp = tmp.shape
        desc = tmp.dtype.descr
        print(frmt.format(i, name, shp, desc))
    n0, n1, n2, n3, n4 = names
    geo = npGeo.Geo(arrs[n0],
                    IFT=arrs[n1],
                    Kind=int(arrs[n2]),
                    Extent=arrs[n3],
                    SR=str(arrs[n4])
                    )
    if suppress_extras:
        return geo
    return geo, arrs, names


def save_geo(g, f_name, folder):
    """Save an array as an npz file.

    Parameters
    ----------
    g : Geo array
        A complete Geo array.
    f_name : text
        Filename without file extension or path.
    folder : text
        A local folder.  It will be checked for path name compliance.
    The type of data in each column is arbitrary.  It will be cast to the
    given dtype at runtime.
    """
    check = all([hasattr(g, i) for i in ['IFT', 'K', 'XT', 'SR']])
    if not check:
        print("Not a fully formed Geo array")
        return None
    IFT, K, XT, SR = [g.IFT, g.K, g.XT, g.SR]  # g is a Geo array
    folder = folder.replace("\\", "/")
    if f_name[-4:] == ".npz":
        f_name = f_name[:-4]
    out_name = "{}/{}.npz".format(folder, f_name)
    np.savez(out_name, g=g, ift=IFT, kind=K, extents=XT, spatial_ref=SR)
    print("\nGeo array saved to ... {} ...".format(out_name))
    return


def load_txt(name="arr.txt", data_type=None):
    """Read a structured/recarray created by save_txt.

    Parameters
    ----------
    dtype : data type
        If `None`, it allows the structure to be read from the array.
    delimiter : string
        Use a comma delimiter by default.
    skip_header : int
        Number of rows to skip at the beginning.
    names : boolean
        If `True`, the first row contains the field names.
    encoding :
        Set to None to use system default.
    see np.genfromtxt for all `args` and `kwargs`.

    """
    a = np.genfromtxt(
        name, dtype=data_type, delimiter=", ", names=True, autostrip=True,
        encoding=None)  # ,skip_header=1)
    return a


def save_txt(a, name="arr.txt", sep=", ", dt_hdr=True):
    """Save a NumPy structured/recarray to text.

    Parameters
    ----------
    a : array
        Input array.
    name : filename
        Output filename and path otherwise save to script folder.
    sep : separator
        Column separator, include a space if needed.
    dt_hdr : boolean
        If True, add dtype names to the header of the file.

    """
    a_names = ", ".join(a.dtype.names)
    hdr = ["", a_names][dt_hdr]  # use "" or names from input array
    s = np.array(a.tolist(), dtype=np.unicode_)
    widths = [max([len(i) for i in s[:, j]])
              for j in range(s.shape[1])]
    frmt = sep.join(["%{}s".format(i) for i in widths])
    np.savetxt(name, a, fmt=frmt, header=hdr, comments="")
    print("\nFile saved...")


# ============================================================================
# ---- (2) json section ------------------------------------------------------
# load json and geojson information
def load_geojson(pth, full=False, geometry=True):
    """Load a geojson file and convert to a Geo Array.

    The geojson is from the ``Features to JSON`` tool listed in the references.

    Parameters
    ----------
    pth : file path
        Full file path to the geojson file.
    full : boolean
        True to return a formatted geojson file.
    geometry : boolean
        True returns just the geometry of the file.

    Returns
    -------
    data : dictionary
        The full geojson dictionary of the geometry and its attributes.  The
        result is a nested dictionary::

    >>> data
    ... {'type':                                  # first feature
    ...  'crs': {'type': 'name', 'properties': {'name': 'EPSG:2951'}},
    ...  'features':
    ...     [{'type': 'Feature',
    ...       'id': 1,
    ...       'geometry': {'type':  'MultiPolygon',
    ...                    'coordinates': snip},  # coordinate values
    ...       'properties': snip }},              # attribute values from table
    ... {'type':                                  # next feature
    ...  ... repeat}

    geometry : list
        A list of lists representing the features, their parts (for multipart
        features) and inner holes (for polygons).

    Notes
    -----
    Using the Features to JSON tool in ArcGIS PRO, the .json option was used.
    - Unchecked : The output will be created as Esri JSON (.json).
    - Checked   : The output will be created in the GeoJSON format (.geojson).

    References
    ----------
    `geojson specification in detail
    <https://geojson.org/>`_.

    `Wikipedia link
    <https://en.wikipedia.org/wiki/GeoJSON>`_.

    `Features to JSON
    <https://pro.arcgis.com/en/pro-app/tool-reference/conversion/
    features-to-json.htm>`_.

    `JSON to Features
    <https://pro.arcgis.com/en/pro-app/tool-reference/conversion/
    json-to-features.htm>`_.
    """
    # import json  # required if run outside
    with open(pth) as f:
        data = json.load(f)
        type_key = pth.split(".")[-1]
    keys = list(data.keys())
    if 'features' in keys:
        shapes = data['features']
        coord_key = ['rings', 'coordinates'][type_key == 'geojson']
        coords = [s['geometry'][coord_key] for s in shapes]  # 'rings'
    if full and geometry:
        return data, coords
    if full:
        return data
    if geometry:
        return coords


def geojson_Geo(pth, kind=2, info=None, to_origin=False):
    """Convert GeoJSON file to Geo array using `npGeo.arrays_to_Geo`.

    Parameters
    ----------
    pth : string
        Full path to the geojson file.
    kind : integer
        Polygon, Polyline or Point type are identified as either 2, 1, or 0.
    info : text
        Supplementary information.
    """
    coords = load_geojson(pth)
    # a_2d, ift, extents = npGeo.array_IFT(coords)
    return npGeo.arrays_to_Geo(coords, kind=kind, info=info,
                               to_origin=to_origin)


# ===========================================================================
# -- main section
if __name__ == "__main__":
    """optional location for parameters"""
    print("\n{}".format(script))

r"""
'C:/arcpro_npg/npg/data/sq.json'
'C:/arcpro_npg/npg/data/sq.geojson'
'C:/arcpro_npg/npg/data/g.npz'
"""

"""
lists to dictionary

list1 =  [('84116', 1750),('84116', 1774),('84116', 1783),('84116',1792)]
list2 = [('84116', 1783),('84116', 1792),('84116', 1847),('84116', 1852),
         ('84116', 1853)]
Lst12 = list1 + list2
dt = [('Keys', 'U8'), ('Vals', '<i4')]
arr = np.asarray((list1 + list2), dtype=dt)
a0 =np.unique(arr)
k = np.unique(arr['Keys'])
{i : a0['Vals'][a0['Keys'] == i].tolist() for i in k}

"""
