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
    2023-111-14

Purpose
-------
Tools for working with point and poly features as an array class.
Requires npGeo to implement the array geometry class.

Note
----
See `npg_arc_npg.py` for bringing and saving attribute data.

See Also
--------
__init__ :
    The ``.../npg/__init__.py`` script has further information on arcpy
    related functionality.
npGeo :
    A fuller description of the Geo class, its methods and properties is given
    ``.../npg/npGeo``.  This script focuses on getting arcpy or geojson
    geometry into numpy arrays.

Example
-------
see : "C:/arcpro_npg/npg/docs/json_conversion_notes.txt"

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

from npg import npGeo

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

__all__ = [
    'dtype_info', 'load_geo', 'load_geo_attr', 'save_geo', 'load_txt',
    'save_txt', 'load_geojson', 'geojson_Geo'
    ]


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


def load_geo(f_name, extras=False, prn_info=False):
    r"""Load a well formed `npy` file representing a structured array.

    Unpack an `npz` file containing a Geo array.

    Parameters
    ----------
    f_name : text
        Full path and filename.
    extras : boolean
        If False, only the Geo array is returned, otherwise, the Geo array,
        the constituent arrays and their names are returned.
    prn_info : boolean
        If True, then run info is printed.

    Returns
    -------
    geo : Geo array
        The Geo array is created within this function and returned along with
        the base arrays, (`arrs`) and the list of array names (`names`), if
        ``suppress_extras`` is True.
    arrs : arrays
        The arrays within the npz,
    names : list
        A list of the array names which you use to slice particular arrays..


    Example
    -------
    This a sample::

        f_name = "C:/arcpro_npg/data/sq2.npz"
        sq2, arrs, names = npg.load_geo(f_name, extras=False, prn_info=False)
        # the array coordinates
        sq2
        Geo([[ 10.00,  10.00],
             [ 10.00,   0.00],
             [  1.50,   1.50],
              ...,
             [ 10.00,  10.00],
             [ 15.00,  18.00],
             [ 14.00,  10.00]])
        # -- the extra array names if `extras=True`.
        arrs
        NpzFile 'C:/arcpro_npg/data/g_arr.npz' with keys:
            g, ift, kind, extents, spatial_ref
        #
        # -- returns the list of array names inside
        arr_names = arrs.files
        ['g', 'ift', 'kind', 'extents', 'spatial_ref']
        # -- the array of bounding rectangle extents
        arrs['extents']
        array([[ 300000.00,  5000000.00],
               [ 300025.00,  5000018.00]])

        An array or arrays. The description, field names and their size of each
        are returned.

    Notes
    -----
    From above::

        arrs = np.load(f_name)
        arrs : numpy.lib.npyio.NpzFile
            The type of file.
        other properties : dir(arrs)
            'allow_pickle', 'close', 'f', 'fid', 'files', 'get', 'items',
            'iteritems', 'iterkeys', 'keys', 'pickle_kwargs', 'values', 'zip'
    """
    def _to_geo_(arrs, names):
        """Pull out the info."""
        n0, n1, n2, n3, n4 = names
        geo = npGeo.Geo(
            arrs[n0], IFT=arrs[n1], Kind=int(arrs[n2]), Extent=arrs[n3],
            SR=str(arrs[n4])
            )
        return geo

    arrs = np.load(f_name)
    names = arrs.files  # the array names
    msg = "\nLoading...{}\nArray(s) include...{}"
    geo = _to_geo_(arrs, names)
    if extras:
        if prn_info:
            print(msg.format(f_name, names))
        return geo, arrs, names
    if prn_info:
        print(msg.format(f_name, names[0]))
    return geo


def load_geo_attr(f_name, prn_info=False):
    """Load the attributes in an npy file associated with a geo array.

    Parameters
    ----------
    f_name : text
        The complete filename path and extension.
    prn_info : boolean
        If True, then run info is printed.

    Returns
    -------
    names : the list of array names.

    arrs : the arrays themselves `arr` and `fields`.

    The two arrays are the:

    - attribute data
    - field names and data type.

    Example
    -------
    >>> f_name = "C:/arcpro_npg/data/ontario_attr.npz"
    >>> names, arrs = load_geo_attr(f_name)
    >>> names  # ['arr', 'fields']
    >>> arr = arrs[names[0]]
    >>> fields = arrs[names[1]]

    See Also
    --------
    `npg_arc_npg.attr_to_npz` which is used to create the .npz file.
    """
    arrs = np.load(f_name)
    names = arrs.files  # array names
    if prn_info:
        frmt0 = "\nLoading attributes from...  {}\n\nArrays include...{}"
        print(frmt0.format(f_name, names))
        frmt1 = "\n({}) name : {}"
        for i, name in enumerate(names):
            print(frmt1.format(i, name))
        msg = """
        To use :
        >>> n0, n1 = names
        >>> n0, n1
        ... ('arrs', 'fields')
        >>> arr = arrs[n0]
        >>> flds = arrs[n1]
        """
        print(msg)
    return names, arrs


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


def load_txt(name="arr.txt", names=None, data_type=None):
    r"""Read a structured/recarray created by save_txt.

    Parameters
    ----------
    name : text
        Raw-encoding path to *.csv.  eg.  'C:/Data/csv/SampleData.csv'
    data type : dtype
        If `None`, it allows the structure to be read from the array.
    delimiter : string
        Use a comma delimiter by default.
    skip_header : int
        Number of rows to skip at the beginning.
    names : boolean
        If `True`, the first row contains the field names.
    encoding :
        Set to None to use system default or utf-8
    see np.genfromtxt for all `args` and `kwargs`.

    Example
    -------

    sample csv ::

        "IDs","Float_vals","Text_vals","Int_vals","Int_as_text","Lead_int"
        1,2.98,a,38,38,0038
        2,9.99,b,74,74,0074
        3,1.23,c,35,35,0035
        4,3.45,d,9,9,0009
        5,4.56,e,10,10,0010

    >>> dt = np.dtype([('IDs', 'i8'), ('Float_vals', 'f8'),
    ...                ('Text_vals', 'U5'), ('Int_vals', 'i8'),
    ...                 ('Int_as_text', 'U8'), ('Lead_int', 'U8')])
    >>> a = np.genfromtxt(tbl, dtype=dt, delimiter=",", names=True,
    ...                   autostrip=True, encoding='utf-8')
    array([(1,   2.98, 'a', 38, '38', '0038'),
           (2,   9.99, 'b', 74, '74', '0074'),
           (3,   1.23, 'c', 35, '35', '0035'),
           (4,   3.45, 'd',  9, '9', '0009'),
           (5,   4.56, 'e', 10, '10', '0010')],
          dtype=[('IDs', '<i8'), ('Float_vals', '<f8'),
                 ('Text_vals', '<U5'), ('Int_vals', '<i8'),
                 ('Int_as_text', '<U8'), ('Lead_int', '<U8')])
    """
    if names is not None:
        names = names
    a = np.genfromtxt(
        name, dtype=data_type, delimiter=",", names=names, autostrip=True,
        encoding=None)
    return a


def save_txt(a, name="arr.txt", sep=",", dt_hdr=True):
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
    hdr = ""
    if a.dtype.names is not None:
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
