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
    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-05-29

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

import npg
from npg import npGeo

# -- Keep for now.
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# import npGeo
# from npGeo import *

# ---- ---------------------------
# ---- Constants
#
script = sys.argv[0]

FLOATS = np.typecodes['AllFloat']  # np.typecodes.keys() to see all keys
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS


__all__ = [
    'dtype_info',                      # (1) in and out
    'load_geo',
    'load_geo_attr',
    'save_geo',
    'load_txt',
    'save_txt',
    'load_geojson',                    # (2) json section
    'geo_to_geojson',
    'geojson_to_geo',
    'get_keys',
    'prn_keys',
    'len_check',
    'lists_to_arrays',
    'nested_len'
]


# ---- ---------------------------
# ---- (1) arrays : in and out
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

    arrs = np.load(f_name, allow_pickle=True)  #
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
    >>> f_name = "C:/arcpro_npg/data/npz_npy/ontario_attrib.npz"
    >>> names, arrs = load_geo_attr(f_name)
    >>> names  # ['arr', 'fields']
    >>> arr = arrs[names[0]]
    >>> fields = arrs[names[1]]

    See Also
    --------
    `npg_arc_npg.attr_to_npz` which is used to create the .npz file.
    """
    if f_name[-3:] == "npz":
        arrs = np.load(f_name)
        names = arrs.files  # array names
        return names, arrs
    elif f_name[-3:] == "npy":
        arr = np.load(f_name)
        return arr


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
    s = np.array(a.tolist(), dtype=np.str_)  # unicode_) for versions < 2.0
    widths = [max([len(i) for i in s[:, j]])
              for j in range(s.shape[1])]
    frmt = sep.join(["%{}s".format(i) for i in widths])
    np.savetxt(name, a, fmt=frmt, header=hdr, comments="")
    print("\nFile saved...")


# ===== ======================================================================
# ---- ---------------------------
# ---- (2) json section

# load json and geojson information
def load_geojson(pth, full=True, just_geometry=False):
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
    if full and just_geometry:
        print("\nReturning full geojson and just the geometry portion")
        print("as a list")
        return data, coords
    if full:
        return data
    if just_geometry:
        return coords


def geo_to_geojson(arr, shift_back=True):
    """Save a `geo` array as a geojson.

    Parameters
    ----------
    arr : geo array

    """
    chk = npg.is_Geo(arr, verbose=True)  # is `geo`?, if not bail with message
    if not chk:
        return None
    # -- all is good, continue on
    d = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": arr.SR
                }
            },
        "features": [{}]
        }

    if shift_back:
        arr = arr.shift(arr.LL[0], arr.LL[1])
    IDs_, Fr_, To_, CL_, PID_, Bit_ = arr.IFT.T
    K_ = arr.K
    if K_ == 2:
        K_ = "Polygon"
    # prev_ = -1
    uniq_ids = np.unique(IDs_)
    # d2 = {}
    d1 = []
    for u in uniq_ids:
        rows = arr.IFT[arr.IFT[:, 0] == u]
        coords = []
        for row in rows:
            id_, fr_, to_, cl_, pid_, bit_ = row
            # print(IDs_, Fr_, To_, CL_, PID_, Bit_)
            coords.append(arr.XY[fr_:to_])
        d1.append({"type": "Feature",
                   "id": id_,
                   "geometry": {
                       "type": K_,
                       "coordinates": coords
                       }
                   })
    d["features"] = d1

    return d


def geojson_to_geo(pth, kind=2, info=None, to_origin=False):
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
    coords = load_geojson(pth, full=False, just_geometry=True)
    # a_2d, ift, extents = npGeo.array_IFT(coords)
    return npGeo.arrays_to_Geo(coords, kind=kind, info=info,
                               to_origin=to_origin)


def get_keys(data, num):
    """Return dictionary keys by level.

    Parameters
    ----------
    data : dictionary in geojson format
    num : beginning index number

    Useage
    ------
    r = get_keys(data, 0)
    """
    keys = []
    if isinstance(data, dict):
        num += 1
        for key, value in data.items():
            t = type(value).__name__
            keys.append((num, key, t))
            keys += get_keys(value, num)
    elif isinstance(data, list):
        for value in data[:1]:
            keys += get_keys(value, num)
            # num += 1
    return keys


def prn_keys(data, num=0):
    """Print the keys of a geojson."""
    keys = get_keys(data, num)
    o_array = np.asarray(keys, dtype='O')
    s0 = np.max([len(i) for i in o_array[:, 1]]) + 2
    s1 = np.max([len(i) for i in o_array[:, 2]]) + 2
    f0 = "{:<5}" + " {{!s:<{}}} {{!s:<{}}}".format(s0, s1)
    f1 = "  " + f0
    f2 = "    " + f0
    for i in o_array:
        if i[0] == 1:
            print(f0.format(*i))
        elif i[0] == 2:
            print(f1.format(*i))
        elif i[0] == 3:
            print(f2.format(*i))


def len_check(arr):
    """Check iterator lengths."""
    arr = np.asarray(arr, dtype='O').squeeze()
    if arr.shape[0] == 1:
        return False, len(arr)
    q = [len(a) == len(arr[0])      # check subarray and array lengths
         if hasattr(a, '__iter__')  # if it is an iterable
         else False                 # otherwise, return False
         for a in arr]              # for each subarray in the array
    return np.all(q), len(arr)


def lists_to_arrays(coords, out=[]):  # ** works
    """Return coordinates from a list of list of coordinates.

    Parameters
    ----------
    coords : nested lists of x,y values, as lists or ndarrays
    out : empty list

    Notes
    -----
    This is a specialty function that uses recursion.

    See Also
    --------
    `nested_len` can be used if you just want the depth of the nested lists.
    """
    if isinstance(coords, (list, np.ndarray)):
        for sub in coords:
            _is_, sze = len_check(sub)
            if _is_:
                arrs = [np.array(i) for i in sub]
                out.append(np.asarray(arrs, dtype=np.float64).squeeze())
            else:
                prt = []
                for j in sub:
                    arrs = [np.array(i) for i in j]
                    prt.append(np.asarray(arrs, dtype='O'))
                out.append(np.asarray(prt, dtype='O').squeeze())
    else:
        return out
    return out


def nested_len(obj, out=[], target_cls=list):
    """Return the lengths of nested iterables.

    Parameters
    ----------
    obj : iterable
        The iterable must be the same type as the `target_cls`.
    out : list
        The returned results in the form of a list of lists.
    target_cls : class
        A python, numpy class that is iterable.

    Notes
    -----
    Array type and depth/size::

          []         1
          [[]]       2, size 1
          [[], []]   2, size 2
          [[[]]]     3, size 1
          [[[], []]] 3, size 2

    Example
    -------
        >>> # input list            nested_len
        >>> lst = [                  idx0, idx1, count
        ...        [[1, 2, 3],      [[0, 0, 3],       two lists in list 0
        ...         [4, 5]],         [0, 1, 2],
        ...        [[1, 2]],         [1, 0, 2],       one list in list 1
        ...        [1, 2],           [2, 0], [2, 1],  no lists in list 2
        ...        [[]],             [3, 0, 0]        one list in list 3
        ...        [[1, 2, 3],       [4, 0, 3],       three lists in list 4
        ...         [4, 5],          [4, 1, 2],
        ...         [6, 7, 8]],      [4, 2, 3],
        ...        [[[1, 2, 3],      [5, 0, 2],       two lists in list 5
        ...          [4, 5]],
        ...         [[6, 7, 8]]]     [5, 1, 1]]
        ...       ]

    """

    def _len(obj):
        """Object length."""
        sub = len(obj) if isinstance(obj, target_cls) else 0
        return sub

    if not hasattr(obj, '__iter__'):
        print("\nIterable required (e.g. list/tuple")
        return None
    #
    for i, v0 in enumerate(obj):
        num = _len(v0)
        for j in range(0, num):
            k = v0[j]
            sub = [i, j]
            if isinstance(k, target_cls):
                s = _len(k)
                sub.append(s)
            out.append(sub)
    return out


"""
out = []

nl = nested_len(coords, out=[], target_cls=list)


# same as nested_len
#
out = []
sub = []
s = True
while s:
    for j in range(0, len(coords)):
        for k in range(0, len(coords[j])):
            chk = len(coords[j][k])
            out.append([j, k, chk])
    s = False
#

def depth(lst):
    d = 0
    for item in lst:
        if isinstance(item, list):
            d = max(depth(item), d)
    return d + 1



def nested_len(obj, *, target_cls=list):
    return [len(x) if isinstance(x, target_cls) else
            nested_len(x) for x in obj]


nested_len(coords, target_cls=list)
Out[9]: [2, 2, 1]

[nested_len(i, target_cls=list) for i in coords]
Out[10]: [[4, 2], [1, 4], [4]]

[nested_len(i, target_cls=list) for i in coords[0]]
Out[38]: [[5, 5, 4, 4], [5, 4]]

[nested_len(i, target_cls=list) for i in coords[0][0]]
Out[39]: [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]

def flatten(arr, list_=None):
    if list_ is None:
        list_ = []
    if isinstance(arr, list):
        for i in arr:
            if isinstance(i[0], list) and len(i) > 1:
                flatten(i, list_)
            else:
                list_.append(np.array(i))
    else:
        list_.append(np.array(arr))
    return list_

"""

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
