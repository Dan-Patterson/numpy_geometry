# -*- coding: utf-8 -*-
r"""
--------
  npg_io
--------

**Input/Output related functions**

Load and save Geo arrays like you can with numpy arrays.  All required
information is saved in the standard `.npz` format for easy packing and
unpacking.  Json and GeoJSON are supported.  Specialized print functions
facilitate the viewing of the Geo array geometry and attribute information.

----

Script :
    .../npgeom/npg_io.py

Author :
    Dan_Patterson@carleton.ca

Modified : 2020-09-17

Purpose
-------
Tools for working with point and poly features as an array class.
Requires npGeo to implement the array geometry class.

See Also
--------
__init__ :
    The `.../npgeom/__init__.py` script has further information on arcpy
    related functionality.
npGeo :
    A fuller description of the Geo class, its methods and properties is given
    `.../npgeom/npGeo`.  This script focuses on getting arcpy or geojson
    geometry into numpy arrays.

References
----------
**General**

`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.
"""
# pylint: disable=C0103, C0302, C0330, C0415
# pylint: disable=E0611, E1101, E1136, E1121
# pylint: disable=R0902, R0904, R0914
# pylint: disable=W0105, W0201, W0212, W0221, W0612, W0614, W0621, W0105

import sys
from textwrap import indent, dedent
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
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=160, precision=2, suppress=True,
    threshold=100, formatter=ft
)

__all__ = [
    'dtype_info', 'load_geo', 'save_geo', 'load_txt', 'save_txt',
    'load_geojson', 'geojson_Geo', 'prn_q', 'prn_', 'prn_tbl', 'prn_geo',
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
        The type of file
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
    a = np.genfromtxt(name, dtype=data_type,
                      delimiter=",",
                      names=True,
                      autostrip=True,
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
    keys = list(data.keys())
    if 'features' in keys:
        shapes = data['features']
        coords = [s['geometry']['rings'] for s in shapes]
    if full and geometry:
        return data, coords
    if full:
        return data
    if geometry:
        return coords


def geojson_Geo(pth, kind=2, info=None):
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
    return npGeo.arrays_to_Geo(coords, kind=kind, info=info)
    # return npGeo.Geo(a_2d, IFT=ift, Extent=extents, Kind=kind)


# ============================================================================
# ---- (3) Print etc ---------------------------------------------------------
# printing based on arraytools.frmts.py using prn_rec and dependencies

def prn_q(a, edges=5, max_lines=25, width=120, decimals=2):
    """Format an array so that it wraps.

    An ndarray is changed to a structured array.
    """
    width = min(len(str(a[0])), width)
    with np.printoptions(edgeitems=edges, threshold=max_lines, linewidth=width,
                         precision=decimals, suppress=True, nanstr='-n-'):
        print("\nArray fields/values...:")
        if a.dtype.kind == 'V':
            print("  ".join(a.dtype.names))
        print(a)


# ---- helpers
# required for prn_tbl and prn_geo
def _ckw_(a, name, deci):
    """Array `a` c(olumns) k(ind) and w(idth)."""
    c_kind = a.dtype.kind
    if (c_kind in FLOATS) and (deci != 0):  # float with decimals
        c_max, c_min = np.round([np.min(a), np.max(a)], deci)
        c_width = len(max(str(c_min), str(c_max), key=len))
    elif c_kind in NUMS:  # int, unsigned int, float with no decimals
        c_width = len(max(str(np.min(a)), str(np.max(a)), key=len))
    elif c_kind in ('U', 'S', 's'):
        c_width = len(max(a, key=len))
    else:
        c_width = len(str(a))
    c_width = max(len(name), c_width) + deci
    return [c_kind, c_width]


def _col_format(pairs, deci):
    """Assemble the column format."""
    form_width = []
    dts = []
    for c_kind, c_width in pairs:
        if c_kind in INTS:  # ---- integer type
            c_format = ':>{}.0f'.format(c_width)
        elif c_kind in FLOATS:
            c_format = ':>{}.{}f'.format(c_width, deci)
        else:
            c_format = "!s:<{}".format(c_width)
        dts.append(c_format)
        form_width.append(c_width)
    return dts, form_width


# ---- main print functions
#
def col_hdr(num=8):
    """Print numbers from 1 to 10*num to show column positions."""
    args = [(('{:<10}')*num).format(*'0123456789'),
            '0123456789'*num, '-'*10*num]
    s = "\n{}\n{}\n{}".format(args[0][1:], args[1][1:], args[2])  # *args)
    print(s)


def make_row_format(dim=3, cols=5, a_kind='f', deci=1,
                    a_max=10, a_min=-10, width=100, prnt=False):
    """Format the row based on input parameters.

    `dim` - int
        Number of dimensions.
    `cols` : int
        Columns per dimension.

    `a_kind`, `deci`, `a_max` and `a_min` allow you to specify a data type,
    number of decimals and maximum and minimum values to test formatting.
    """
    if a_kind not in NUMS:
        a_kind = 'f'
    w_, m_ = [[':{}.0f', '{:0.0f}'], [':{}.{}f', '{:0.{}f}']][a_kind == 'f']
    m_fmt = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
    w_fmt = w_.format(m_fmt, deci)
    suffix = '  '
    while m_fmt*cols*dim > width:
        cols -= 1
        suffix = '.. '
    row_sub = (('{' + w_fmt + '}')*cols + suffix)
    row_frmt = (row_sub*dim).strip()
    if prnt:
        frmt = "Row format: dim cols: ({}, {})  kind: {} decimals: {}\n\n{}"
        print(dedent(frmt).format(dim, cols, a_kind, deci, row_frmt))
        a = np.random.randint(a_min, a_max+1, dim*cols)
        col_hdr(width//10)  # run col_hdr to produce the column headers
        print(row_frmt.format(*a))
    else:
        return row_frmt


def prn_(a, deci=2, width=120, prefix=". . "):
    """Alternate format to prn_nd function from `arraytools.frmts`.
    Inputs are largely the same.

    Parameters
    ----------
    a : ndarray
        An np.ndarray with `ndim` 1 through 5 supported.
    others :
        self-evident
    """
    def _piece(sub, i, frmt, linewidth):
        """Piece together 3D chunks by row."""
        s0 = sub.shape[0]
        block = np.hstack([sub[j] for j in range(s0)])
        txt = ""
        if i is not None:
            fr = ("({}" + ", {}"*len(a.shape[1:]) + ")\n")
            txt = fr.format(i, *sub.shape)
        for line in block:
            ln = frmt.format(*line)[:linewidth]
            end = ["\n", "...\n"][len(ln) >= linewidth]
            txt += indent(ln + end, ". . ")
        return txt
    # ---- main section ----
    # out = "\n{}... ndim: {}  shape: {}\n".format(title, a.ndim, a.shape)
    out = "\n"
    linewidth = width
    if a.ndim <= 1:
        return a
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    # ---- pull the 1st and 3rd dimension for 3D and 4D arrays
    frmt = make_row_format(dim=a.shape[-3],
                           cols=a.shape[-1],
                           a_kind=a.dtype.kind,
                           deci=deci,
                           a_max=a.max(),
                           a_min=a.min(),
                           width=width,
                           prnt=False)
    if a.ndim == 3:
        out += _piece(a, None, frmt, linewidth)  # ---- _piece ----
    elif a.ndim == 4:
        for i in range(a.shape[0]):  # s0):
            out += "\n" + _piece(a[i], i, frmt, linewidth)  # ---- _piece
    elif a.ndim == 5:
        frmt = frmt * a.shape[-4]
        for i in range(a.shape[0]):  # s0):
            for j in range(a.shape[1]):
                out += "\n" + _piece(a[i][j], i, frmt, linewidth)
    with np.printoptions(precision=deci, linewidth=width):
        print(out)


def prn_tbl(a, rows_m=20, names=None, deci=2, width=75):
    """Format a structured array with a mixed dtype.

    Parameters
    ----------
    a : array
        A structured/recarray. To print a single or multiple shapes, use
        a.get_shape('id') or a.pull_shapes(['list of ids'])
    rows_m : integer
        The maximum number of rows to print.  If rows_m=10, the top 5 and
        bottom 5 will be printed.
    names : list/tuple or None
        Column names to print, or all if None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters.

    See Also
    --------
    Alternate formats and information in `g.info` and `g.structure()` where
    `g` in a geo array.
    """
    # ----
    if hasattr(a, "IFT"):  # geo array
        a = a.IFT_str
    dtype_names = a.dtype.names
    if dtype_names is None:
        print("Structured/recarray required")
        return None
    if names is None:
        names = dtype_names
    # ---- slice off excess rows, stack upper and lower slice using rows_m
    if a.shape[0] > rows_m * 2:
        a = np.hstack((a[:rows_m], a[-rows_m:]))
    # ---- get the column formats from ... _ckw_ and _col_format ----
    pairs = [_ckw_(a[name], name, deci) for name in names]  # -- column info
    dts, wdths = _col_format(pairs, deci)                   # format column
    # ---- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    a = a[list(names[:N])]
    # ---- Assemble the formats and print
    tail = ['', ' ...'][N < len(names)]
    row_frmt = "  ".join([('{' + i + '}') for i in dts[:N]])
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = "  ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " ...   " + hdr2.format(*names[:N]) + tail
    header = "{}\n{}".format(header, "-" * len(header))
    txt = [header]
    for idx, i in enumerate(range(a.shape[0])):
        if idx == rows_m:
            txt.append("...")
        else:
            t = " {:>03.0f} ".format(idx) + row_frmt.format(*a[i]) + tail
            txt.append(t)
    msg = "\n".join(txt)
    print(msg)
    # return row_frmt, hdr2  # uncomment for testing


def prn_geo(a, rows_m=100, names=None, deci=2, width=75):
    """Format a structured array with a mixed dtype.

    Derived from arraytools.frmts and the prn_rec function therein.

    Parameters
    ----------
    a : array
        A structured/recarray.
    rows_m : integer
        The maximum number of rows to print.  If rows_m=10, the top 5 and
        bottom 5 will be printed.
    names : list/tuple or None
        Column names to print, or all if names is None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters.
    """
    # ----
    if names is None:
        names = ['shape', 'part', 'X', 'Y']
    # ---- slice off excess rows, stack upper and lower slice using rows_m
    if not hasattr(a, 'IFT'):
        print("Requires a Geo array")
        return None
    ift = a.IFT
    c = [np.repeat(ift[i, 0], ift[i, 2] - ift[i, 1])
         for i, p in enumerate(ift[:, 0])]
    c = np.concatenate(c)
    # ---- p: __ shape end, p0: x parts, p1: o start of parts, pp: concatenate
    p = np.where(np.diff(c, append=0) == 1, "___", "")
    p1 = np.asarray(["" if i not in ift[:, 2] else 'o' for i in range(len(p))])
    p0 = np.asarray(["-" if i == 'o' else "" for i in p1])
    pp = np.asarray([p[i] + p0[i] + p1[i] for i in range(len(p))])
    if a.shape[0] > rows_m:
        a = a[:rows_m]
        c = c[:rows_m]
        p = p[:rows_m]
    # ---- get the column formats from ... _ckw_ and _col_format ----
    deci = [0, 0, deci, deci]
    flds = [c, pp, a[:, 0], a[:, 1]]
    pairs = [_ckw_(flds[n], names[n], deci[n])
             for n, name in enumerate(names)]  # -- column info
    dts, wdths = _col_format(pairs, deci)      # format column
    # ---- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice the
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    # ---- Assemble the formats and print
    # row_frmt = " {:>03.0f} " + "  ".join([('{' + i + '}') for i in dts[:N]])
    row_frmt = " {:>03.0f} {:>5.0f}  {!s:<4}  {:>6.2f}  {:>6.2f}"
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = "  ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " pnt " + hdr2.format(*names[:N])
    header = "\n{}\n{}".format(header, "-" * len(header))
    txt = [header]
    for i in range(a.shape[0]):
        txt.append(row_frmt.format(i, c[i], pp[i], a[i, 0], a[i, 1]))
    msg = "\n".join(txt)
    print(msg)
    # return row_frmt, hdr2  # uncomment for testing


# =============================================================================
# ----  Extras ---------------------------------------------------------------
#
def gms(arr):
    """Get the maximum dimension in a list/array.

    Returns
    -------
    A list with the format - [3, 2, 4, 10, 2]. Representing the maximum
    expected value in each column::
      [ID, parts, pieces, points, pair]
    """
    from collections import defaultdict

    def get_dimensions(arr, level=0):
        yield level, len(arr)
        try:
            for row in arr:
                # print("{}\n{}".format(dimensions(row, level +1)))
                yield from get_dimensions(row, level + 1)
        except TypeError:  # not an iterable
            pass
    # ----
    dimensions = defaultdict(int)
    for level, length in get_dimensions(arr):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    print("\n{}".format(script))

"""
'C:/Git_Dan/npgeom/Polygons2_geo.geojson'
'C:/Git_Dan/npgeom/data/Polygons2_esrijson.json'
'C:/Git_Dan/npgeom/data/Polygons2_geojson.json'
'C:/Git_Dan/npgeom/data/g.npz'
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
