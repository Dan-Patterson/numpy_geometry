# -*- coding: utf-8 -*-
r"""\

npg_io
------

Script :
    .../npgeom/npg_io.py

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-12-12
    Creation date during 2019 as part of ``arraytools``.

Purpose
-------
Tools for working with point and poly features as an array class.
Requires npGeo to implement the array geometry class.

See Also
--------
__init__ :
    `__init__.py` has further information on arcpy related functionality.
npGeo :
    A fuller description of the Geo class, its methods and properties is given
    there.  This script focuses on getting arcpy or geojson geometry into
    numpy arrays.

References
----------
**General**

`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.
"""
# pylint: disable=C0330  # Wrong hanging indentation
# pylint: disable=E0611  # stifle the arcgisscripting
# pylint: disable=E265   # blocked comment thing
# pylint: disable=E1101  # ditto for arcpy
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=W0621  # redefining name
# pylint: disable=W0614  # unused import ... from wildcard import

# pylint: disable=C0103, C0302, C0415, E1136, E1121, R0904, R0914, W0212, W0221
# pylint: disable=R0902,  # attribute defined outside __init__... none in numpy

import sys
# from textwrap import indent  # dedent,
import json
import numpy as np
# import npgeom as npg
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

null_pnt = (np.nan, np.nan)  # ---- a null point

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=160, precision=2, suppress=True,
                    threshold=100, formatter=ft)

__all__ = [
    'dtype_info', 'load_geo', 'save_geo', 'load_txt', 'save_txt',
    'load_geojson',
    'prn_q', '_check', 'prn_tbl', 'prn_geo',
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
    names = list(a.dtype.names)  # [i[0] for i in dt]
    formats = [i[1] for i in dt]
    if as_string and names is not None:
        names = ", ".join([i for i in names])
        formats = ", ".join([i for i in formats])
    return names, formats


def load_geo(f_name, all_info=True):
    """Load a well formed `npy` file representing a structured array.

    An array, the description, field names and their size are returned.
    """
    npzfiles = np.load(f_name)
    f = npzfiles.files
    g = npzfiles['g']
    print("\nLoading...{}\nArrays included...{}".format(f_name, f))
    if all_info:
        desc = g.dtype.descr
        nms = g.dtype.names
        sze = [i[1] for i in g.dtype.descr]
        return g, desc, nms, sze
    return g


def save_geo(g, fname, folder):
    """Save an array as an npy file.

    Parameters
    ----------
    g : Geo array
        A complete Geo array
    fname : text
        Filename without file extention.
    folder : text
        A local folder.  It will be checked for path name compliance.
    The type of data in each column is arbitrary.  It will be cast to the
    given dtype at runtime
    """
    check = all([hasattr(g, i) for i in ['IFT', 'K', 'XT', 'SR']])
    if not check:
        print("Not a fully formed Geo array")
        return None
    IFT, K, XT, SR = [g.IFT, g.K, g.XT, g.SR]  # g is a Geo array
    folder = folder.replace("\\", "/")
    out_name = "{}/{}.npz".format(folder, fname)
    np.savez(out_name, g=g, ift=IFT, kind=K, extents=XT, spatial_ref=SR)
    print("\nGeo array saved to ... {} ...".format(out_name))
    return


def load_txt(name="arr.txt", data_type=None):
    """Read the structured/recarray created by save_txt.

    Parameters
    ----------
    dtype : data type
        If `None`, it allows the structure to be read from the array.
    delimiter : string
        Use a comma delimiter by default.
    skip_header : int
        Number of rows to skip at the beginning
    names : boolean
        If `True`, the first row contains the field names.
    encoding :
        Set to None to use system default
    see np.genfromtxt for all `args` and `kwargs`.

    """
    a = np.genfromtxt(name, dtype=data_type,
                      delimiter=",",
                      names=True,
                      autostrip=True,
                      encoding=None)  # ,skip_header=1)
    return a


def save_txt(a, name="arr.txt", sep=", ", dt_hdr=True):
    """Save a NumPy structured, recarray to text.

    Parameters
    ----------
    a : array
        input array
    fname : filename
        output filename and path otherwise save to script folder
    sep : separator
        column separater, include a space if needed
    dt_hdr : boolean
        if True, add dtype names to the header of the file

    """
    a_names = ", ".join(i for i in a.dtype.names)
    hdr = ["", a_names][dt_hdr]  # use "" or names from input array
    s = np.array(a.tolist(), dtype=np.unicode_)
    widths = [max([len(i) for i in s[:, j]])
              for j in range(s.shape[1])]
    frmt = sep.join(["%{}s".format(i) for i in widths])
    # vals = ", ".join([i[1] for i in a.dtype.descr])
    np.savetxt(name, a, fmt=frmt, header=hdr, comments="")
    print("\nFile saved...")


# ---- (2) json section ------------------------------------------------------
#
def load_geojson(pth, full=False, geometry=True):
    """Load a geojson file and convert to a Geo Array.

    The geojson is from the Features to JSON tool listed in the references.

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
    ... {'type':
    ...  'crs': {'type': 'name', 'properties': {'name': 'EPSG:2951'}},
    ...  'features': [{'type': 'Feature',
    ...    'id': 1,
    ...    'geometry': {'type':  'MultiPolygon',
    ...     'coordinates': snip},  # coordinate values
    ...     'properties': snip }}, # attribute values from table
    ... {'type': ... repeat}

    geometry : list
        A list of lists representing the features, their parts *for multipart
        features) and inner holes (for polygons).

    References
    ----------
    `geojson specification in detail
    <https://geojson.org/>`_.

    `Features to JSON
    <https://pro.arcgis.com/en/pro-app/tool-reference/conversion/
    features-to-json.htm>`_.

    `JSON to Features
    <https://pro.arcgis.com/en/pro-app/tool-reference/conversion/
    json-to-features.htm>`_.
    """
    # import json
    with open(pth) as f:
        data = json.load(f)
    shapes = data['features']
    coords = [s['geometry']['coordinates'] for s in shapes]
    if full and geometry:
        return data, coords
    if full:
        return data
    if geometry:
        return coords


# ============================================================================
# ---- (3) Print etc ---------------------------------------------------------
# printing based on arraytools.frmts.py using prn_rec and dependencies
#
def prn_q(a, edges=3, max_lines=25, width=120, decimals=2):
    """Format a structured array by setting the width so it wraps."""
    width = min(len(str(a[0])), width)
    with np.printoptions(edgeitems=edges, threshold=max_lines, linewidth=width,
                         precision=decimals, suppress=True, nanstr='-n-'):
        print("\nArray fields/values...:")
        print("  ".join([n for n in a.dtype.names]))
        print(a)


def _check(a):
    """Check dtype and max value for formatting information."""
    return a.shape, a.ndim, a.dtype.kind, np.min(a), np.max(a)


def prn_tbl(a, rows_m=20, names=None, deci=2, width=75):
    """Format a structured array with a mixed dtype.

    Derived from arraytools.frmts and the prn_rec function therein.

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
        Print width in characters
    """
    def _ckw_(a, name, deci):
        """Array `a` c(olumns) k(ind) and w(idth)."""
        c_kind = a.dtype.kind
        if (c_kind in FLOATS) and (deci != 0):  # float with decimals
            c_max, c_min = np.round([np.min(a), np.max(a)], deci)
            c_width = len(max(str(c_min), str(c_max), key=len))
        elif c_kind in NUMS:      # int, unsigned int, float wih no decimals
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
            elif c_kind in FLOATS:  # and np.isscalar(c[0]):  # float rounded
                c_format = ':>{}.{}f'.format(c_width, deci)
            else:
                c_format = "!s:<{}".format(c_width)
            dts.append(c_format)
            form_width.append(c_width)
        return dts, form_width
    # ----
    dtype_names = a.dtype.names
    if dtype_names is None:
        print("Structured/recarray required")
        return None
    if names is None:
        names = dtype_names
    # ---- slice off excess rows, stack upper and lower slice using rows_m
    if a.shape[0] > rows_m*2:
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
    header = " ... " + hdr2.format(*names[:N]) + tail
    header = "{}\n{}".format(header, "-"*len(header))
    txt = [header]
    for idx, i in enumerate(range(a.shape[0])):
        if idx == rows_m:
            txt.append("...")
        else:
            t = " {:>03.0f} ".format(idx) + row_frmt.format(*a[i]) + tail
            txt.append(t)
    msg = "\n".join([i for i in txt])
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
        Column names to print, or all if None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters.

    Notes
    -----
    >>> toos = s0.IFT[:,2]
    >>> nans = np.where(np.isnan(s0[:,0]))[0]  # array([10, 21, 31, 41]...
    >>> dn = np.digitize(nans, too)            # array([1, 2, 3, 4]...
    >>> ift[:, 0][dn]                          # array([1, 1, 2, 2])
    >>> np.sort(np.concatenate((too, nans)))
    ... array([ 5, 10, 16, 21, 26, 31, 36, 41, 48, 57, 65], dtype=int64)
    """
    def _ckw_(a, name, deci):
        """Columns `a` kind and width."""
        c_kind = a.dtype.kind
        if (c_kind in FLOATS) and (deci != 0):  # float with decimals
            c_max, c_min = np.round([np.min(a), np.max(a)], deci)
            c_width = len(max(str(c_min), str(c_max), key=len))
        elif c_kind in NUMS:      # int, unsigned int, float wih no decimals
            c_width = len(max(str(np.min(a)), str(np.max(a)), key=len))
        else:
            c_width = len(name)
        c_width = max(len(name), c_width) + deci
        return [c_kind, c_width]

    def _col_format(pairs, deci):
        """Assemble the column format."""
        form_width = []
        dts = []
        for c_kind, c_width in pairs:
            if c_kind in INTS:  # ---- integer type
                c_format = ':>{}.0f'.format(c_width)
            elif c_kind in FLOATS:  # and np.isscalar(c[0]):  # float rounded
                c_format = ':>{}.{}f'.format(c_width, deci[-1])
            else:
                c_format = "!s:^{}".format(c_width)
            dts.append(c_format)
            form_width.append(c_width)
        return dts, form_width
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
    pp = np.asarray([p[i]+p0[i]+p1[i] for i in range(len(p))])
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
    row_frmt = " {:>03.0f} " + "  ".join([('{' + i + '}') for i in dts[:N]])
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = "  ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " pnt " + hdr2.format(*names[:N])
    header = "\n{}\n{}".format(header, "-"*len(header))
    txt = [header]
    for i in range(a.shape[0]):
        txt.append(row_frmt.format(i, c[i], pp[i], a[i, 0], a[i, 1]))
    msg = "\n".join([i for i in txt])
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
