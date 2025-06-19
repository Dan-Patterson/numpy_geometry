# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
--------
  npg_prn
--------

**Print related functions**

Specialized print functions facilitate the viewing of the Geo array geometry
and attribute information.

----

Script :
    .../npg/npg_prn.py

Author :
    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-02-13

Purpose
-------
Tools to print Geo arrays and related objects.

See Also
--------
__init__ :
    The ``.../npgeom/__init__.py`` script has further information on arcpy
    related functionality.
npGeo :
    A fuller description of the Geo class, its methods and properties is given
    ``.../npgeom/npGeo``.

"""
# pylint: disable=C0103, C0302, C0415
# pylint: disable=E0611, E1101, E1136, E1121
# pylint: disable=R0902, R0904, R0914
# pylint: disable=W0105, W0201, W0212, W0221, W0612, W0614, W0621, W0105

import sys
from textwrap import indent, dedent
import numpy as np

# import npGeo

from npg import npg_geom_hlp as n_h
# import npg_geom_hlp as n_h
# from npg_geom_hlp import shape_finder

# ---- Keep for now.
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts  #noqa
# import npGeo
# from npGeo import *


# ---- Constants
#
script = sys.argv[0]  # print this should you need to locate the script

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS


np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    legacy='1.25',
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 6.2f}'.format}
    )  # legacy=False or legacy='1.25'
np.ma.masked_print_option.set_display('-')  # change to a single -

__all__ = [
    'prn_q',
    'prn_',
    'prn_tbl',
    'prn_geo',
    'prn_lists',
    'prn_arrays',
    'prn_as_obj',
    'prn_Geo_shapes',
    '_svg'
]

__helpers__ = [
    '_ckw_',
    '_col_format',
    'col_hdr',
    'make_row_format'
]


# ---- ---------------------------
# ---- (1) print functions
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


def col_hdr(num=8):
    """Print numbers from 1 to 10*num to show column positions."""
    args = [(('{:<10}') * num).format(*'0123456789'),
            '0123456789' * num, '-' * 10 * num]
    s = "\n{}\n{}\n{}".format(args[0][1:], args[1][1:], args[2])  # *args)
    print(s)


def make_row_format(dim=3, cols=5, a_kind='f', deci=1,
                    a_max=10, a_min=-10, width=100, prnt=False):
    """Format the row based on input parameters.

    dim : int
        Number of dimensions.
    cols : int
        Columns per dimension.

    ``a_kind``, ``deci``, ``a_max`` and ``a_min`` allow you to specify a data
    type, number of decimals and maximum and minimum values to test formatting.

    Requires
    --------
    ``col_hdr``

    """
    if a_kind not in NUMS:
        a_kind = 'f'
    w_, m_ = [[':{}.0f', '{:0.0f}'], [':{}.{}f', '{:0.{}f}']][a_kind == 'f']
    m_fmt = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
    w_fmt = w_.format(m_fmt, deci)
    suffix = '  '
    while m_fmt * cols * dim > width:
        cols -= 1
        suffix = '.. '
    row_sub = (('{' + w_fmt + '}') * cols + suffix)
    row_frmt = (row_sub * dim).strip()
    if prnt:
        frmt = "Row format: dim cols: ({}, {})  kind: {} decimals: {}\n\n{}"
        print(dedent(frmt).format(dim, cols, a_kind, deci, row_frmt))
        a = np.random.randint(a_min, a_max + 1, dim * cols)
        col_hdr(width // 10)  # run col_hdr to produce the column headers
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

    Requires
    --------
    - `from textwrap import indent`
    - `make_row_format`, `col_hdr`
    """
    def _piece(sub, i, frmt, linewidth):
        """Piece together 3D chunks by row."""
        s0 = sub.shape[0]
        block = np.hstack([sub[j] for j in range(s0)])
        txt = ""
        if i is not None:
            fr = ("({}" + ", {}" * len(a.shape[1:]) + ")\n")
            txt = fr.format(i, *sub.shape)
        for line in block:
            ln = frmt.format(*line)[:linewidth]
            end = ["\n", "...\n"][len(ln) >= linewidth]
            txt += indent(ln + end, prefix)
        return txt
    # -- main section ----
    # out = "\n{}... ndim: {}  shape: {}\n".format(title, a.ndim, a.shape)
    out = "\n"
    linewidth = width
    if a.ndim <= 1:
        for i, arr in enumerate(a):
            print("{} ...\n{}".format(i, arr))
        return None
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    # -- pull the 1st and 3rd dimension for 3D and 4D arrays
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


def prn_tbl(a, rows_m=20, names=None, deci=2, width=88):
    """Print and format a structured array with a mixed dtype.

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

    Requires
    --------
    `_ckw_`, `_col_format`

    See Also
    --------
    Alternate formats and information in `g.facts` and `g.structure` where
    `g` in a geo array.

    """
    # --
    if hasattr(a, "IFT"):  # geo array
        a = a.IFT_str
    dtype_names = a.dtype.names
    if dtype_names is None:
        print("Structured/recarray required")
        return None
    if names is None:
        names = dtype_names
    # -- slice off excess rows, stack upper and lower slice using rows_m
    if a.shape[0] > rows_m * 2:
        a = np.hstack((a[:rows_m], a[-rows_m:]))
    # -- get the column formats from ... _ckw_ and _col_format ----
    pairs = [_ckw_(a[name], name, deci) for name in names]  # -- column info
    dts, wdths = _col_format(pairs, deci)                   # format column
    # -- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    a = a[list(names[:N])]
    # -- Assemble the formats and print
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
    return None
    # return row_frmt, hdr2  # uncomment for testing


def prn_geo(a, rows_m=100, names=None, deci=2, width=75):
    """Print and format a `geo` array with ring information.

    Derived from arraytools.frmts and the prn_rec function therein.

    Parameters
    ----------
    a : array
        A geo array.
    rows_m : integer
        The maximum number of rows to print.  If rows_m=10, the top 5 and
        bottom 5 will be printed.
    names : list/tuple or None
        Column names to print, or all if names is None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters.

    Requires
    --------
    `_ckw_`, `_col_format`

    """
    # --
    if names is None:
        names = ['shape', 'part', 'X', 'Y']
    # -- slice off excess rows, stack upper and lower slice using rows_m
    if not hasattr(a, 'IFT'):
        print("Requires a Geo array")
        return None
    ift = a.IFT
    c = [np.repeat(ift[i, 0], ift[i, 2] - ift[i, 1])
         for i, p in enumerate(ift[:, 0])]
    c = np.concatenate(c)
    # -- p: __ shape end, p0: x parts, p1: o start of parts, pp: concatenate
    p = np.where(np.diff(c, append=0) == 1, "___", "")
    p1 = np.asarray(["" if i not in ift[:, 2] else 'o' for i in range(len(p))])
    p0 = np.asarray(["-" if i == 'o' else "" for i in p1])
    pp = np.asarray([p[i] + p0[i] + p1[i] for i in range(len(p))])
    if a.shape[0] > rows_m:
        a = a[:rows_m]
        c = c[:rows_m]
        p = p[:rows_m]
    # -- get the column formats from ... _ckw_ and _col_format ----
    deci = [0, 0, deci, deci]
    flds = [c, pp, a[:, 0], a[:, 1]]
    pairs = [_ckw_(flds[n], names[n], deci[n])
             for n, name in enumerate(names)]  # -- column info
    dts, wdths = _col_format(pairs, deci)      # format column
    # -- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice the
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    # -- Assemble the formats and print
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
    return None
    # return row_frmt, hdr2  # uncomment for testing


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


# ---- ---------------------------
# ----  (2) print & display Geo and ndarrays
#
def prn_lists(a, max_=None, prn_structure=False):
    """Print nested lists as string.

    Requires
    --------
    ``npg.shape_finder``

    See Also
    --------
    ``npg_geom_hlp.shape_finder`` to print or return the structure of the
    nested structure.
    """
    if prn_structure:
        n_h.shape_finder(a, prn=True)  # see npg_geom_hlp
    if max_ is None:
        max_ = 70
    for i, v in enumerate(a):
        print(f"\n({i})...")
        for j in v:
            vals = repr(j).split("],")
            for val in vals:
                val = "{}]".format(val)
                s_len = len(val)
                ending = ["", " ..."][s_len > max_]
                print("{}{}".format(indent(val[:max_], "   "), ending))
    # return


def prn_arrays(a, edgeitems=2):
    """Print a different representation of object or ndarrays.

    The expectation is that the array has nested objects or ndim is > 3:
    edgeitems, threshold : integer
        This is on a per sub array basis.

    Note
    ----
    `npGeo.prn_arr(self)` is a shortcut to this.  The Geo array is converted to
    a list of arrays using `geo.as_arrays()`, then passed to here.
    """
    def _ht_(a, _e):
        """Print 2d array."""
        head = repr(a[:_e].tolist())[:-1]
        tail = repr(a[-_e:].tolist())[1:]
        return head, tail

    if a.ndim == 2:
        return a
    _e = edgeitems
    s = n_h.shape_finder(a)
    u, cnts = np.unique(s[['shape', 'part']], return_counts=True)
    s0 = stu(u)
    N = np.arange(len(s0))
    tb = " ... "
    for cnt in N:
        i, j = s0[cnt]
        sub = a[i]
        if sub.ndim == 2:
            head, tail = _ht_(sub, _e)
            print("\n({},{},0) {}{}{}".format(i, j, head, tb, tail))
        else:
            sub = sub[j]
            if sub.ndim == 2:
                head, tail = _ht_(sub, _e)
                print("\n({},{},0) {}{}{}".format(i, j, head, tb, tail))
            else:
                print("\n({},{},.)".format(i, j))
                for k, val in enumerate(sub):
                    head, tail = _ht_(val, _e)
                    ht = head + " ... " + tail
                    print("     {} - {}".format(k, ht))  # val.tolist()))
    return None


def prn_as_obj(arr, full=False):
    """Print Geo or ndarray as an object array. Lists use _nested_lists_.

    Parameters
    ----------
    arr : array-like
        Geo or ndarray.  If nested lists are used, the ``_nested_lists_`` is
        used.
    full : boolean
        If the input is a Geo array, then its structure is printed as well.

    Requires
    --------
    ``prn_nested_lists`` for optional printing of nested lists.
    """
    if hasattr(arr, "IFT"):
        arrs = arr.as_arrays()
        # ids = np.unique(arr.IDs)
    elif isinstance(arr, np.ndarray):
        arrs = arr
    else:
        return prn_lists(arr)
    #
    fmt = [repr(arr) for arr in arrs]
    if full and hasattr(arr, "IFT"):
        arr.structure
    else:
        print("\nArray structure by sub-array.")
    with np.printoptions(precision=2):
        for i, z in enumerate(fmt):
            print("{}...\n{}".format(i, z))
    return None


def prn_Geo_shapes(arr, ids=None):
    """Print all shapes if ``ids=None``, otherwise, provide an id list."""
    cases = arr.IFT[:, 0]
    if ids is None:
        ids = arr.IFT[0]
        w = np.isin(cases, cases)
    elif isinstance(ids, (list, tuple)):
        w = np.isin(cases, ids)
    elif isinstance(ids, (int)):
        w = np.isin(cases, ids)
    if np.sum(w) == 0:
        print("\nRecord ID not found...\n")
        return None
    rows = arr.IFT[w]
    hdr = "ID : Shape ID by part\nR  : ring, outer 1, inner 0\n" + \
          "P  : part 1 or more\n"
    hdr += " ID  R  P      x       y\n"
    for i, row in enumerate(rows):
        sub = arr[row[1]:row[2]]
        args = [row[0], row[3], row[4], sub[0]]
        s0 = "{: 3.0f}{: 3.0f}{: 3.0f}  {!s:}\n".format(*args)
        s1 = ""
        for j in sub[1:]:
            s1 += "{} {}\n".format(" " * 10, j)
        hdr += s0 + s1
    print(hdr)
    return None


def _svg(arr, as_polygon=True):
    """Format and show a Geo array, np.ndarray or list structure in SVG format.

    Notes
    -----
    Geometry must be expected to form polylines or polygons.
    IPython required.

    If `arr` is a Geo array, this can be called by **arr.svg()**.

    >>> from IPython.display import SVG

    Alternate colors::

        white, silver, gray black, red, maroon, purple, blue, navy, aqua,
        green, teal, lime, yellow, magenta, cyan
    """
    def svg_path(g_bits, scale_by, o_f_s):
        """Make the svg from a list of 2d arrays."""
        opacity, fill_color, stroke = o_f_s
        pth = [" M {},{} " + "L {},{} " * (len(b) - 1) for b in g_bits]
        ln = [pth[i].format(*b.ravel()) for i, b in enumerate(g_bits)]
        pth = "".join(ln) + "z"
        s = ('<path fill-rule="evenodd" fill="{0}" stroke="{1}" '
             'stroke-width="{2}" opacity="{3}" d="{4}"/>'
             ).format(fill_color, stroke, 1.5 * scale_by, opacity, pth)
        return s
    # --
    msg0 = "\nImport error..\n>>> from IPython.display import SVG\nfailed."
    msg1 = "A Geo array or ndarray (with ndim >=2) is required."
    # --
    # Geo array, np.ndarray check
    try:
        from IPython.core.display import SVG  # 2020-07-02
    except ImportError:
        print(dedent(msg0))
        return None
    # -- checks for Geo or ndarray. Convert lists, tuples to np.ndarray
    if isinstance(arr, (list, tuple)):
        dt = "float"
        leng = [len(i) for i in arr]
        if min(leng) != max(leng):
            dt = "O"
        arr = np.asarray(arr, dtype=dt)
    # if ('Geo' in str(type(g))) & (issubclass(g.__class__, np.ndarray)):
    if hasattr(arr, "IFT"):
        GA = True
        g_bits = arr.bits
        L, B = arr.min(axis=0)
        R, T = arr.max(axis=0)
    elif isinstance(arr, np.ndarray):
        GA = False
        if arr.ndim == 2:
            g_bits = [arr]
            L, B = arr.min(axis=0)
            R, T = arr.max(axis=0)
        elif arr.ndim == 3:
            g_bits = [arr[i] for i in range(arr.shape[0])]
            L, B = arr.min(axis=(0, 1))
            R, T = arr.max(axis=(0, 1))
        elif arr.dtype.kind == 'O':
            g_bits = []
            for i, b in enumerate(arr):
                b = np.array(b)
                if b.ndim == 2:
                    g_bits.append(b)
                elif b.ndim == 3:
                    g_bits.extend([b[i] for i in range(b.shape[0])])
            L, B = np.min(np.vstack([np.min(i, axis=0) for i in g_bits]),
                          axis=0)
            R, T = np.max(np.vstack([np.max(i, axis=0) for i in g_bits]),
                          axis=0)
        else:
            print(msg1)
            return None
    else:
        print(msg1)
        return None
    # --
    # derive parameters
    if as_polygon:
        o_f_s = ["0.75", "red", "black"]  # opacity, fill_color, stroke color
    else:
        o_f_s = ["1.0", "none", "red"]
    # --
    d_x, d_y = (R - L, T - B)
    hght = min([max([225., d_y]), 300])  # ---- height 150-200, or 225-300
    width = int(d_x / d_y * hght)
    scale_by = max([d_x, d_y]) / max([width, hght])
    # --
    # derive the geometry path
    pth_geom = svg_path(g_bits, scale_by, o_f_s)  # ---- svg path string
    # construct the final output
    view_box = "{} {} {} {}".format(L, B, d_x, d_y)
    transform = "matrix(1,0,0,-1,0,{0})".format(T + B)
    hdr = '<svg xmlns="http://www.w3.org/2000/svg" ' \
          'xmlns:xlink="http://www.w3.org/1999/xlink" '
    f0 = 'width="{}" height="{}" viewBox="{}" '.format(width, hght, view_box)
    f1 = 'preserveAspectRatio="xMinYMin meet">'
    f2 = '<g transform="{}">{}</g></svg>'.format(transform, pth_geom)
    s = hdr + f0 + f1 + f2
    if GA:  # Geo array display
        arr.SVG = s
        return SVG(arr.SVG)  # plot the representation
    return SVG(s)  # np.ndarray display


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
                # print("{}\n".format(dimensions(row, level + 1)))
                yield from get_dimensions(row, level + 1)
        except TypeError:  # not an iterable
            pass  #
    # --
    dimensions = defaultdict(int)
    for level, length in get_dimensions(arr):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


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
