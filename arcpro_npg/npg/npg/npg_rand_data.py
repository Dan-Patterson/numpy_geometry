# -*- coding: utf-8 -*-
"""
-------------
npg_rand_data
-------------

Script :  npg_rand_data.py

Author :
    `<https://github.com/Dan-Patterson>`_.
Modified :
    2025-05-29

Purpose
-------
Tools for working with, and to create, tabular data in the Geo class.

References
----------

`<blog post_. https://community.esri.com/blogs/dan_patterson/2016/04/04/
numpy-lessons-6-creating-data-for-testing-purposes>`_.

`<numpy random documentation.
https://numpy.org/doc/2.2/reference/random/index.html>`_.

`<structured array example.
http://stackoverflow.com/questions/32224220/
methods-of-creating-a-structured-array>`_.

Notes
-----
Generating various random::

    # size : the array shape, eg. (5, 2) for 5 pnts with x,y between 0 - 1
    rng = np.random.default_rng()
    r_ints = rng.integers(low, high=None, size=None,
                          dtype=np.int64, endpoint=False)
    r_floats = rng.random(size=None, dtype=np.float64, out=None)
    r_ch = rng.choice(a, size=None, replace=True, p=None, axis=0,
                      shuffle=True)
    r_shuf = rng.shuffle(x, axis=0)
    r_uni = rng.uniform(low=0.0, high=1.0, size=None)

"""

# ---- imports, formats, constants ----
import sys
import numpy as np
import numpy.lib.recfunctions as rfn  # noqa
import npg  # noqa
from npg.npg_prn import prn_tbl, prn_, prn_q  # noqa

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 6.2f}'.format}
np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter=ft)
np.ma.masked_print_option.set_display("-")  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['strip_concatenate',
           ]

# ---- constants and required
#
str_opt = [
    '0123456789',
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
    'abcdefghijklmnopqrstuvwxyz',
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
]

# -- define `rng`
# rng = np.random.default_rng()


# ----------------------------------------------------------------------------
# ---- functions
def strip_concatenate(in_flds, strip_list=[" ", ",", None]):
    """Return a list of fields with spaces and nulls removed.

    Parameters
    ----------
    in_flds : list of strings
        The field names from an ndarray to clean.
    strip_list : list
        The values to remove.  Examples are provided.

    """
    fixed = []
    fmt = []
    for i in in_flds:
        if i not in strip_list:
            fixed.append(i)
            fmt.append("{}")
    frmt = " ".join([f for f in fmt])
    frmt.strip()
    fixed = [str(i).strip() for i in fixed]
    result = frmt.format(*fixed)
    return result


def concat_flds(a, flds=None, out_name="Concat", sep=" ", with_ids=True):
    """Return a structured array or ndarray from a sequence of fields.

    Parameters
    ----------
    a : array
        structured array
    flds : text
        a list of field names
    sep : text
        the separator between lists
    name : text
        used for structured array

    """
    strip_list = [" ", ",", None]
    if (flds is None) or (a.dtype.names is None):
        msg = "Field/column names are required or need to exist in the array."
        print(msg)
        return a
    N = min(len(flds), len(a.dtype.names))
    if N < 2:
        print("Two fields are required for concatenation")
        return a
    s0 = [str(i) if i not in strip_list else '' for i in a[flds[0]]]
    s1 = [str(i) if i not in strip_list else '' for i in a[flds[1]]]
    c = [("{}{}{}".format(i, sep, j)).strip() for i, j in list(zip(s0, s1))]
    if N > 2:
        for i in range(2, len(flds)):
            f = flds[i]
            f = [str(i) if i not in strip_list else '' for i in a[flds[i]]]
            c = ["{}{}{}".format(i, sep, j) for i, j in list(zip(c, f))]
    c = np.asarray(c)
    sze = c.dtype.str
    if out_name is not None:
        c.dtype = [(out_name, sze)]
    else:
        out_name = 'f'
    if with_ids:
        tmp = np.copy(c)
        dt = [('IDs', '<i8'), (out_name, sze)]
        c = np.empty((tmp.shape[0], ), dtype=dt)
        c['IDs'] = np.arange(1, tmp.shape[0] + 1)
        c[out_name] = tmp
    return c


def colrow_txt(N=10, cols=2, rows=2, zero_based=True):
    """Produce spreadsheet-like labels either 0- or 1-based.

    Parameters
    ----------
    N : number
        Number of records/rows to produce.
    cols/rows : numbers
        This combination will control the output of the values
        cols=2, rows=2 - yields (A0, A1, B0, B1)
        as optional classes regardless of the number of records being produced.
    zero-based : boolean
        True for conventional array structure,
        False for spreadsheed-style classes
    """
    if zero_based:
        start = 0
    else:
        start = 1
        rows = rows + 1
    UC = (list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))[:cols]  # see constants
    dig = (list('0123456789'))[start:rows]
    cr_vals = [c + r for r in dig for c in UC]
    rng = np.random.default_rng()
    return rng.choice(cr_vals, N)


def rowcol_txt(N=10, rows=2, cols=2):
    """Produce array-like labels in a tuple format."""
    rc_vals = ["({},{})".format(r, c)
               for c in range(cols)
               for r in range(rows)]
    rng = np.random.default_rng()
    return rng.choice(rc_vals, N)


def pnts_IdShape(N=10, x_min=0, x_max=10, y_min=0, y_max=10, simple=True):
    """Return an array with a nested dtype.

    Parameters
    ----------
    N : integer
        Samples size
    x_min, x_max, y_min, y_max : integer
    which emulates a shapefile's
    data structure.  This array is used to append other arrays to enable
    import of the resultant into ArcMap.  Array construction, after hpaulj

    """
    rng = np.random.default_rng()
    Xs = rng.integers(x_min, x_max, size=N)
    Ys = rng.integers(y_min, y_max, size=N)
    IDs = np.arange(0, N)
    c_stack = np.column_stack((IDs, Xs, Ys))
    if simple:     # version 1  short version, optional form
        dt = [('ID', '<i4'), ('X', '<f8'), ('Y', '<f8')]
        a = np.ones(N, dtype=dt)
        a['ID'] = c_stack[:, 0]
        a['X'] = c_stack[:, 1]         # this line too
        a['Y'] = c_stack[:, 2]
    else:          # version 2
        dt = [('ID', '<i4'), ('Shape', ([('X', '<f8'), ('Y', '<f8')]))]
        a = np.ones(N, dtype=dt)
        a['ID'] = c_stack[:, 0]
        a['Shape']['X'] = c_stack[:, 1]
        a['Shape']['Y'] = c_stack[:, 2]
    return a


# ---- private helpers
#
def _rand_text(N=10, cases=3, vals=str_opt[3]):
    """Return sample letters.

    Notes
    -----
    Generate `N` samples from the letters of the alphabet denoted by the
    number of `cases`.  If you want greater control on the text and
    probability, see `_rand_case` or `_rand_str`.

    vals:  see `str_opt` in required constants section.
    """
    rng = np.random.default_rng()
    vals = list(vals)
    rng = np.random.default_rng()
    return rng.choice(vals[:cases], N)


def _rand_str(N=10, low=1, high=10, vals=str_opt[3]):
    """Return randomly constructed strings.

    Notes
    -----
    Returns N strings constructed from 'size' random letters to form a
    string

    - create the cases as a list:  string.ascii_lowercase or ascii_uppercase
    - determine how many letters. Ensure min <= max. Add 1 to max alleviate
      low==high
    - shuffle the case list each time through loop
    """
    vals = list(vals)
    letts = np.arange(min([low, high]), max([low, high])+1)  # num letters
    result = []
    rng = np.random.default_rng()
    for i in range(N):
        rng.shuffle(vals)
        size = rng.choice(letts, 1)
        result.append("".join(vals[:size]))
    return np.array(result)


def _rand_case(N=10, cases=["Aa", "Bb"], p_vals=[0.8, 0.2]):
    """Return samples from a list of cases with a specified probability.

    Notes
    -----
    Generate N samples from a list of classes with an associated probability

    - ensure: len(cases)==len(p_vals) and  sum(p_values) == 1
    - small sample sizes will probably not yield the desired p-values
    """
    p = (np.array(p_vals))*N   # convert to integer
    kludge = [np.repeat(cases[i], p[i]).tolist() for i in range(len(cases))]
    case_vals = np.array([val for i in range(len(kludge))
                          for val in kludge[i]])
    rng = np.random.default_rng()
    rng.shuffle(case_vals)
    return case_vals


def _rand_int(N=10, begin=0, end=10):
    """Generate N random integers within the range begin - end."""
    rng = np.random.default_rng()
    return rng.integers(begin, end, size=(N))


def _rand_float(N=10, begin=0, end=10):
    """Generate N random floats within the range begin - end.

    Technically, N random integers are produced then a random
    amount within 0-1 is added to the value
    """
    rng = np.random.default_rng()
    float_vals = rng.randint(begin, end-1, size=(N))
    return float_vals + rng.rand(N)


# def blog_post():
#     """sample run."""
#     N = 10000
#     rng = np.random.default_rng()
#     id_shape = pnts_IdShape(N,
#                             x_min=300000,
#                             x_max=305000,
#                             y_min=5000000,
#                             y_max=5005000)
#     case1_fld = _rand_case(N,
#                            cases=['A', 'B', 'C', 'D'],
#                            p_vals=[0.4, 0.3, 0.2, 0.1])
#     int_fld = _rand_int(N, begin=0, end=10)
#     float_0 = _rand_float(N, 5, 15)
#     float_1 = _rand_float(N, 5, 20)
#     fld_names = ['Case', 'Observed', 'Size', 'Mass']
#     fld_data = [case1_fld, int_fld, float_0, float_1]
#     arr = rfn.append_fields(id_shape, fld_names, fld_data, usemask=False)
#     return arr


# def blog_post2(N=20):
#     """sample run
#     : import arcpy
#     : out_fc = r'C:\GIS\A_Tools_scripts\Graphing\Graphing_tools\
#     :            Graphing_tools.gdb\data_01'
#     : arcpy.da.NumPyArrayToFeatureClass(a, out_fc, ['X', 'Y'])
#     """
    # N = 10
    # ids = np.arange(1, N + 1)  # construct the base array of IDs to append to
    # ids = np.asarray(ids, dtype=[('Ids', '<i8')])
    # int_fld = _rand_int(N, begin=10, end=1000)
    # case1 = _rand_case(N,
    #                   cases=['N', 'S', 'E', 'W', ''],
    #                   p_vals=[0.1, 0.1, 0.2, 0.2, 0.4])
    # case2 = _rand_case(N,
    #                   cases=['Maple', 'Oak', 'Elm', 'Pine', 'Spruce'],
    #                   p_vals=[0.3, 0.15, 0.2, 0.25, 0.1])
    # case3 = _rand_case(N,
    #                   cases=['Ave', 'St', 'Crt'],
    #                   p_vals=[0.3, 0.6, 0.1])
    # case4 = _rand_case(N,
    #                   cases=['Carp', 'Almonte',
    #                          'Arnprior', 'Carleton Place'],
    #                   p_vals=[0.3, 0.3, 0.2, 0.2])
    # fld_names = ['Str_Number', 'Prefix', 'Str_Name', 'Str_Type', 'Town']
    # fld_data = [int_fld, case1, case2, case3, case4]
    # result =  rfn.append_fields(ids, fld_names, fld_data, usemask=False)
#    return result

# def joe_demo():
#     """To use...
#     zz, uni, idx, cnt, sub, final = joe_demo()
#     """
# zz = np.array([( 1, 316, '', 'Maple', 'St', 'Arnprior'),
#                ( 2, 257, 'E', 'Pine', 'Ave', 'Carp'),
#                ( 3, 561, '', 'Oak', 'St', 'Arnprior'),
#                ( 4, 771, '', 'Elm', 'Ave', 'Carleton Place'),
#                ( 5, 488, 'E', 'Spruce', 'St', 'Arnprior'),
#                ( 6, 523, 'W', 'Spruce', 'Ave', 'Arnprior'),
#                ( 7,  29, 'W', 'Elm', 'St', 'Almonte'),
#                ( 8, 374, '', 'Elm', 'St', 'Carp'),
#                ( 9, 477, 'W', 'Elm', 'St', 'Carp'),
#                (10, 714, '', 'Oak', 'St', 'Almonte'),
#                (11, 714, '', 'Oak', 'St', 'Almonte'),
#                (12, 477, 'W', 'Elm', 'St', 'Carp'),
#                (13, 374, '', 'Elm', 'St', 'Carp'),
#                (14,  29, 'W', 'Elm', 'St', 'Almonte'),
#                (15, 523, 'W', 'Spruce', 'Ave', 'Arnprior'),
#                (16, 488, 'E', 'Spruce', 'St', 'Arnprior'),
#                (17, 771, '', 'Elm', 'Ave', 'Carleton Place'),
#                (18, 561, '', 'Oak', 'St', 'Arnprior'),
#                (19, 257, 'E', 'Pine', 'Ave', 'Carp'),
#                (20, 316, '', 'Maple', 'St', 'Arnprior')],
#               dtype=[('Ids', '<i8'), ('Str_Number', '<i8'),
#                      ('Prefix', '<U1'), ('Str_Name', '<U6'),
#                      ('Str_Type', '<U3'), ('Town', '<U14')])
# uni, idx, cnt = np.unique(zz, True, False, True)
# all_names =  list(zz.dtype.names)
# uni_flds = list(zz.dtype.names[1:])  # keep on the address fields
# uni, idx, cnt = np.unique(zz[uni_flds], True, False, True)
# sub = zz[idx]
# final = sub[np.argsort(sub, order=all_names)]
#     return zz, uni, idx, cnt, sub, final

# zz, uni, idx, cnt, sub, final = joe_demo()
# uniq_towns, twn_cnt =  np.unique(zz['Town'], False, False, True)
# oh_where = np.logical_and(zz['Str_Name'] == 'Elm', zz['Str_Type'] == 'St')
# pick_one = zz[oh_where]

# just print the above to see how they work


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """create ID,Shape,{txt_fld,int_fld...of any number}
    """
