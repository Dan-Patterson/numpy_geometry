# -*- coding: utf-8 -*-
"""
=========
npg_table
=========

Script :
    npg_table.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-10-27

Purpose :
    Tools for working with tabular data in the Geo class.

import numpy.lib.recfunctions as rfn

>>> dir(rfn)
... ["MaskedArray", "MaskedRecords", "__all__", "__builtins__", "__cached__",
...  "__doc__", "__file__", "__loader__", "__name__", "__package__",
...  "__spec__", ..., "_check_fill_value", ..., "_fix_defaults", "_fix_output",
...  "_get_fields_and_offsets", "_get_fieldspec", "_is_string_like",
...  "_izip_fields", "_izip_fields_flat", "_izip_records", ..., "_keep_fields",
...  "_zip_descr", "_zip_dtype", "absolute_import", "append_fields",
...  "apply_along_fields", ..., "assign_fields_by_name", "basestring",
...  "division", "drop_fields", "find_duplicates", "flatten_descr",
...  "get_fieldstructure", "get_names", "get_names_flat", ...,
...  "join_by", ..., "merge_arrays", ...,
...  "rec_append_fields", "rec_drop_fields", "rec_join", "recarray",
...  "recursive_fill_fields", "rename_fields", "repack_fields",
...  "require_fields", "stack_arrays", "structured_to_unstructured",
...  "suppress_warnings", ..., "unstructured_to_structured"]

Useful ones: append_fields, drop_fields, _keep_fields, join_by, repack_fields
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=unused-import
# pylint: disable=W0611

import sys
from textwrap import dedent
import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import _keep_fields

import npg_io
# from npg_io import prn_tbl

ft = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": "{: 0.3f}".format}
np.set_printoptions(
        edgeitems=10, linewidth=80, precision=2, suppress=True, threshold=100,
        formatter=ft
        )
np.ma.masked_print_option.set_display("-")  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    "_as_pivot", "crosstab_tbl", "crosstab_rc", "crosstab_array",
    "calc_stats", "_get_numeric_fields", "col_stats",
    "group_stats", "find_a_in_b", "find_in", "group_sort",
    "keep_fields_by_kind", "n_largest_vals", "n_smallest_vals",
    "split_sort_slice"
    ]


def nd2struct(a, fld_names=None):
    """Return a view of an ndarray as structured array with a uniform dtype.
    Same as unstructured_to_structured in np.lib.recfunctions.

    Parameters
    ----------
    a : array
        ndarray with a uniform dtype.
    fld_names : list/tuple
        One name for each column/field.  If None is provided, then the field
        names are assigned from an alphabetical list up to 26 fields.
        The dtype of the input array is retained, but can be upcast.

    Examples
    --------
    >>> a = np.arange(2*3).reshape(2, 3)
    array([[0, 1, 2],
           [3, 4, 5]])  # dtype('int64')
    >>> b = nd2struct(a)
    array([(0, 1, 2), (3, 4, 5)],
          dtype=[('A', '<i8'), ('B', '<i8'), ('C', '<i8')])
    >>> c = nd2struct(a.astype(np.float64))
    array([( 0.,  1.,  2.), ( 3.,  4.,  5.)],
          dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])

    See Also
    --------
    Equivalent, but faster than.

    `<from numpy.lib.recfunctions import unstructured_to_structured as uts>`_.

    - pack_last_axis(arr, names=None) at the end
    - nd_struct(flds=None, types=None)  if you want to provide dtypes as well
    """
    if a.dtype.names:  # return if a structured array already
        return a
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if a.ndim != 2:
        frmt = "Wrong array shape... read the docs..\n{}"
        print(frmt.format(nd2struct.__doc__))
        return a
    _, cols = a.shape
    if fld_names is None:
        names = list(alph)[:cols]
    elif (len(fld_names) == cols) and (cols < 26):
        names = fld_names
    else:  # from... pack_last_axis
        names = ['f{:02.0f}'.format(i) for i in range(cols)]
    return a.view([(n, a.dtype) for n in names]).squeeze(-1)  # or .[..., 0]


def struct2nd(a):
    """Return a view of a structured array of a uniform dtype to a regular
    ndarray using the the first column dtype and length/number of columns.

    Notes
    -----
    The is a quick function.  The expectation is that the array contains a
    uniform dtype (e.g "f8").  For example, coordinate values in the form
    ``dtype([("X", "<f8"), ("Y", "<f8")])`` maybe with a Z

    See ``structured_to_unstructured`` in np.lib.recfunctions and the imports.
    """
    return a.view((a.dtype[0], len(a.dtype.names)))  # alternate to stu


# ==== Crosstabulation tools =================================================
# ---- fancy print/string formatter for crosstabulation and pivot
def _prn(r, c, a, stat_name="Total"):
    """Fancy print formatting.
    """
    r = r.tolist()
    r.append(stat_name)
    c = c.tolist()
    c.append(stat_name)
    r_sze = max(max([len(str(i)) for i in r]), 8)
    c_sze = [max(len(str(i)), 5) for i in c]
    f_0 = "{{!s:<{}}} ".format(r_sze)
    f_1 = ("{{!s:>{}}} "*len(c)).format(*c_sze)
    frmt = f_0 + f_1
    hdr = "Result" + "_"*(r_sze-7)
    txt = [frmt.format(hdr, *c)]
    txt2 = txt + [frmt.format(r[i], *a[i]) for i in range(len(r))]
    result = "\n".join(txt2)
    return result


def _as_pivot(a):
    """Used by ``crosstab_tbl``. Present results in pivot table format."""
    if a.dtype.fields is None:
        print("\n...\nStructured array with field names is required")
        return a
    flds = list(a.dtype.names)
    r = np.unique(a[flds[0]])
    c = np.unique(a[flds[1]])
    z = np.zeros((len(r)+1, len(c)+1), dtype=np.float)
    rc = [[(np.where(r == i[0])[0]).item(),
           (np.where(c == i[1])[0]).item()] for i in a]
    for i in range(len(a)):
        rr, cc = rc[i]
        z[rr, cc] = a[i][2]
    z[-1, :] = np.sum(z, axis=0)
    z[:, -1] = np.sum(z, axis=1)
    result = _prn(r, c, z, stat_name="Count")
    return result


"""
#out_tbl is a featureclass table name
#txt_file is a text file name
#
#    if not (out_tbl in ["#", "", None]):
#        arcpy.da.NumPyArrayToTable(ctab, out_tbl)
#    if not (txt_file in ["#", "", None]):
#        with open(txt_file, "w") as f:
#            f.write("Crosstab for ... {}\n".format(in_tbl))
#            f.write(result)
"""


# ---- (1) from featureclass table
def crosstab_tbl(in_tbl, flds=None, as_pivot=True):
    """Derive the unique attributes in a table for all or selected fields.

    Parameters
    ----------
    in_tbl : table
        A featureclass or its table.
    flds : fields
        If None, then all fields in the table are used.
        Make sure that you do not include sequential id fields or all table
        records will be returned.

    Notes
    -----
    None or <null> values in tables are converted to proper nodata values
    depending on the field type.  This is handled by the call to fc_data which
    uses _make_nulls_ to do the work.
    """
    a = npg_io.fc_data(in_tbl)
    if flds is None:
        flds = list(a.dtype.names)
    uni, idx, cnts = np.unique(a[flds], True, False, True)
    out_arr = rfn.append_fields(uni, "Counts", cnts, usemask=False)
    if as_pivot:
        return _as_pivot(out_arr)
    return out_arr


# ---- (2) from two, 1D numpy ndarrays
def crosstab_rc(row, col, reclassed=False):
    """Crosstabulate 2 data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Parameters
    ----------
    row, col : text
        row and column array/field

    Returns
    -------
    ctab : the crosstabulation result as row, col, count array
    rc_ : similar to above, but the row/col unique pairs are combined.
    """
    dt = np.dtype([("row", row.dtype), ("col", col.dtype)])
    rc_zip = list(zip(row, col))
    rc = np.asarray(rc_zip, dtype=dt)
    u, idx, cnts = np.unique(rc, return_index=True, return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(("Count", "<i4"))
    ctab = np.asarray(list(zip(u["row"], u["col"], cnts)), dtype=rcc_dt)
    # ----
    if reclassed:
        rc2 = np.array(["{}_{}".format(*i) for i in rc_zip])
        u2, idx2, cnts2 = np.unique(rc2, return_index=True, return_counts=True)
        dt = [("r_c", u2.dtype.str), ("cnts", "<i4")]
        rc_ = np.array(list(zip(u2, cnts2)), dtype=dt)
        return rc_
    return ctab


# ---- (3) from a structured array
def crosstab_array(a, flds=None):
    """Frequency and crosstabulation for structured arrays.

    Parameters
    ----------
    a : array
       Input structured array.
    flds : string or list
       Fields/columns to use in the analysis.  For a single column, a string
       is all that is needed.  Multiple columns require a list of field names.

    Notes
    -----
    (1) Slice the input array by the classification fields.
    (2) Sort the sliced array using the flds as sorting keys.
    (3) Use unique on the sorted array to return the results.
    (4) Reassemble the original columns and the new count data.
    """
    if flds is None:
        return None
    if isinstance(flds, (str)):
        flds = [flds]
    a = rfn.repack_fields(a[flds])  # need to repack fields
    # a = _keep_fields(a, flds)  # alternative to rfn.repack_fields
    idx = np.argsort(a, axis=0, order=flds)  # (2) sort
    a_sort = a[idx]
    uni, cnts = np.unique(a_sort, return_counts=True)  # (3) unique, count
    dt = uni.dtype.descr
    dt.append(("Count", "<i4"))
    fr = np.empty_like(uni, dtype=dt)
    names = fr.dtype.names
    vals = list(zip(*uni)) + [cnts.tolist()]  # (4) reassemble
    N = len(names)
    for i in range(N):
        fr[names[i]] = vals[i]
    return fr


# ---- Summarize tools -------------------------------------------------------
# ---- (1) statistics functions
def calc_stats(arr, axis=None, deci=4):
    """Calculate stats for an array of number types, with nodata (nan, None)
    in the column.

    Notes
    -----
    See the args tuple for examples of nan functions.

    >>> np.nansum(b, axis=0)   # by column
    >>> np.nansum(b, axis=1)   # by row
    >>> c_nan = np.count_nonzero(~np.isnan(b), axis=0) count nan if needed

    [1, 0][True]  # ax = [1, 0][colwise]  colwise= True
    """
    if (axis is None) and (len(arr.shape) == 1):
        ax = 0
    else:
        ax = axis
    #
    kind = arr.dtype.kind
    arr_dt = arr.dtype
    if kind == "i":
        nulls = [np.iinfo(arr_dt).min, np.iinfo(arr_dt).max]
    elif kind == "f":
        nulls = [np.nan, np.finfo(arr_dt).min, np.finfo(arr_dt).max]
    elif kind in ("U", "S"):
        return None
    #
    nin = ~np.isin(arr, nulls)  # nin... Not In Nulls
    a = arr[nin]
    if len(arr.shape) > 1:
        a = a.reshape(arr.shape)
    mask = np.isnan(arr)
    N = len(a)
    cnt = np.sum(~mask, axis=ax, dtype=np.intp, keepdims=False)
    n_sum = np.nansum(a, axis=ax)
    n_min = np.nanmin(a, axis=ax)
    n_max = np.nanmax(a, axis=ax)
    n_mean = np.nanmean(a, axis=ax)
    n_med = np.nanmedian(a, axis=ax)
    n_std = np.nanstd(a, axis=ax)
    n_var = np.nanvar(a, axis=ax)
    s = [N, N-cnt, n_sum, n_min, n_max, n_mean, n_med, n_std, n_var]
    s = [np.around(i, deci) for i in s]
    return s


def _get_numeric_fields(a, fields):
    """Determine numeric fields in a structured/recarray.
    """
    num_flds = []
    dt_names = a.dtype.names
    dt_kind = a.dtype.kind
    if fields is None:
        if dt_names is None:
            if dt_kind not in ("i", "f"):
                return None
        elif dt_kind in ["V"]:
            num_flds = [i for i in dt_names if a[i].dtype.kind in ("i", "f")]
        else:
            a = a.ravel()
    elif isinstance(fields, (str)):
        if a[fields].dtype.kind in ("i", "f"):
            num_flds = fields
    else:
        num_flds = [i for i in fields if a[i].dtype.kind in ("i", "f")]
    return num_flds


def col_stats(a, fields=None, deci=2, verbose=False):
    """Calculate statistics for a structured/recarray with or without specified
    fields.  Efforts have been made to check for all possible scenarios, but
    human intelligence should prevail when one decides what to throw at it.

    Parameters
    ----------
    a : array
        A structured/recarray.
    fields : list, string or None
      - None, checks all fields or assumes that the input array is a singleton.
      - String, a single field name, if the column names are known.
      - List,  a list of field names.
    deci : integer
        An attempt to format floats with deci(mal) places.

    Requires
    --------
    _get_numeric_fields : function
        Returns the numeric fields in a structured/recarray.
    _calc_stats : function
        Performs the actual field calculations.
    """
    if isinstance(fields, str):
        fields = [fields]
    num_flds = _get_numeric_fields(a, fields)
    # ---- made it thus far
    if len(num_flds) == 0:
        num_flds = ["array"]
        s_lst = [calc_stats(a.ravel(), axis=None, deci=deci)]
    else:
        s_lst = [calc_stats(a[fld], deci=deci) for fld in num_flds]
    #
    dts = [("Statistic", "U10")] + [(i, "<f8") for i in num_flds]
    col_names = np.array(["N (size)", "n (nans)", "sum", "min", "max", "mean",
                          "median", "std", "var"])
    z = np.zeros((len(col_names),), dtype=dts)
    z["Statistic"] = col_names
    N = len(num_flds)
    for i in range(N):
        fld = num_flds[i]
        z[fld] = s_lst[i]
    if verbose:
        args = ["="*25, "Numeric fields"]
        print("\n{}\nStatistics for... a\n{!s:>32}".format(*args))
        npg_io.prn_tbl(z)
    return z


def group_stats(a, case_fld=None, num_flds=None, deci=2, verbose=False):
    """Group column statistics.

    Parameters
    ----------
    a : structured/recarray
        Make sure that you know the field names in advance.
    case_fld : string, list
        String, summarized by the unique values in the case_fld.
        List, to further fine-tune the selection or crosstabulation.
    num_flds : string, list
        You can limit the input fields accordingly, if you only need a few
        know numeric fields.

    Requires
    --------
    col_stats : function ... which requires
      : _get_numeric_fields : function
          returns the numeric fields in a structured/recarray
      : _calc_stats : function
          performs the actual field calculations
    """
    results = []
    uniq, counts = np.unique(a[case_fld], return_counts=True)
    n = len(uniq)
    if num_flds is None:
        num_flds = _get_numeric_fields(a, None)
    for i in range(n):
        u = uniq[i]
        if counts[i] >= 1:
            sub = a[a[case_fld] == u]
            z = col_stats(sub, fields=num_flds, deci=deci)
            if verbose:
                args = ["="*25, u, "Numeric fields"]
                print("\n{}\nStatistics for... a[{}]\n{!s:>32}".format(*args))
                npg_io.prn_tbl(z)
            results.append([u, z])
        else:
            print("\nToo few cases... ({}) for a[{}]...".format(counts[i], u))
    return results


# ---- (2) identify functions
def find_a_in_b(a, b, a_fields=None, b_fields=None):
    """Find the indices of the elements in a smaller 2d array contained in
    a larger 2d array. If the arrays are stuctured with field names,then these
    need to be specified.  It should go without saying that the dtypes need to
    be the same.

    Parameters
    ----------
    a, b : 1D or 2D, ndarray or structured/record arrays
        The arrays are arranged so that `a` is the smallest and `b` is the
        largest.  If the arrays are stuctured with field names, then these
        need to be specified.  It should go without saying that the dtypes
        need to be the same.
    a_fields, b_fields : list of field names
        If the dtype has names, specify these in a list.  Both do not need
        names.

    Examples
    --------
    To demonstrate, a small array was made from the last 10 records of a larger
    array to check that they could be found.

    >>> a.dtype # ([("ID", "<i4"), ("X", "<f8"), ("Y", "<f8"), ("Z", "<f8")])
    >>> b.dtype # ([("X", "<f8"), ("Y", "<f8")])
    >>> a.shape, b.shape # ((69688,), (10,))
    >>> find_a_in_b(a, b, flds, flds)
    array([69678, 69679, 69680, 69681, 69682,
           69683, 69684, 69685, 69686, 69687], dtype=int64)

    References
    ----------
    This is a function from the arraytools.tbl module.

    `<https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-
    several-values-in-a-numpy-array/38674038#38674038>`_.
    """
    def struct2nd(a):
        """from the same name in arraytools"""
        return a.view((a.dtype[0], len(a.dtype.names)))
    #
    small, big = [a, b]
    if a.size > b.size:
        small, big = [b, a]
    if a_fields is not None:
        small = small[a_fields]
        small = struct2nd(small)
    if b_fields is not None:
        big = big[b_fields]
        big = struct2nd(big)
    if a.ndim == 1:  # last slice, if  [:2] instead, it returns both indices
        indices = np.where((big == small).all(-1))[0]
    elif a.ndim == 2:
        indices = np.where((big == small[:, None]).all(-1))[1]
    return indices


def find_in(a, col, what, where="in", any_case=True, pull="all"):
    """Query a recarray/structured array for values

    Parameters
    ----------
    a : recarray/structured array
        Only text columns can be queried
    col : column/field to query
        Only 1 field can be queried at a time for the condition.
    what : string or number
        The query.  If a number, the field is temporarily converted to a
        text representation for the query.
    where : string
        s, i, eq, en  st(arts with), in, eq(ual), en(ds with)
    any_case : boolean
        True, will find records regardless of ``case``, applies to text fields
    extract : text or list
        - `all`,  extracts all records where the column case is found
        - `list`, extracts the records for only those fields in the list
    Example
    -------
    >>> find_text(a, col=`FULLNAME`, what=`ABBEY`, pull=a.dtype.names[:2])
    """
    # ---- error checking section ----
    e0 = """
    Query error: You provided...
    dtype: {}  col: {} what: {}  where: {}  any_case: {}  extract: {}
    Required...\n{}
    """
    if a is None:
        return a
    err1 = "\nField not found:\nQuery fields: {}\nArray fields: {}"
    errors = [a.dtype.names is None,
              col is None, what is None,
              where.lower()[:2] not in ("en", "eq", "in", "st")]
    if a.dtype.names is not None:
        errors += [col not in a.dtype.names]
    if sum(errors) > 0:
        arg = [a.dtype.kind, col, what, where, any_case, pull, find_in.__doc__]
        print(dedent(e0).format(*arg))
        return None
    if isinstance(pull, (list, tuple)):
        names = a.dtype.names
        r = [i in names for i in pull]
        if sum(r) != len(r):
            print(err1.format(pull, names))
            return None
    # ---- query section
    # convert column values and query to lowercase, if text, then query
    c = a[col]
    if c.dtype.kind in ("i", "f", "c"):
        c = c.astype("U")
        what = str(what)
    elif any_case:
        c = np.char.lower(c)
        what = what.lower()
    where = where.lower()[0]
    if where == "i":
        q = np.char.find(c, what) >= 0   # ---- is in query ----
    elif where == "s":
        q = np.char.startswith(c, what)  # ---- startswith query ----
    elif where == "eq":
        q = np.char.equal(c, what)
    elif where == "en":
        q = np.char.endswith(c, what)    # ---- endswith query ----
    if q.sum() == 0:
        print("none found")
        return None
    if pull == "all":
        return a[q]
    pull = np.unique([col] + list(pull))
    return a[q][pull]


# ---- sorting and slicing --------------------------------------------------
# ---- (1) row sorting and slicing
def split_sort_slice(a, split_fld=None, order_fld=None):
    """Split a structured array into groups of common values based on the
    split_fld, key field.  Once the array is split, the array is sorted on a
    val_fld and sliced for the largest or smallest `num` records.

    See Also
    --------
    Documentation is shown in `group_sort`

    """
    def _split_(a, fld):
        """split unsorted array"""
        out = []
        uni, _ = np.unique(a[fld], True)
        for _, j in enumerate(uni):
            key = (a[fld] == j)
            out.append(a[key])
        return out
    #
    err_0 = """
    A structured/recarray with a split_field and a order_fld is required.
    You provided\n    array type  : {}"""
    err_1 = """
    split_field : {}
    order field : {}
    """
    if a.dtype.names is None:
        print(err_0.format(type(a)))
        return a
    if split_fld is None:
        print("No split_fld")
        return a
    elif not isinstance(split_fld, (list, tuple, np.ndarray)):
        split_fld = [split_fld]
    if order_fld is None:
        order_fld = split_fld
        if len(split_fld) > 1:
            order_fld = split_fld[0]
    checks = split_fld + [order_fld]
    check = sum([i in a.dtype.names for i in checks])
    if check < 2:
        print((err_0 + err_1).format(type(a), split_fld, order_fld))
        return a
    #
    subs = _split_(a, split_fld)
    ordered = []
    for _, sub in enumerate(subs):
        r = sub[np.argsort(sub, order=order_fld)]
        ordered.append(r)
    return ordered


def group_sort(a, group_fld, sort_fld=None, ascend=True, sort_name=None):
    """Group records in an structured array and sort on the sort_field.  The
    order of the grouping field will be in ascending order, but the order of
    the sort_fld can sort internally within the group.

    Parameters
    ----------
    a : structured/recarray
        Array must have field names to enable splitting on and sorting by
    group_fld : string or list/tuple
        The field/name in the dtype used to identify groupings of features
    sort_fld : string
        As above, but this field contains the values that you want to sort on.
    ascend : boolean
        **True**, sorts in ascending order, so you can slice for the lowest
        `num` records. **False**, sorts in descending order if you want to
        slice the top `num` records
    sort_name : text
        Name to give the new sorted order field

    Example
    -------
    >>> fn = "C:/Git_Dan/arraytools/Data/pnts_in_poly.npy"
    >>> a = np.load(fn)
    >>> out = _split_sort_slice_(a, split_fld="Grid_codes", val_fld="Norm")
    >>> arcpy.da.NumPyArrayToFeatureClass(out, out_fc, ["Xs", "Ys"], "2951")

    References
    ----------
    `<https://community.esri.com/blogs/dan_patterson/2019/01/29/split-sort-
    slice-the-top-x-in-y>`_.

    `<https://community.esri.com/thread/227915-how-to-extract-top-five-max-
    points>`_
    """
    ordered = split_sort_slice(a, split_fld=group_fld, order_fld=sort_fld)
    if not ascend:
        ordered = [i[::-1] for i in ordered]
    final = rfn.stack_arrays(ordered, usemask=False)
    if sort_name is None:
        sort_name = 'Order'
    if sort_name in final.dtype.names:
        sort_name += "_"
    n = final.shape[0]
    final = rfn.append_fields(final,
                              sort_name,
                              np.arange(n),
                              usemask=False,
                              asrecarray=False
                              )
    return final


def n_largest_vals(a, group_fld=None, val_fld=None, num=1):
    """Run `split_sort_slice` to get the N largest values in the array.
    """
    ordered = split_sort_slice(a, split_fld=group_fld, order_fld=val_fld)
    final = []
    for r in ordered:
        r = r[::-1]
        num = min(num, r.size)
        final.extend(r[:num])
    return np.asarray(final)


def n_smallest_vals(a, group_fld=None, val_fld=None, num=1):
    """Run `split_sort_slice` to get the N smallest values in the array.
    """
    ordered = split_sort_slice(a, split_fld=group_fld, order_fld=val_fld)
    final = []
    for r in ordered:
        num = min(num, r.size)
        final.extend(r[:num])
    return np.asarray(final)


# ---- (x) field appending
#
def _field_specs(a):
    """Produce a list of name/dtype pairs for fields in a structured array.
    Derived from rfn._get_fieldspec
    The input is a structured array `a`, rather than the dtype

    Notes
    -----
    >>> z = np.arange(0, 27).reshape(9,3)
    >>> z0 = uts(z, names=['a', 'b', 'c'])  # rfn.unstructured_to_structured
    >>> _field_specs(z)
    ... ['No_name', dtype('int32')]
    >>> _field_specs(z0)
    ... [('a', dtype('int32')), ('b', dtype('int32')), ('c', dtype('int32'))]

    See also
    --------
    np.lib.recfunctions _get_fieldspec(dtype) for more information.
    """
    dt = a.dtype
    if dt.names is None:
        return [('No_name'), dt]
    fields = ((name, dt.fields[name]) for name in dt.names)
    fld_spec = [(name if len(f) == 2 else (f[2], name), f[0])
                for name, f in fields]
    return fld_spec


def _append_fields(a, fld_names, fld_values=None):
    """
    Add fields to an existing structured array returning a new array with
    the new field(s).

    If the number of rows in the destination array and the
    number of fld_values are not equal a ValueError will be returned.
    """
    msg = "\nValueError\nThe number of fld_names and fld_values don't match."
    fld_spec = _field_specs(a)
    if isinstance(fld_names, str):
        fld_names = [fld_names, ]
        if isinstance(fld_values, np.ndarray):
            if len(fld_values.shape) > 1:
                print(msg)
                return None
            fld_values = [fld_values, ]
    if len(fld_names) != len(fld_values):
        print(msg)
        return None
    to_keep = []
    for i, vals in enumerate(fld_values):
        if a.size == vals.size:
            data = np.array(vals, copy=False, subok=True).ravel()
            to_keep.append([fld_names[i], data])
    if len(to_keep) == 0:
        print("\nValueError\nArrays are not of the appropriate size.")
        return None
    data = [d.view([(name, d.dtype)]) for (name, d) in to_keep]
    n = len(data)
    if n == 1:
        new_spec = _field_specs(data[0])
    else:
        new_spec = []
        for i in data:
            new_spec.extend(_field_specs(i))
    dt = np.dtype(fld_spec + new_spec)
    names = list(dt.names)
    out_array = np.full_like(a, 0, dtype=dt)  # the new array with new dtype
    out_array[names[:-n]] = a
    for i in data:
        _fill_fields(i, out_array)
    return out_array


def merge_arrays(a, others):
    """merge a list of structured arrays
    """
    if not isinstance(others, (list, tuple)):
        others = [others, ]
    n = len(others)
    first = others.pop(0)
    names = first.dtype.names
    data = [first[name] for name in names]
#    for ar in arrays[1:]:
#        names.append(ar.dtype.names)
#        ar_names = ar.dtype.names
#        for nm in ar_names:
#            data.append(ar[nm])
#        names.append(ar_names)
#    return _append_fields(a, names, data)


# ---- (2) field slicing
#
def _fill_fields(in_arr, out_arr):
    """Fill an output array fields from an input array selection

    Parameters
    ----------
    in_arr, out_arr : structured arrays
        Both arrays must have the same number of records.  The fields of the
        output array will be filled with values from the input array.

    Simplified from ``recursively_fill_fields`` in
    `<https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py>`_.
    """
    newdtype = out_arr.dtype
    for field in newdtype.names:
        try:
            current = in_arr[field]
        except ValueError:
            continue
        if current.dtype.names is not None:
            _fill_fields(current, out_arr[field])
        else:
            out_arr[field][:len(current)] = current
    return out_arr


def keep_fields_by_name(in_arr, names):
    """Returns an output array with a selection of fields from the input array

    Parameters
    ----------
    in_arr : structured array
        The array to pull fields/columns from.
    names : list/tuple
        The field names from the input array to retain.

    Notes
    -----
    simplified from ``recursively_fill_fields`` in
    `<https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py>`_.
    """
    newdtype = [(n, in_arr.dtype[n]) for n in names]
    out_arr = np.empty(in_arr.shape, dtype=newdtype)
    out_arr = _fill_fields(in_arr, out_arr)
    return out_arr


def keep_fields_by_kind(in_arr, field_kind=("i", "f", "U")):
    """Reorder fields in a structured array by type and returns those that meet
    the requirement.

    Parameters
    ----------
    in_arr : structured array
        The array to pull fields/columns from.
    field_kind : list/tuple
        i(nteger), f(loat) and U(nicode) aka string

    Notes
    -----
    Omit the kind you do not wish to carry over.
    The order that they are entered will be reflected in the output.
    """
    dt = in_arr.dtype
    dt_names = np.asarray(dt.names)
    dt_kind = np.asarray([dt.fields[name][0].kind for name in dt_names])
    uni, idx = np.unique(dt_kind, True)
    to_keep = []
    if not isinstance(field_kind, (list, tuple)):
        field_kind = [field_kind]
    for k in field_kind:
        if (k in field_kind) and (k in uni):
            to_keep.extend(dt_names[dt_kind == k].tolist())
    out = keep_fields_by_name(in_arr, to_keep)
    return out


# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # msg = _demo_()
