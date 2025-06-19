# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
------------
npg_helpers
-----------

** General helpers for the npg package and ndarrays in general.

----

Script :
    npg_helpers.py

Author :
    `<https://github.com/Dan-Patterson>`_.

Modified :
    2025-06-01

Purpose
-------
General helper functions.

Notes
-----
To add
"""

import sys
import numpy as np
import npg  # noqa

script = sys.argv[0]

__all__ = [
    'cartesian_product',               # (2) main functions
    'drop_seq_dupl',
    'separate_string_number',
    'sequences',
    'stride_2d',
    'uniq_1d',
    'uniq_2d',
    'flatten',
    'unpack'
]

__helpers__ = [                        # (1) private helpers
    '_base_',
    '_isin_2d_',
    '_iterate_',
    '_to_lists_',
    '_view_as_struct_'
]

# __imports__ = ['roll_arrays']


# ---- ---------------------------
# ---- (1) private helpers
def _base_(a):
    """Return the base array of a Geo array.  Shave off microseconds."""
    if hasattr(a, "IFT"):
        return a.XY
    return a


def _isin_2d_(a, b, as_integer=False):
    """Perform a 2d `isin` check for 2 arrays.

    Parameters
    ----------
    a, b : arrays
        The arrays to compare.
    as_integer : boolean
        False, returns a list of booleans.  True, returns an integer array
        which may useful for some operations.

    Example
    -------
    >>> a = np.array([[ 5.00,  10.00], [ 5.00,  12.00], [ 6.00,  12.00],
                      [ 8.00,  12.00], [ 8.00,  11.00], [ 5.00,  10.00]])
    >>> b = np.array([[ 5.00,  12.00], [ 5.00,  15.00], [ 7.00,  14.00],
                      [ 6.00,  12.00], [ 5.00,  12.00]])
    >>> w0 = (a[:, None] == b).all(-1).any(-1)
    array([0, 1, 1, 0, 0, 0])
    >>> a[w0]
    array([[ 5.00,  12.00], [ 6.00,  12.00]])
    >>> w1 = (b[:, None] == a).all(-1).any(-1)
    >>> b[w1]
    array([[ 5.00,  12.00], [ 6.00,  12.00], [ 5.00,  12.00]])

    Reference
    ---------
    `<https://stackoverflow.com/a/51352806/6828711>`_.
    """
    a = _base_(a)
    b = _base_(b)
    out = (a[:, None] == b).all(-1).any(-1)
    if as_integer:
        return out.astype('int')
    return out.tolist()


def _iterate_(N, n):
    """Return combinations for array lengths."""
    import itertools
    combos = itertools.combinations(np.arange(N), n)
    return list(combos)


def _to_lists_(a, outer_only=True):
    """Return list or list of lists for a Geo or ndarray.

    Parameters
    ----------
    a : array-like
        Either a Geo array or ndarray.
    outer_only : boolean
        True, returns the outer-rings of a Geo array.  False, returns the bit.

    See Also
    --------
    `Geo_to_lists`, `Geo_to_arrays` if you want to maintain the potentially
    nested structure of the geometry.
    """
    if hasattr(a, "IFT"):
        if outer_only:
            return a.outer_rings(False)  # a.bits
        return a.bits
    if isinstance(a, np.ndarray):
        if a.dtype.kind == 'O':
            return a
        if a.ndim == 2:
            return [a]
        if a.ndim == 3:
            return list(a)
    return a  # a list already


def _view_as_struct_(a, return_all=False):
    """Key function to get uniform 2d arrays to be viewed as structured arrays.

    A bit of trickery, but it works for all set-like functionality.
    Use `uts` for more complicated dtypes.

    Parameters
    ----------
    a : array
        Geo array or ndarray to be viewed.

    Returns
    -------
    Array view as structured/recarray, with shape = (N, 1)

    References
    ----------
    See `unstructured_to_structured` in... numpy/lib/recfunctions.py

    >>> from numpy.lib.recfunctions import unstructured_to_structured as uts
    """
    shp = a.shape
    dt = a.dtype
    a_view = a.view(dt.descr * shp[1])[..., 0]
    if return_all:
        return a_view, shp, dt
    return a_view


# ---------------------------
# ---- (2) main functions
#
#

def cartesian_product(sequences):
    """Construct an index grid using 1D array_like sequences.

    arrays : array_like
        At least 2 array_like sequences to form the indices/product.

    Example
    -------
    >>> cartesian_product([[0, 1]), [0, 1, 2]])
    ...array([[0, 0],
    ...       [0, 1],
    ...       [0, 2],
    ...       [1, 0],
    ...       [1, 1],
    ...       [1, 2]])
    >>> cartesian_product([[0], [2, 3], [5, 4]])
    ...array([[0, 2, 5],
    ...       [0, 2, 4],
    ...       [0, 3, 5],
    ...       [0, 3, 4]])

    Reference
    ---------
    `<https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and
    -y-array-points-into-single-array-of-2d-points>`_.
    """
    arrays = [np.array(i) for i in sequences]
    len_ = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [len_], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, len_)


def drop_seq_dupl(a):
    """Remove sequential duplicates from an array.

    The array is stacked with the first value in the sequence to retain it.
    Designed to removed sequential duplicates in point arrays.
    """
    uni = a[np.where(a[:-1] != a[1:])[0] + 1]
    if a.ndim == 1:
        uni = np.hstack((a[0], uni))
    else:
        uni = np.vstack((a[0], uni))
    uni = np.ascontiguousarray(uni)
    return uni


def separate_string_number(string, as_list=False):
    """Return a string split into strings and numbers, as a list.

    z = 'Pj 60Pt 30Bw 10'
    z0 = 'PJ60PT30BW10'
    separate_string_number(z)
    separate_string_number(z0)
    returned value
    ['Pj', '60', 'Pt', '30', 'Bw', '10']

    separate_string_number("A .1 in the 1.1 is not 1")
    ['A', '.1', 'in', 'the', '1.1', 'is', 'not', '1']

    Modified from https://stackoverflow.com/a/57359921/6828711
    """
    groups = []
    prev = string[0]
    newword = string[0]
    if len(string) <= 1:
        return [string]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and prev.isalpha():
            newword += i
        elif (i.isnumeric() or i == '.') and (prev.isnumeric() or prev == '.'):
            newword += i
        else:
            groups.append(newword.strip())
            newword = i
        prev = i
        if x == len(string) - 2:
            groups.append(newword.strip())  # strip any spaces
            newword = ''
    # remove extraneous space values in groups
    groups = [i for i in groups if i != '']
    # -- for arrays
    # np.asarray(np.split(groups, groups.size//2)  # split into pairs.)
    if as_list:
        return groups
    # -- pair values, special case
    s = " ".join(["".join(groups[pos:pos + 2])
                  for pos in range(0, len(groups), 2)]
                 )
    return s


def sequences(data, stepsize=0):
    """Return an array of sequence information denoted by stepsize.

    Parameters
    ----------
    data : array-like
        List/array of values in 1D
    stepsize : integer
        Separation between the values.
    If stepsize=0, sequences of equal values will be searched.  If stepsize
    is 1, then sequences incrementing by 1 etcetera.

    Stepsize can be both positive or negative::

        >>> # check for incrementing sequence by 1's
        >>> d = [1, 2, 3, 4, 4, 5]
        >>> s = sequences(d, 1)
        |array([(0, 0, 4, 1, 4), (1, 4, 6, 4, 2)],
        |      dtype=[('ID', '<i4'), ('From_', '<i4'), ('To_', '<i4'),
        |             ('Value', '<i4'), ('Count', '<i4')])
        >>> npg.prn(s)
        id  ID    From_   To_   Value   Count
        ----------------------------------------
        000     0       0     4       1       4
        001     1       4     6       4       2

    Notes
    -----
    For strings, use

    >>> partitions = np.where(a[1:] != a[:-1])[0] + 1

    Change **N** in the expression to find other splits in the data

    >>> np.split(data, np.where(np.abs(np.diff(data)) >= N)[0]+1)

    Keep for now::

        checking sequences of 0, 1
        >>> a = np.array([1,0,1,1,1,0,0,0,0,1,1,0,0])
        | np.hstack([[x.sum(), *[0]*(len(x) -1)]
        |           if x[0] == 1
        |           else x
        |           for x in np.split(a, np.where(np.diff(a) != 0)[0]+1)])
        >>> # array([1, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0])

    References
    ----------
    `<https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-
    sequences-elements-from-an-array-in-numpy>`__.
    """
    #
    a = np.array(data)
    a_dt = a.dtype.kind
    dt = [('ID', '<i4'), ('From_', '<i4'), ('To_', '<i4'),
          ('Value', a.dtype.str), ('Count', '<i4'),
          ]
    if a_dt in ('U', 'S'):
        seqs = np.split(a, np.where(a[1:] != a[:-1])[0] + 1)
    elif a_dt in ('i', 'f'):
        seqs = np.split(a, np.where(np.diff(a) != stepsize)[0] + 1)
    vals = [i[0] for i in seqs]
    cnts = [len(i) for i in seqs]
    seq_num = np.arange(len(cnts))
    too = np.cumsum(cnts)
    frum = np.zeros_like(too)
    frum[1:] = too[:-1]
    out = np.array(list(zip(seq_num, frum, too, vals, cnts)), dtype=dt)
    return out


def stride_2d(a, win=(2, 2), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of a 2D array.

    Parameters
    ----------
    a : 2D array
    win : tuple
        Window size in terms of rows and columns.
    stepby : tuple
        The steps to take in the X and Y direction along the array.

    Notes
    -----
    You can ravel a 2D array to facilitate moving from row to row.  See the
    examples.  Alternately, you can use moving blocks to obtain things like
    statistical parameters on `raster` data.

    This function is coming in version 1.20 in numpy.lib.stride_tricks.py::

        sliding_window_view(x, window_shape, axis=None, *,
                            subok=False, writeable=False):

    Example
    -------
    Create from-to points::

        # -- produces a `view` and not a copy
        >>> a = np.array([[0, 1], [2, 3], [4, 5]])
        >>> stride_2d(a.ravel(), win=(4,), stepby=(2,))

        array([[0, 1, 2, 3],
               [2, 3, 4, 5]])

        # -- alternatives, but they produce copies
        >>> np.concatenate((a[:-1], a[1:]), axis=1)
        >>> np.asarray(list(zip(a[:-1], a[1:])))

        # -- concatenate is faster, with 500 points in `s`.
        %timeit stride_2d(s.ravel(), win=(4,), stepby=(2,))
        21.7 µs ± 476 ns per loop (mean ± std. dev. of 7 runs, 10000 loops

        %timeit np.concatenate((s[:-1], s[1:]), axis=1)
        8.41 µs ± 158 ns per loop (mean ± std. dev. of 7 runs, 100000 loops

    A different stride::

        >>> stride_2d(b, win=(2, 2), stepby=(1, 1))
        array([[[0, 1],
                [2, 3]],

               [[2, 3],
                [4, 5]]])

    """
    from numpy.lib.stride_tricks import as_strided
    shp = np.array(a.shape)    # array shape 2D (r, c) or 3D (d, r, c)
    win_shp = np.array(win)    # window    (4,) (3, 3) or    (1, 3, 3)
    ss = np.array(stepby)      # step by   (2,) (1, 1) or    (1, 1, 1)
    newshape = tuple(((shp - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides, subok=True)
    return a_s.squeeze()


def uniq_1d(arr):
    """Return mini `unique` 1D."""
    mask = np.empty(arr.shape, dtype=np.bool_)
    mask[:1] = True
    a_copy = np.sort(arr)
    mask[1:] = a_copy[1:] != a_copy[:-1]
    return a_copy[mask]


def uniq_2d(arr, return_sorted=False):  # *** keep but slower than unique
    """Return mini `unique` for 2D coordinates.  Derived from np.unique.

    Notes
    -----
    For returning in the original order this is equivalent to::

        u, idx = np.unique(x_pnts, return_index=True, axis=0)
        x_pnts[np.sort(idx)]

    References
    ----------
    `NumPy unique
    <https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/
    arraysetops.py#L138-L320>`_.
    """
    def _reshape_uniq_(uniq, dt, shp):
        n = len(uniq)
        uniq = uniq.view(dt)
        uniq = uniq.reshape(n, *shp[1:])
        uniq = np.moveaxis(uniq, 0, 0)
        return uniq

    shp = arr.shape
    dt = arr.dtype
    st_arr = arr.view(dt.descr * shp[1])
    ar = st_arr.flatten()
    if return_sorted:
        perm = ar.argsort(kind='mergesort')
        aux = ar[perm]
    else:  # removed ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    ret = aux[mask]
    uniq = _reshape_uniq_(ret, dt, shp)
    if return_sorted:  # return_index in unique
        return uniq, perm[mask]
    return uniq


def flatten(a_list, flat_list=None):
    """Change the isinstance as appropriate.

    :  Flatten an object using recursion
    :  see: itertools.chain() for an alternate method of flattening.
    """
    if flat_list is None:
        flat_list = []
    for item in a_list:
        if isinstance(item, list):
            flatten(item, flat_list)
        else:
            flat_list.append(item)
    return flat_list


def unpack(iterable, param='__iter__'):
    """Unpack an iterable based on the param(eter) condition using recursion.

    :Notes:
    : ---- see main docs for more information and options ----
    : To produce an array from this, use the following after this is done.
    :   out = np.array(xy).reshape(len(xy)//2, 2)
    """
    xy = []
    for x in iterable:
        if hasattr(x, '__iter__'):
            xy.extend(unpack(x))
        else:
            xy.append(x)
    return xy


# ---- ---------------------------
# ---- Final main section
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")
