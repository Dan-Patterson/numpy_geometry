# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
---------
npg_split
---------

----

Script :
    npg_split.py

Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2021-01-10

Purpose
-------
Functions for clipping geomgons.
"""

import sys
# from textwrap import dedent
import numpy as np

# -- optional numpy imports
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import repack_fields

if 'npg' not in list(locals().keys()):
    import npg
from npg_helpers import _to_lists_
from npg_plots import plot_polygons

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['_split_']


# ----------------------------------------------------------------------------
# ---- (1)
#
def _split_(splitter, geom, return_pnts=True):
    """Return if and where line segments cross.  Used for clipping.

    Parameters
    ----------
    splitter, geom : ndarrays
        Arrays representing the clip geometry and the geometry being clipped.

    Notes
    -----
    `right` = inside
    `left`  = outside
    For clockwise ordered polygon segments.

    Multi-line implementation of line_crosses.
    Used by ``clip_polygons`` via ``_clip_.

    from `p_c_p` and `_line_crossing_`
    a_num = a_0 - a1 from below
    """
    #
    p_cl, c_cl = [i.XY if hasattr(i, "IFT") else i for i in [geom, splitter]]
    # check = _wn_(p_cl, c_cl, False)  # full version of _chk_
    # if check.size < 1:  # no points in splitter
    #     return None, None, None
    x0s, y0s = geom[:-1].T        # NOTE
    x1s, y1s = geom[1:]. T        # using notation from: _clip_line_crosses_
    x2s, y2s = splitter[:-1].T
    x3s, y3s = splitter[1:].T
    dc_x = x3s - x2s
    dc_y = y3s - y2s
    a_0 = (y0s - y2s[:, None]) * dc_x[:, None]
    a_1 = (x0s - x2s[:, None]) * dc_y[:, None]
    b_0 = (y1s - y2s[:, None]) * dc_x[:, None]
    b_1 = (x1s - x2s[:, None]) * dc_y[:, None]
    a = a_0 <= a_1
    b = b_0 <= b_1
    # w0 = np.logical_and(a, b) * 2     # both on right  (T, T)
    w1 = np.logical_and(a, ~b) * 1      # start on right (T, F)
    w2 = np.logical_and(~a, b) * -1     # start on left  (F, T)
    # w3 = np.logical_and(~a, ~b) * -2  # both on left   (F, F)
    z = w1 + w2  # w0 + w1 + w2 + w3
    whr = np.argwhere(abs(z) == 1)
    #
    if not return_pnts:
        return z, whr, None
    denom = (x1s - x0s) * dc_y[:, None] - (y1s - y0s) * dc_x[:, None]
    with np.errstate(all="ignore"):  # ignore all errors
        ua = (a_0 - a_1)/denom
        x_int = x0s[None, :] + (x1s-x0s)[None, :] * ua
        y_int = y0s[None, :] + (y1s-y0s)[None, :] * ua
        wh = w1 + w2
        whr = np.vstack(wh.nonzero()).T
        xs = x_int[whr[:, 0], whr[:, 1]]
        ys = y_int[whr[:, 0], whr[:, 1]]
        pnts = np.concatenate((xs[:, None], ys[:, None]), axis=1)
    return z, whr, pnts


def _cut_(splitter, geom):
    """Return the result of a polygon split.

    Parameters
    ----------
    splitter, geom : geometry
        `splitter` is a line which crosses two segments of geom.
        `geom` is a polygon feature.

    Notes
    -----
    Split into chunks.

    >>> spl = np.array([[ 0.0, 7.5], [10.0, 2.5]])
    >>> sq = np. array([[ 0.0, 0.0], [ 0.0, 10.0], [10.0, 10.0],
    ...                 [10.0, 0.0], [ 0.0,  0.0]])
    >>> result = _c_(spl, sq)
    >>> result
    ... array([[ 0.0, 0.0], [ 0.0, 7.5], [10.0, 2.5],
    ...        [10.0, 0.0], [ 0.0, 0.0]])

    Reassemble an from-to array back to an N-2 array of coordinates. Also,
    split, intersection points into pairs.

    >>> coords = np.concatenate((r0[0, :2][None, :], r0[:, -2:]), axis=0)
    >>> from_to = [o_i[i: i+2] for i in range(0, len(o_i), 2)]
    """
    ft_p = np.concatenate((geom[:-1], geom[1:]), axis=1)
    _, whr, x_pnts = _split_(splitter, geom)
    #
    if x_pnts is None:
        return None
    #
    uni, idx, cnts = np.unique(whr[:, 0], True, return_counts=True)
    crossings = uni[cnts >= 2]  # a line has to produce at least 2 x_sections
    #
    polys = []
    is_first = True
    ps = np.copy(ft_p)
    for i, seg in enumerate(crossings):  # clipper ids that cross poly
        w = whr[:, 0] == seg
        out_in = whr[w]
        all_pairs = x_pnts[w]
        chunks = [out_in[i: i+2] for i in range(0, len(out_in), 2)]
        pair_chunk = [all_pairs[i: i+2] for i in range(0, len(out_in), 2)]
        for j, o_i in enumerate(chunks):
            pairs = pair_chunk[j]
            if is_first:
                p0, p1 = o_i[:, 1]
                is_first = False  # cancel first
            else:
                o_i = o_i[:, 1] - (ft_p.shape[0] - ps.shape[0])
                p0, p1 = o_i
            if (p1 - p0) >= 2:    # slice out extra rows, but update sp first
                sp = ps[p0: p1 + 1]
                ps = np.concatenate((ps[:p0 + 1], ps[p1:]))
            else:
                sp = ps[p0:p1 + 1]
                ps = np.copy(ps)
            ps[p0, -2:] = pairs[0]
            ps_new = pairs[:2].ravel()
            ps[p0 + 1, :2] = pairs[1]
            #
            pieces = [ps[:(p0 + 1)], ps_new, ps[(p0 + 1):]]
            z0 = [np.atleast_2d(i) for i in pieces]
            z0 = np.concatenate(z0, axis=0)
            z_0 = np.concatenate((z0[0, :2][None, :], z0[:, -2:]))
            #
            ps = np.copy(z0)  # copy if there is more than 1 chunk!!
            #
            sp[0, :2] = pairs[0]
            sp[-1, -2:] = pairs[1]
            sp_new = np.concatenate((pairs[1], pairs[0]))
            z1 = np.concatenate((sp, sp_new[None, :]), axis=0)
            z_1 = np.concatenate((z1[0, :2][None, :], z1[:, -2:]))
            polys.extend([z_0, z_1])
    return polys


def split_polygon(splitters, geom):
    """Return the result of a polygon clip.

    Parameters
    ----------
    splitters, geom : geometry
        Splitters are lines.  A line consists of pairs of from-to, (x, y)
        coordinates of array shape (2, 2).
        `geom` is a polygon features with at least 3 self-closing sides.  A
        triangle would consist of 3 side made up of 4 points with the first
        and last point being replicates.
    """
    if isinstance(splitters, (list, tuple)):
        splitters = np.asarray(splitters)
    if (splitters.ndim == 2) and (splitters.shape[1] == 2):
        ft_s = np.concatenate((splitters[:-1], splitters[1:]), axis=1)
        splitters = [i.reshape(2, 2) for i in ft_s]
    out = []
    for splitter in splitters:
        result = _cut_(splitter, geom)
        if result:
            out.append(np.asarray(result, dtype='object').squeeze())
    return out
