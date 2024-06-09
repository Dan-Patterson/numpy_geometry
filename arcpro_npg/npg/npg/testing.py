# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:31:53 2024

@author: dan_p
"""

import sys
import numpy as np
import npg
from npg.npGeo import roll_arrays
from npg.npg_plots import plot_polygons  # noqa
from copy import deepcopy
from npg_bool_ops import add_intersections, connections_dict, _renumber_pnts_

def tst(poly, clp):
    result = add_intersections(
                poly, clp,
                roll_to_minX=True,
                p0_pgon=True, p1_pgon=True,
                class_ids=False
                )
    # pl_ioo polygon in-out-on, cl_ioo for clip
    pl_n, cl_n, id_plcl, onConP, x_pnts, ps_info, cs_info = result
    p_out, p_on, p_in, pl_ioo = ps_info
    c_out, c_on, c_in, cl_ioo = cs_info
    #
    p_col = pl_ioo[:, 1]  # noqa
    c_col = cl_ioo[:, 1]  # noqa
    #
    N_c = cl_n.shape[0] - 1  # noqa
    N_p = pl_n.shape[0] - 1  # noqa
    #
    p_ft = list(zip(p_on[:-1], p_on[1:] + 1))
    p_subs = [pl_ioo[i[0]: (i[1])] for i in p_ft]
    p_vals = [sub[1][1] for sub in p_subs]
    p_ft_v = np.concatenate((p_ft, np.array(p_vals)[:, None]), axis=1)
    #
    #
    c_ft = list(zip(c_on[:-1], c_on[1:] + 1))
    c_subs = [cl_ioo[i[0]:i[1]] for i in c_ft]
    c_vals = [sub[1][1] for sub in c_subs]
    c_ft_v = np.concatenate((c_ft, np.array(c_vals)[:, None]), axis=1)
    #
    new_ids, old_new, CP_ = _renumber_pnts_(cl_n, pl_n)
    #
    # -- produce the connections dictionary
    c_ft_orig = np.array(list(zip(np.arange(0, N_c), np.arange(1, N_c + 1))))
    # p_ft_orig = np.array(list(zip(np.arange(0, N_p), np.arange(1, N_p + 1))))
    p_ft_new = np.array(list(zip(old_new[:-1, 2], old_new[1:, 2])))
    ft_new = np.concatenate((c_ft_orig, p_ft_new), axis=0)
    conn_dict = connections_dict(ft_new, bidirectional=True)
    #
    # -- CP_ can be used to plot the combined, resultant polygon
    # -- poly = CP_[old_new[:, 2]]    # to plot poly from CP_
    # -- clp = CP_[old_new[:N_c + 1, 0]]    # to plot poly from CP_
    #
    # -- # uniq points and lex sorted l-r b-t
    cp_uni, idx_cp = np.unique(CP_, True, axis=0)
    lex_ids = new_ids[idx_cp]
    # -- poly, clp segments
    clp_s = [cl_n[i:j] for i,j in c_ft_v[:, :2]]
    ply_s = [pl_n[i:j] for i,j in p_ft_v[:, :2]]
    c_segs = np.arange(c_ft_v.shape[0])
    p_segs = np.arange(p_ft_v.shape[0])


    # pon_whr = np.nonzero((p_on == old_new[:, 0][:, None]).any(-1))
    # pon_new = old_new[pon_whr]

    # p_new = new_ids[N_c + 1:]
    # p_ft_new = 
    # c0, c1 = p_ft_v[:, :2].T
    # wc0 = np.nonzero((c0== old_new[:, 0][:, None]).any(-1))[0]
    # wc1 = np.nonzero((c1 == old_new[:, 0][:, None]).any(-1))[0]

    # p_fnew = old_new[c0][:, -1]
    #
    # p_out_new = p_out + N_c + 1
    # p_in_new = p_in + N_c + 1
    # p_on_new = p_on + N_c + 1

    p_ft_v_new = np.copy(p_ft_v)
    p_ft_v_new[:, 0] += N_c + 1
    p_ft_v_new[:, 1] += N_c + 1
    #
    # ft_c_p_new = np.concatenate((c_ft_v, p_ft_v), axis= 1)  # or ...
    ft_c_p_new = np.concatenate((c_ft_v, p_ft_v_new), axis= 0)
    cp_dict = connections_dict(ft_c_p_new[:, :2], bidirectional=False)  # single
    cp_dict = connections_dict(ft_c_p_new[:, :2], bidirectional=True)  # multiple
"""
onConP = np.array([[0,  0,  0,  0],
                   [1,  2,  1,  2],
                   [2,  3,  1,  1],
                   [3, 10,  1,  7],
                   [5,  9,  2, -1],
                   [6,  4,  1, -5],
                   [7,  5,  1,  1],
                   [8,  8,  1,  3],
                   [10, 15,  2,  7],
                   [11, 18,  1,  3],
                   [12, 19,  1,  1],
                   [13, 14,  1, -5],
                   [15, 13,  2, -1],
                   [16, 20,  1,  7],
                   [17, 21,  1,  1]])

id_plcl = np.array([[0,  0],
                    [2,  1],
                    [3,  2],
                    [4,  6],
                    [5,  7],
                    [8,  8],
                    [9,  5],
                    [10,  3],
                    [13, 15],
                    [14, 13],
                    [15, 10],
                    [18, 11],
                    [19, 12],
                    [20, 16],
                    [21, 17]])

x_pnts = np.array([[0.00,   0.00], [2.00,   0.00], [2.00,  10.00],
                   [2.67,   2.00], [3.33,   2.00], [3.60,   8.00],
                   [4.00,   0.00], [4.00,  10.00], [4.80,   8.00],
                   [5.50,   2.00], [6.00,   0.00], [6.00,  10.00],
                   [7.00,   8.00], [8.00,  10.00]])

cl_n = [[0.0, 0.0], [2.0, 10.0], [4.0, 10.0], [3.60, 8.0],
        [3.0, 5.0], [4.8, 8.0], [6.0, 10.0], [8.0, 10.0], [7.0, 8.0],
        [5.0, 4.0], [5.5, 2.0], [6.0, 0.0], [4.0, 0.0],
        [3.33, 2.0], [3.0, 3.0], [2.67, 2.0],
        [2.0, 0.0], [0.0, 0.0]]
cl_n = np.array(cl_n)

pl_n = [[0.0, 0.0], [0.0, 10.0], [2.0, 10.0], [4.0, 10.0], [6.0, 10.0],
        [8.0, 10.0], [10.0, 10.0], [10.0, 8.0], [7.0, 8.0], [4.8, 8.0],
        [3.6, 8.0], [2.0, 8.0], [2.0, 2.0],
        [2.67, 2.0], [3.33, 2.0], [5.5, 2.0],
        [10.0, 2.0], [10.0, 0.0], [6.0, 0.0], [4.0, 0.0], [2.0, 0.0],
        [0.0, 0.0]]

pl_n = np.array(pl_n)

CP_ = np.concatenate((cl_n, pl_n), axis=0)

# find first intersections
ids = np.arange(0, CP_.shape[0])
N_c = cl_n.shape[0] - 1
N_p = pl_n.shape[0] - 1



wp, wc = np.nonzero((cl_n == pl_n[:, None]).all(-1))  # id_plcl
#  [ 0,  0,  2,  3,  4,  5,  8,  9, 10, 13, 14, 15, 18, 19, 20, 21, 21]
#  [ 0, 17,  1,  2,  6,  7,  8,  5,  3, 15, 13, 10, 11, 12, 16,  0, 17]

wc0, wp0 = np.nonzero((pl_n == cl_n[:, None]).all(-1))  # onConP
# [ 0,  0,  1,  2,  3,  5,  6,  7,  8, 10, 11, 12, 13, 15, 16, 17, 17]
# [ 0, 21,  2,  3, 10,  9,  4,  5,  8, 15, 18, 19, 14, 13, 20,  0, 21]

wpx, xp = np.nonzero((x_pnts == pl_n[:, None]).all(-1))
# [ 0,  2,  3,  4,  5,  8,  9, 10, 13, 14, 15, 18, 19, 20, 21]
# [ 0,  2,  7, 11, 13, 12,  8,  5,  3,  4,  9, 10,  6,  1,  0]

wcx, xc = np.nonzero((x_pnts == cl_n[:, None]).all(-1))
# [ 0,  1,  2,  3,  5,  6,  7,  8, 10, 11, 12, 13, 15, 16, 17]
# [ 0,  2,  7,  5,  8, 11, 13, 12,  9, 10,  6,  4,  3,  1,  0]

# --- new attempt with x_pnts lex sorted from top left
#
x_lex_ids = np.lexsort((-x_pnts[:, 1], x_pnts[:, 0]))
x_lex = x_pnts[x_lex_ids]

wpx_lex, xp_lex = np.nonzero((x_lex == pl_n[:, None]).all(-1))
wcx_lex, xc_lex = np.nonzero((x_lex == cl_n[:, None]).all(-1))


c_ids = np.arange(0, N_c + 1)
c_ids[-1] = 0
p_ids = np.arange(0, N_p + 1)
p_ids[-1] = 0
p_ids[1:-1] = np.arange(c_ids[-2] + 1, N_c + N_p - 1)

cp_ids = np.concatenate((c_ids, p_ids), axis=0)

zz = wp0[1:-1] + N_c + 1
zz0 = list(zip(zz, wc0[1:-1]))
zz0 = np.array(zz0)
zz1 = np.copy(cp_ids)
r0, r1 = (zz1 == zz0[:, 0][:, None]).nonzero()
zz1[r1] = zz0[:, 1]
zz1[(zz1 == N_c).nonzero()] = 0

frto = np.concatenate((zz1[:-1][:, None], zz1[1:][:, None]), axis=1)
p = CP_[zz1]

# from
#  https://stackoverflow.com/questions/48705143/efficiency-2d-list-to-dictionary-in-python
# second element is the key
l = frto
d = {}
for elem in l:
    if elem[1] in d:
        d[elem[1]].append(elem[0])
    else:
        d[elem[1]] = [elem[0]]

# first element is the key
l = frto
d = {}
for elem in l:
    if elem[0] in d:
        d[elem[0]].append(elem[1])
    else:
        d[elem[0]] = [elem[1]]


#  Splitting example
# =========================
# polygons poly, clp:  E, d0_

p_ft_v = np.array([[0, 3, -1],
                   [2,  6, -1],
                   [5,  7,  0],
                   [6,  9, -1],
                   [8, 14,  1],
                   [13, 16, -1],
                   [15, 17,  0],
                   [16, 20, -1],
                   [19, 22, -1]])

c_ft_v = np.array([[0,  2,  0],
                   [1,  4, -1],
                   [3,  5,  0],
                   [4,  6,  0],
                   [5, 10, -1],
                   [9, 11,  0],
                   [10, 12,  0],
                   [11, 14, -1],
                   [13, 15,  0]])


out_p = []
for i in p_ft_v[:, :2]:
    fr_, to_ = i
    out_p.append(pl_n[fr_:to_])
out_c = []
for i in c_ft_v[:, :2]:
    fr_, to_ = i
    out_c.append(cl_n[fr_:to_])

# -- out_p

# [array([[  0.00,   5.00],
#         [  0.00,  10.00],
#         [  5.00,  10.00]]),
#  array([[  5.00,  10.00],
#         [ 10.00,  10.00],
#         [ 10.00,   8.00],
#         [  6.33,   8.00]]),
#  array([[  6.33,   8.00],
#         [  3.67,   8.00]]),
#  array([[  3.67,   8.00],
#         [  2.00,   8.00],
#         [  2.00,   6.33]]),
#  array([[  2.00,   6.33],
#         [  2.00,   5.50],
#         [  5.00,   5.50],
#         [  5.00,   4.00],
#         [  2.00,   4.00],
#         [  2.00,   3.67]]),
#  array([[  2.00,   3.67],
#         [  2.00,   2.00],
#         [  3.67,   2.00]]),
#  array([[  3.67,   2.00],
#         [  6.33,   2.00]]),
#  array([[  6.33,   2.00],
#         [ 10.00,   2.00],
#         [ 10.00,   0.00],
#         [  5.00,   0.00]]),
#  array([[  5.00,   0.00],
#         [  0.00,   0.00],
#         [  0.00,   5.00]])]

# -- out_c

# [array([[  0.00,   5.00],
#         [  2.00,   6.33]]),
#  array([[  2.00,   6.33],
#         [  3.00,   7.00],
#         [  3.67,   8.00]]),
#  array([[  3.67,   8.00],
#         [  5.00,  10.00]]),
#  array([[  5.00,  10.00],
#         [  6.33,   8.00]]),
#  array([[  6.33,   8.00],
#         [  7.00,   7.00],
#         [ 10.00,   5.00],
#         [  7.00,   3.00],
#         [  6.33,   2.00]]),
#  array([[  6.33,   2.00],
#         [  5.00,   0.00]]),
#  array([[  5.00,   0.00],
#         [  3.67,   2.00]]),
#  array([[  3.67,   2.00],
#         [  3.00,   3.00],
#         [  2.00,   3.67]]),
#  array([[  2.00,   3.67],
#         [  0.00,   5.00]])]


ps = np.array_split(pl_n, p_ft_v[:, 1], axis=0)

# [array([[  0.00,   5.00],
#         [  0.00,  10.00],
#         [  5.00,  10.00]]),
#  array([[ 10.00,  10.00],
#         [ 10.00,   8.00],
#         [  6.33,   8.00]]),
#  array([[  3.67,   8.00]]),
#  array([[  2.00,   8.00],
#         [  2.00,   6.33]]),
#  array([[  2.00,   5.50],
#         [  5.00,   5.50],
#         [  5.00,   4.00],
#         [  2.00,   4.00],
#         [  2.00,   3.67]]),
#  array([[  2.00,   2.00],
#         [  3.67,   2.00]]),
#  array([[  6.33,   2.00]]),
#  array([[ 10.00,   2.00],
#         [ 10.00,   0.00],
#         [  5.00,   0.00]]),
#  array([[  0.00,   0.00],
#         [  0.00,   5.00]]),
#  array([], shape=(0, 2), dtype=float64)]




# def line_sweep(segments):
#     events = []  # List to store events (start and end points of segments)
#     for segment in segments:
#         events.append((segment.start, 'start', segment))
#         events.append((segment.end, 'end', segment))

#     events.sort()  # Sort the events by x-coordinate

#     active_segments = set()  # Set to keep track of active segments

#     intersections = []  # List to store intersection points

#     for event in events:
#         point, event_type, segment = event
#         if event_type == 'start':
#             for active_segment in active_segments:
#                 if intersection(segment, active_segment):
#                     intersections.append((point, active_segment, segment))
#             active_segments.add(segment)
#         else:  # event_type == 'end'
#             active_segments.remove(segment)

#     return intersections
"""


