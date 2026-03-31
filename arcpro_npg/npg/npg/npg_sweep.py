# -*- coding: utf-8 -*-
# noqa: D205, D208, D400, F403
r"""
------------
npg_sweep
------------

Modified :
    2026-02-26

** Boolean operations on poly geometry.


----

Script :
    npg_sweep.py

Author :
    `<https://github.com/Dan-Patterson>`_.

"""
# pylint: disable=C0103,C0201,C0209,C0302,C0415
# pylint: disable=R0902,R0904,R0912,R0913,R0914,R0915
# pylint: disable=W0105,W0201,W0212,W0221,W0611,W0612,W0613,W0621
# pylint: disable=E0401,E0611,E1101,E1121

import sys
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as uts  # noqa

fmt_ = {"bool": lambda x: repr(x.astype(np.int32)),
      "float_kind": '{: 0.3f}'.format}
np.set_printoptions(precision=3, threshold=100, edgeitems=10, linewidth=80,
                    suppress=True,
                    formatter=fmt_,
                    floatmode='maxprec_equal',
                    legacy='1.25')  # legacy=False or legacy='1.25'
np.ma.masked_print_option.set_display('-')  # change to a single -

import npg  # noqa
# from npg import npGeo
# from npg.npg_pip import np_wn
from npg.npg_bool_ops import sweep_srt
from npg.npg_bool_hlp import _del_seq_dupl_pnts_
from npg.npg_geom_hlp import _orient_clockwise_  # , _bit_area_
from npg.npg_prn import prn_, prn_as_obj  # noqa


script = sys.argv[0]

__all__ = [
    'sweep'
]   # 'dissolve'

# __helpers__ = []

__imports__ = [
    'npGeo',                  # main import from npg
    '_del_seq_dupl_pnts_',    # npg_bool_hlp
    '_orient_clockwise_',     # npg.npg_geom_hlp
    'sweep_srt'               # npg_bool_ops
    'prn_',                   # npg_prn
    'prn_as_obj'
]

# ---- ---------------------------
# ---- (1) general helpers
#

# ---- `sweep` works sort of for E, d1_
def sweep(seq_srted,
          _CP_,
          _in_,
          _out_,
          _on_,
          descending_y=True,
          increasing_angle=True
          ):
    """Return segments sorted lexicographically by start point and segment.

    Parameters
    ----------
    seq_srted : list of array segments from `polygon_overlay`.
    _CP_ : array
        The coordinates of the intersection points.
    _in_, _out_, _on_ : arrays
        The id values of the points making up the overlay

    Requires
    --------
    `sweep_srt`
    """

    def es_chk(s_, e_, st_en_srt, arrs_flat):
        """Check start-end within closure."""
        s_e = np.array([s_, e_])
        w0_ = np.nonzero((st_en_srt == s_e).all(-1))[0]
        if len(w0_) == 0:
            w1_ = np.nonzero((st_en_srt == s_e[::-1]).all(-1))[0]
            if len(w1_) == 0:
                return None
            return arrs_flat[w1_[0]]
        else:
            return arrs_flat[w0_[0]]

    def ee_chk(f, s, st_en_srt, arrs_flat):
        """Return segment end_end closure result."""
        ee = np.array([f[-1], s[-1]])
        w0_ = np.nonzero((st_en_srt == ee).all(-1))[0]
        if len(w0_) > 0:
            fst = w0_.tolist()[0]
            return arrs_flat[fst]
        return None

    def closer(v, arrs_flat, _out_, cnt):
        """Try closing the array."""
        q = []
        v1 = v[::]  # copy v
        for i in arrs_flat[cnt + 2:]:
            chk = len(o_set.intersection(i)) == 0
            # -- 2025-12-16  sneak in a closure test
            seg_ = es_chk(v1[0], v1[-1], st_en_srt, arrs_flat)
            if seg_ is not None:
                if seg_[-1] == v1[-1]:
                    v1 = np.concatenate((v1, seg_[::-1]))
                else:
                    v1 = np.concatenate((v1, seg_))
            # --
            if abs(v1[-1] - v1[0]) <= 1:
                q.append("i, v1 {} {}".format(i, v1))
                break  # -- return v1, q  # didn't work
            # print(f"cnt : {cnt} chk {chk} i {i}")
            if chk:
                if i[0] == v1[-1]:  # start meets end
                    v1 = np.concatenate((v1, i))
                    q.append("i, v1 {} {}".format(i, v1))
                    continue
                if i[-1] == v1[-1]:  # ends meet end
                    v1 = np.concatenate((v1, i[::-1]))
                    q.append("i, v1 {} {}".format(i, v1))
                    continue
                # if i[-1] == v1[0]:  # end meets start
                #     v1 = np.concatenate((i, v1))
                q.append("i, v1 {} {}".format(i, v1))
        return v1, q

    def pair_(ar):
        """Form pairs from a nested list of lists."""
        out_ = []
        for i in ar:
            if len(i) > 1:
                v = list(zip(i[:-1], i[1:]))
                for j in v:
                    out_.append(j)
            else:
                out_.append(i)
        return out_

    # -- perform the sweep sort returning segments arranged in lexicographic
    #    order as well as duplicate segments found during processing
    # -- un-assign descending_y and increasing_angle 
    args = sweep_srt(seq_srted, _CP_, descending_y=True, increasing_angle=True)
    #
    arrs_, arrs_flat, sw_srt, dups, st_en_srt, seq_rev = args
    #
    # -- _on_ sorted lexicographically
    c_on_xy = _CP_[_on_]  # duplicate start/end cropped from c_on
    c_on_xy_lex = np.lexsort((-c_on_xy[:, 1], c_on_xy[:, 0]))  # sorted
    c_on_lex = _on_[c_on_xy_lex]  # used to process the sweepline id values
    #
    # -- sweep line ids as a `list of lists` --  keep
    # -- by x
    x_lex = c_on_xy[c_on_xy_lex][:, 0]
    x_dif = (x_lex[1:] - x_lex[:-1])
    spl_whr2 = np.nonzero(x_dif > 0.0)[0] + 1
    swp_ln_ids = np.array_split(c_on_lex, spl_whr2)  # -- sorted by x
    # -- by y
    # c_on_xy_lex_2 = np.lexsort((c_on_xy[:, 0], -c_on_xy[:, 1]))  # sorted
    # c_on_lex_2 = _on_[c_on_xy_lex_2]
    # #
    # y_lex = c_on_xy[c_on_xy_lex_2][:, 1]
    # y_dif = (y_lex[1:] - y_lex[:-1])
    # spl_whr2_2 = np.nonzero(y_dif < 0.0)[0] + 1  # -- sign changed from above
    # swp_ln_ids_2 = np.array_split(c_on_lex_2, spl_whr2_2)
    # !!!! work from here on
    """
    [len(o_set.intersection(i)) for i in c_seq]
    [len(i_set.intersection(i)) for i in c_seq]
    """
    # ---- construct pairs, version 1
    # swp_ln_ids gives the ids list of points on a vertical line
    # swp_ln_ids_2 gives the ids for a horizonal line
    #
    # ---- construct pairs, version 2
    frmt_ = "cnt {} f {}, s {} \n    closed {}\n    keep_ {}"  # formatter
    #
    o_set = set(_out_)
    i_set = set(_in_)
    cur_id = c_on_lex[0]  # first sweepline
    keep_ = []
    closed_ = []
    # clp_ = []
    #
    cur_cnt = 0
    cur_id = c_on_lex[cur_cnt]
    # nxt_id = c_on_lex[cur_cnt + 1]
    #
    # _sweep_id_ = c_on_lex.tolist()  # used as a counter for the sweep line ids
    #
    new_count = 0
    for cnt, ar in enumerate(arrs_flat[:-1]):  # start at XX tomorrow
        v = None
        f, s = arrs_flat[cnt:cnt+2]
        # -- classify segments
        f_out = len(o_set.intersection(f)) > 0  # check first (f)
        f_in = len(i_set.intersection(f)) > 0
        s_out = len(o_set.intersection(s)) > 0
        s_in = len(i_set.intersection(s)) > 0
        #
        # -- transition check
        paired = cur_id in f and cur_id in s  # -- upper
        #
        if paired:  # -- eg.  where cur_id is in f and s at any location
            #
            if f[0] == s[0]:  # first ids are the same eg. [0, 21, 7],  [0, 1]
                if f_out:  # frst segment is definitely out, secn is in or on
                    v = np.concatenate((s[::-1], f))  # form outer polygon
                elif abs(f[-1] - s[-1]) == 1:  # check last of both for closure
                    v = np.concatenate((f, s[::-1]))
                else:
                    # -- 2025-12-16 end, end check to close if segment
                    #  gaps can appear as in  E, d1_ 19, 18 bit 16, 29, 19
                    chk_ends = ee_chk(f, s, st_en_srt, arrs_flat)
                    if chk_ends is not None:  # -- bit is 18, 17, 16
                        v = np.concatenate((f, chk_ends, s[::-1]))
                    else:
                        v = np.concatenate((s[::-1], f))
            elif f[0] == s[-1]:  # start of frst matches last of second
                v = np.concatenate((s, f))
            #
            # -- closure check, special cases
            diff_ = v[0] - v[-1]
            if diff_ == 0:
                closed_.append(v)
                # -- keep_ check for clipper [2, 4] [2, 3, 4] where 3 is in
                if s_in and (len(f) == 2):
                    if (f == keep_[0][-2:]).all():
                        v_ = np.concatenate((keep_[0][:-2], s))
                        keep_[0] = v_
            elif diff_ == -1:
                closed_.append(v)
            elif abs(v[0] - v[-1]) <= 2:
                if v[0] < v[-1]:
                    v = np.array(v.tolist() + [v[-1] - 1, v[0]])
                else:
                    v = np.array([v[0], v[0] + 1] + v.tolist())
                if abs(s[0] - s[-1]) > 1:  # [14, 25, 13], [14, 15] in E, d1_
                    closed_.append(v)
            else:  # try closing, may require multple segments.  v_, new array
                # run `closer` for multi segment joins
                v_, rpt = closer(v, arrs_flat, _out_, cnt)  # rpt is a report
                if v_[0] == v_[-1]:  # -- maybe combine with below
                    closed_.append(v_)  # append the closed version
                elif abs(v_[-1] - v_[0]) <= 1:
                    v_ = np.concatenate((v_, v_[[-1,0]]))  # -- 2025-12-16
                    closed_.append(v_)
                else:
                    keep_.append(v)  # otherwise append the original
        #
        # -- transition condition, new sweepline id encountered
        elif cur_id not in s:  # -- first and second sweepline ids differ
            # prev_id = cur_id
            cur_cnt += 1
            cur_id = c_on_lex[cur_cnt]
            # -- check all entries in keep_, use the first found if any
            end_chk = np.nonzero([s[-1] == i[-1] for i in keep_])[0]
            st_chk = np.nonzero([s[0] == i[0] for i in keep_])[0]
            # -- 2025-12-01
            st_en_chk = np.nonzero([s[0] == i[-1] for i in keep_])[0]
            # st_st_chk = 
            #
            if len(end_chk) > 0:  # fails under circumstances in upper clip
                # 3 sequential point check
                _chk_ = sorted(list(set(f.tolist() + s.tolist())))
                if len(_chk_) >= 3 and (_chk_[-1] - _chk_[0] > 2):
                    print("\ncnt {} end_chk :f {}, s {}".format(cnt, f, s))
                    v = keep_[end_chk[0]].tolist() + s.tolist()[::-1]
                    keep_[end_chk[0]] = np.array(v)
            if len(st_chk) > 0:
                # print("\ncnt {} st_chk :f {}, s {}".format(cnt, f, s))
                k_id = st_chk[0]
                v = s.tolist()[::-1] + keep_[k_id].tolist()
                # -- 2025-11-28 to check for closure
                # just append 1st
                if abs(v[0] - v[-1]) <= 2:
                    v = np.array(v + [v[0]])
                    closed_.append(v)
                    keep_.pop(k_id)  # -- now remove from keep_
                else:
                    keep_[k_id] = np.array(v)
            #
            # -- 2025-12-01 added st_end_chk ** clean this up of duplicates
            s_diff = s[-1] - s[0] 
            if len(st_en_chk) > 0 and not s_out:
                if s_diff != 1:
                    # print("\ncnt {} st_en_chk :f {}, s {}".format(cnt, f, s))
                    k_id = st_en_chk[0]
                    v = keep_[k_id].tolist() + s.tolist()
                    if abs(v[0] - v[-1]) <= 2:
                        v = np.array(v + [v[0]])
                        closed_.append(v)
                        keep_.pop(k_id)  # -- now remove from keep_
                    else:
                        keep_[k_id] = np.array(v)
                elif s_diff == 1:  # added 2025-17 for E, d1_ to add 9, 10
                    k_id = st_en_chk[0]
                    v = keep_[k_id].tolist() + s.tolist()
                    if abs(v[0] - v[-1]) <= 2:
                        v = np.array(v + [v[0]])
                        closed_.append(v)
                        keep_.pop(k_id)
                    else:
                        keep_[k_id] = np.array(v)
        #
        # -- 2025-11-29 to try and add segs to upper clipper
        #  uses s_in, for transition
        s_diff = s[-1] - s[0]  # -- 2025-12-03 for single crossover 3,4 on
        if s_in or f_out and s_diff != 1:
            st_en_chk = np.nonzero([s[0] == i[-1] for i in keep_])[0]
            en_st_chk = np.nonzero([s[-1] == i[0] for i in keep_])[0]
            if len(st_en_chk) > 0:
                k_id = st_en_chk[0]
                v = keep_[k_id].tolist() + s.tolist()
                if v[0] - v[-1] == 0:
                    v = np.array(v)
                    closed_.append(v)
                    keep_.pop(k_id) 
                elif abs(v[0] - v[-1]) <= 2:
                    v = np.array(v + [v[0]])
                    closed_.append(v)
                    keep_.pop(k_id) 
                else:
                    keep_[k_id] = np.array(v)
        #
        n_ = len(closed_)
        if n_ > new_count:
            cl = closed_[-1]
            new_count += 1
        else:
            cl = ""
        print(frmt_.format(cnt, f, s, cl, keep_))  # -- frmt_ is above
    #
    # geom = []
    geom = [_CP_[i] for i in closed_]
    geom = [_del_seq_dupl_pnts_(i) for i in geom]
    geom = _orient_clockwise_(geom)
    return arrs_, geom

# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print(f"\nRunning... {script}\n")
