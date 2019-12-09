# -*- coding: UTF-8 -*-
"""
===========
npg_plots
===========

Script : npg_plots.py

Author : Dan_Patterson@carleton.ca

Modified : 2019-11-09

Purpose:
--------
    Sample scatterplot and line plotting, in 2D and 3D

Notes
-----
    >>> print(plt.style.available)
    >>> import matplotlib.pyplot.figure as fig
    # figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None,
    #        frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>,
    #        clear=False, **kwargs)
    # matplotlib.pyplot.subplots
    # subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
    #          subplot_kw=None, gridspec_kw=None, **fig_kw)

References
----------
`<https://matplotlib.org/3.1.0/gallery/color/named_colors.html#>`_.

`<https://matplotlib.org/users/customizing.html>`_.

`<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html>`_.

`<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html>`_.

"""
# ---- imports, formats, constants ----

import sys
import numpy as np
import npgeom as npg
import npg_create

# import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.lines as lines
from matplotlib.markers import MarkerStyle

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['plot_2d', 'plot_3d']


# ---- functions ----
#
def plot_2d(pnts, label_pnts=False, connect=False,
            title='Title', invert_y=False, ax_lbls=None,
            ):
    """Plot points for Nx2 array representing x,y or row,col data.

    >>> plot_2d([p, c, out], [True, True, False], [True, True, False])
    >>> plot_2d(p, True, True)

    Parameters
    ----------
    see _params() to specify special parameters

    pnts : array-like or list of arrays
        2D array of point-like objects ie a row/column array
    invert_y : boolean
        If True, the y-axis is inverted to represent row-column formatting
        rather than x,y formatting.
    ax_lbls : list
        A list, like ['X', 'Y'] is needed, if left to `None` then that will
        be the default.
    pnt_labels : boolean or list of booleans
        True for point labels.
    connect : boolean or list of booleans
        True to connect points in sequential order.

    Notes
    -----
    If passing a list of arrays to pnts, then make sure you specify a same
    sized list to pnt_labels and connect

    markers::

     '.' - point   'o' - circle     '+' - filled
     '^', 'v', '<', '>' - triangle, up, down, left, right
     's' - square  'X' - x filled   'D' - diamond

    lines::

     '-' solid, '--' dashed, '-.' dashdot, ':' dotted or None

    Returns
    -------
    A scatterplot representing the data.  It is easier to modify the
    script below than to provide a whole load of input parameters.

    References
    ----------
    `<https://matplotlib.org/>`_.
    """
    def scatter_params(plt, fig, ax, title="Title", ax_lbls=None):
        """Default parameters for plots

        Notes
        -----
        ticklabel_format(useoffset), turns off scientific formatting
        """
        fig.set_figheight = 8
        fig.set_figwidth = 8
        fig.dpi = 200
        if ax_lbls is None:
            ax_lbls = ['X', 'Y']
        x_label, y_label = ax_lbls
        font1 = {'family': 'sans-serif', 'color': 'black',
                 'weight': 'bold', 'size': 14}  # set size to other values
        ax.set_aspect('equal', adjustable='box')
        ax.ticklabel_format(style='sci', axis='both', useOffset=False)
        ax.xaxis.label_position = 'bottom'
        ax.yaxis.label_position = 'left'
        plt.xlabel(x_label, fontdict=font1, labelpad=12, size=14)
        plt.ylabel(y_label, fontdict=font1, labelpad=12, size=14)
        plt.title(title + "\n", loc='center', fontdict=font1, size=20)
        plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
        plt.grid(True)
        return

    def _scatter(p, plt, color, marker):
        """Do the actual point plotting."""
        X, Y = p[:, 0], p[:, 1]
        plt.scatter(X, Y, s=10, c=color, linewidths=None, marker=marker)

    def _line(p, plt, color, marker, linewdth):
        """Connect the"""
        X, Y = p[:, 0], p[:, 1]
        plt.plot(X, Y, color=color, marker=marker, linestyle='solid',
                 linewidth=linewdth)

    def _label_pnts(pnts, plt):
        """as it says"""
        lbl = np.arange(len(pnts))
        for label, xpt, ypt in zip(lbl, pnts[:, 0], pnts[:, 1]):
            plt.annotate(label, xy=(xpt, ypt), xytext=(2, 2), size=12,
                         textcoords='offset points', ha='left', va='bottom')

    # ---- main plotting routine
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    markers = MarkerStyle.filled_markers  # set default marker style
    # ---- set basic parameters ----
    scatter_params(plt, fig, ax, title, ax_lbls)
    if isinstance(pnts, (list, tuple)):
        mn = np.min([i.min(axis=0) for i in pnts], axis=0)
        mx = np.max([i.max(axis=0) for i in pnts], axis=0)
        buff = (mx - mn) * 0.05  # 5% space buffer
        x_min, y_min = np.floor(mn - buff)
        x_max, y_max = np.ceil(mx + buff)
    else:
        mn = pnts.min(axis=0)
        mx = pnts.max(axis=0)
        buff = (mx - mn) * 0.05  # 5% space buffer
        x_min, y_min = np.floor(mn - buff)
        x_max, y_max = np.ceil(mx + buff)
    #
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if invert_y:
        plt.ylim(y_max, y_min)
    # ---- enable multiple point files ----
    if isinstance(pnts, np.ndarray):
        pnts = [pnts]
    if isinstance(pnts, (list, tuple)):
        colors = ['black', 'blue', 'green', 'red',
                  'darkgrey', 'darkblue', 'darkred', 'darkgreen', 'grey']
        for i, p in enumerate(pnts):
            marker = markers[i]  # see markers = MarkerStyle.filled_markers
            color = colors[i]
            _scatter(p, plt, color, marker)
            if connect:
                _line(p, plt, color, marker, linewdth=2)
            if label_pnts:
                _label_pnts(p, plt)
#    plt.ion()
    plt.show()


def plot_3d(a):
    """Plot an xyz sequence in 3d

    Parameters
    ----------
    a : array-like
        A 3D array of point objects representing X,Y and Z values

    References
    ----------
    `<https://matplotlib.org/tutorials/toolkits/mplot3d.html#sphx-glr-
    tutorials-toolkits-mplot3d-py>`_.

    Example
    -------
    >>> x = np.arange(10)
    >>> y = np.arange(10)
    >>> z = np.array([5,4,3,2,1,1,2,3,4,5])
    >>> xyz = np.c_[(x,y,z)]
    >>> plot_3d(xyz)
    """
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    #
    mpl.rcParams['legend.fontsize'] = 10
    #
    fig = plt.figure()
    fig.set_figheight = 8
    fig.set_figwidth = 8
    fig.dpi = 200
    ax = Axes3D(fig)  # old  #ax = fig.gca(projection='3d')
    #
    x = a[:, 0]
    y = a[:, 1]
    z = a[:, 2]
    ax.plot(x, y, z, label='xyz')
    ax.legend()
    # plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
    plt.show()


def plot_polygons(arr, outline=True):
    """Plot poly boundaries.
    When using the Geo class, pass ``arr.bits`` to it
    """

    def _line(p, plt):  # , color, marker, linewdth):
        """Connect the"""
        X, Y = p[:, 0], p[:, 1]
        plt.plot(X, Y, color='black', linestyle='solid', linewidth=2)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    for shape in arr:
        if outline:
            _line(shape, plt)  # see line for options
        else:
            plt.fill(*zip(*shape))
    plt.show()

# use hexs to demonstrate
hexs = npg_create.hex_flat(dx=1, dy=1, cols=4, rows=3)
# def demo():
#    x1 = [-1,-1,10,10,-1]; y1 = [-1,10,10,-1,-1]
#    x2 = [21,21,29,29,21]; y2 = [21,29,29,21,21]
#    shapes = [[x1,y1],[x2,y2]]
#    for shape in shapes:
#      x,y = shape
#      plt.plot(x,y)
#    plt.show()
"""
see this for holes

https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html


Also

for shape in hexs[:2]:
    x, y = shape[:, 0], shape[:, 1]
    plt.plot(x, y)
plt.show()
"""

# def read_flds(in_fc, x_fld, y_fld):
#    """
#    """
#    flds = [x_fld, y_fld]
#    a = arcpy.da.TableToNumPyArray(in_fc, flds)
#    x_name, y_name = a.dtype.names
#    a = a.view(np.float64).reshape(a.shape[0], 2)
#    return a, x_name, y_name
#
#
#
# def subplts(plots=1, by_col=True, max_rc=4):
#    """specify the num(ber) of subplots desired and return the rows
#    :  and columns to produce the subplots.
#    :  by_col - True for column oriented, False for row
#    :  max_rc - maximum number of rows or columns depending on by_col
#    """
#    row_col = (1, 1)
#    if by_col:
#        if plots <= max_rc:
#            row_col = (1, plots)
#        else:
#            row_col = (plots - max_rc, max_rc)
#    else:
#        if plots <= max_rc:
#            row_col = (plots, 1)
#        else:
#            row_col = (max_rc, plots - max_rc)
#    return row_col
#
#
#def generate():
#    plt.show()
#    #plt.close()
# ----------------------------------------------------------------------------
# ---- running script or testing code section ----
def _demo():
    """Plot 20 points which have a minimum 1 unit point spacing
    :
    """
    a = np.array([[0.4, 0.5], [1.2, 9.1], [1.2, 3.6], [1.9, 4.6],
                  [2.9, 5.9], [4.2, 5.5], [4.3, 3.0], [5.1, 8.2],
                  [5.3, 9.5], [5.5, 5.7], [6.1, 4.0], [6.5, 6.8],
                  [7.1, 7.6], [7.3, 2.0], [7.4, 1.0], [7.7, 9.6],
                  [8.5, 6.5], [9.0, 4.7], [9.6, 1.6], [9.7, 9.6]])
    plot_2d(a, title='Points no closer than... test',
            r_c=False, ax_lbls=None, pnt_labels=True)
    return a


# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    a, plt, ax = _demo()
