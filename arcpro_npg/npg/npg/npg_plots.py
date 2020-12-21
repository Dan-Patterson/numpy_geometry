# -*- coding: UTF-8 -*-
# noqa: D205, D400
r"""
---------
npg_plots
---------

Sample scatterplot and line plots, in 2D and 3D.

The Geo class is used for the input arrays.

----

Script :
    npg_plots.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-12-20

Purpose
-------
Sample scatterplot and line plotting, in 2D and 3D.
Alterior motive is to also represent the line and polygon geometries as
... ``graphs``.

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

`random color discussion
<https://stackoverflow.com/questions/14720331/how-to-generate-random-
colors-in-matplotlib>`_.

`Voxels in matplotlib
<https://stackoverflow.com/questions/56752954/how-to-scale-the-voxel-
dimensions-with-matplotlib>`_.

"""
# ---- imports, formats, constants ----

import sys
# from textwrap import dedent
import numpy as np
# import npg_create
# import npg

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# import matplotlib.lines as lines
# from matplotlib.markers import MarkerStyle

np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 7.3f}'.format})
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['plot_2d', 'plot_3d', 'plot_polygons', 'plot_mesh']


# ---- (1) helper functions ----
#
def axis_mins_maxs(pnts):
    """Return axis mins, maxes from data point (``pnts``), values.

    Parameters
    ----------
    pnts : Geo array or ndarray
        If the array is a Geo array or an ndarray with an ``object`` dtype.
        ``pnts`` will be altered if necessary.  In all cases, they are
        returned to the calling function.
    """
    msg = r"""
        A Geo array or ndarray with dtype='O' is required.
        Check your input data.
        """
    if isinstance(pnts, (list, tuple)):
        if isinstance(pnts[0], (list, tuple)):
            if len(pnts[0]) == 2:
                pnts = np.asarray(pnts)
            else:
                print(msg)
                return None
        else:
            pnts = np.asarray(pnts, dtype='O')
    if hasattr(pnts, 'IFT'):  # Geo array
        mn = np.min(pnts.mins(by_bit=True), axis=0)
        mx = np.max(pnts.maxs(by_bit=True), axis=0)
        pnts = pnts.bits  # convert to object array

    elif isinstance(pnts, np.ndarray):
        if pnts.dtype.kind == 'O':
            mn = np.min([i.min(axis=0) for i in pnts], axis=0)
            mx = np.max([i.max(axis=0) for i in pnts], axis=0)
        else:
            mn = pnts.min(axis=0)
            mx = pnts.max(axis=0)
            pnts = [pnts]
    buff = (mx - mn) * 0.05  # 5% space buffer
    x_min, y_min = np.floor(mn - buff)
    x_max, y_max = np.ceil(mx + buff)
    return pnts, x_min, y_min, x_max, y_max


def subplts(plots=1, by_col=True, max_rc=4):
    """Return subplot layout.

    Parameters
    ----------
    plots : integer
        Specify the num(ber) of subplots desired and return the rows and
        columns to produce the subplots.
    by_col : boolean
        True for column oriented, False for row.
    max_rc : integer
        Maximum number of rows or columns depending on by_col.
    """
    row_col = (1, 1)
    if by_col:
        if plots <= max_rc:
            row_col = (1, plots)
        else:
            row_col = (plots - max_rc, max_rc)
    else:
        if plots <= max_rc:
            row_col = (plots, 1)
        else:
            row_col = (max_rc, plots - max_rc)
    return row_col


def scatter_params(plt, fig, ax, title="Title", ax_lbls=None):
    """Assign default parameters for plots.

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
    # plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
    plt.grid(True)
    return


# ---- (2) plot types
#
def plot_mixed(data, title="Title", invert_y=False, ax_lbls=None):
    """Plot mixed data.

    Parameters
    ----------
    data : list of lists
        [[values, type, color, marker, connect]]

    >>> data = [[p3.bits, 2, 'red', '.', True ], [ps, 0, 'black', 'o', False]]
    >>> plot_mixed(data, title="Points in Polygons",
    ...            invert_y=False, ax_lbls=None)
    >>> out, ift, ps, final = pnts_in_Geo(psrt, p3)
    """

    def _label_pnts(pnts, plt):
        """Label the points.

        Note: to skips the last label for polygons, use
        zip(lbl[:-1], pnts[:-1, 0], pnts[:-1, 1])
        """
        lbl = np.arange(len(pnts))
        for label, xpt, ypt in zip(lbl[:], pnts[:, 0], pnts[:, 1]):
            plt.annotate(label, xy=(xpt, ypt), xytext=(2, 2), size=14,
                         textcoords='offset points', ha='left', va='bottom')

    def _line(p, plt):  # , arrow=True):  # , color, marker, linewdth):
        """Connect the points."""
        X, Y = p[:, 0], p[:, 1]
        plt.plot(X, Y, color='black', linestyle='solid', linewidth=2)

    def _scatter(p, plt, color, marker):
        """Do the actual point plotting. Change `s` to increase marker size."""
        X, Y = p[:, 0], p[:, 1]
        plt.scatter(X, Y, s=50, c=color, linewidths=None, marker=marker)

    # ---- main plotting routine
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    scatter_params(plt, fig, ax, title, ax_lbls)
    # ----
    if len(data) == 5:
        data = [data]
    for i, vals in enumerate(data):
        pnts, kind, color, marker, connect = vals
        if kind == 0:
            _scatter(pnts, plt, color='black', marker='s')
            _label_pnts(pnts, plt)
        elif kind == 2:
            cmap = plt.cm.get_cmap('hsv', len(pnts))
            for j, p in enumerate(pnts):
                clr = cmap(j)  # clr=np.random.random(3,)  # clr = "b"
                clr = 'None'
                _line(p, plt)  # color, marker, linewdth=2)
                plt.fill(*zip(*p), facecolor=clr)
                _scatter(p, plt, color, marker)
    plt.show()


def plot_2d(pnts, label_pnts=False, connect=False,
            title='Title', invert_y=False, ax_lbls=None
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

    Required
    --------
    ``scatterparams``, ``axis_mins_maxs``

    Notes
    -----
    If passing a list of arrays to pnts, then make sure you specify a same
    sized list for ``pnt_labels`` and ``connect``.

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
    def _scatter(p, plt, color, marker):
        """Do the actual point plotting. Change `s` to increase marker size."""
        X, Y = p[:, 0], p[:, 1]
        plt.scatter(X, Y, s=50, c=color, linewidths=None, marker=marker)

    def _line(p, plt, color, marker, linewdth):
        """Connect the points with lines."""
        X, Y = p[:, 0], p[:, 1]
        plt.plot(X, Y, color=color, marker=marker, linestyle='solid',
                 linewidth=linewdth)

    def _label_pnts(pnts, plt):
        """Label the points.

        Note: to skips the last label for polygons, use
        zip(lbl[:-1], pnts[:-1, 0], pnts[:-1, 1])
        """
        lbl = np.arange(len(pnts))
        for label, xpt, ypt in zip(lbl[:], pnts[:, 0], pnts[:, 1]):
            plt.annotate(label, xy=(xpt, ypt), xytext=(2, 2), size=14,
                         textcoords='offset points', ha='left', va='bottom')

    # ---- main plotting routine
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H',
               'D', 'd', 'P', 'X')  # MarkerStyle.filled_markers
    # ---- set basic parameters ----
    scatter_params(plt, fig, ax, title, ax_lbls)
    pnts, x_min, y_min, x_max, y_max = axis_mins_maxs(pnts)
    #
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if invert_y:
        plt.ylim(y_max, y_min)
    # ---- enable multiple point files ----
    if isinstance(pnts, (list, tuple, np.ndarray)):
        colors = ['black', 'blue', 'green', 'red',
                  'darkgrey', 'darkblue', 'darkred', 'darkgreen', 'grey'] * 2
        for i, p in enumerate(pnts):
            marker = markers[i]  # see markers = MarkerStyle.filled_markers
            color = colors[i]
            _scatter(p, plt, color, marker)
            if connect:
                _line(p, plt, color, marker, linewdth=2)
            if label_pnts:
                if (p[0][:, None] == p[-1]).all(-1).any(-1):
                    p = p[:-1]
                _label_pnts(p, plt)
    plt.show()


def plot_3d(a):
    """Plot an xyz sequence in 3d.

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
    """Plot Geo array poly boundaries.

    Parameters
    ----------
    arr : ndarray or Geo array
        If the arrays is a Geo array, it will convert it to `arr.bits`.
    outline : boolean
        True, returns the outline of the polygon.  False, fills the polygon

    References
    ----------
    See module docs for general references.

    Example
    -------
    use hexs to demonstrate::

        h = npg.npg_create.hex_flat(dx=1, dy=1, x_cols=5, y_rows=3,
                                    orig_x=0, orig_y=0, kind=2, asGeo=True)
    """
    def _line(p, plt):  # , arrow=True):  # , color, marker, linewdth):
        """Connect the points."""
        X, Y = p[:, 0], p[:, 1]
        plt.plot(X, Y, color='black', linestyle='solid', linewidth=2)
    # ----
    if hasattr(arr, 'IFT'):
        cw = arr.CW
        shapes = arr.bits
    elif isinstance(arr, np.ndarray):
        if len(arr.shape) == 2:
            shapes = [arr]
        else:
            shapes = arr
    elif isinstance(arr, (list, tuple)):
        shapes = arr
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight = 8
    fig.set_figwidth = 8
    fig.dpi = 200
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
    ax.set_aspect('equal', adjustable='box')
    # cmap = plt.cm.get_cmap(plt.cm.viridis, 143)  # default colormap
    cmap = plt.cm.get_cmap('jet', len(shapes))  # 'hsv'
    for i, shape in enumerate(shapes):
        if outline:   # _line(shape, plt)  # alternate, see line for options
            plt.fill(*zip(*shape), facecolor='none', edgecolor='black',
                     linewidth=3)
        else:
            if cw[i] == 0:
                clr = "w"
            else:
                clr = cmap(i)  # clr=np.random.random(3,)  # clr = "b"
            plt.fill(*zip(*shape), facecolor=clr)
    # ----
    plt.show()
    return plt


def plot_mesh(x=None, y=None):
    """Plot a meshgrid/fishnet given x and y ranges.

    Parameters
    ----------
    x, y : arrays
        Arrays of sequential values representing the x and y ranges
    Requires
    --------
    If not initially imported, add this to the script or function.
    >>> from matplotlib.collections import LineCollection

    https://stackoverflow.com/questions/47295473/how-to-plot-using-
    matplotlib-python-colahs-deformed-grid
    """
    if x is None or y is None:
        x, y = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 0.6, 7))
    segs1 = np.stack((x[:, [0, -1]], y[:, [0, -1]]), axis=2)
    segs2 = np.stack((x[[0, -1], :].T, y[[0, -1], :].T), axis=2)
    fig, ax = plt.subplots(1, 1)
    # ---- use scatter_params to set defaults
    scatter_params(plt, fig, ax, title="Mesh", ax_lbls=['X', 'Y'])
    ax.add_collection(LineCollection(np.concatenate((segs1, segs2))))
    ax.autoscale(enable=True, axis='both', tight=None)
    ax.set_aspect('equal', adjustable='box')
    plt.show()


# def demo():
#     x1 = [-1,-1,10,10,-1]; y1 = [-1,10,10,-1,-1]
#     x2 = [21,21,29,29,21]; y2 = [21,29,29,21,21]
#     shapes = [[x1,y1],[x2,y2]]
#     for shape in shapes:
#       x,y = shape
#       plt.plot(x,y)
#     plt.show()
"""
see this for holes

https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html


Also

for shape in hexs[:2]:
    x, y = shape[:, 0], shape[:, 1]
    plt.plot(x, y)
plt.show()
"""


# ----------------------------------------------------------------------------
# ---- (3) testing section ----

def plot_mst(a, pairs):
    """Plot minimum spanning tree test."""
    fig, ax = plt.subplots(1, 1)
    scatter_params(plt, fig, ax, title="Title", ax_lbls=None)
    plt.scatter(a[:, 0], a[:, 1])
    for pair in pairs:
        i, j = pair
        plt.plot([a[i, 0], a[j, 0]], [a[i, 1], a[j, 1]], c='r')
    lbl = np.arange(len(a))
    for label, xpt, ypt in zip(lbl[:], a[:, 0], a[:, 1]):
        plt.annotate(label, xy=(xpt, ypt), xytext=(2, 2), size=8,
                     textcoords='offset points',
                     ha='left', va='bottom')
    plt.show()


def _demo():
    """Plot 20 points which have a minimum 1 unit point spacing."""
    a = np.array([[0.4, 0.5], [1.2, 9.1], [1.2, 3.6], [1.9, 4.6],
                  [2.9, 5.9], [4.2, 5.5], [4.3, 3.0], [5.1, 8.2],
                  [5.3, 9.5], [5.5, 5.7], [6.1, 4.0], [6.5, 6.8],
                  [7.1, 7.6], [7.3, 2.0], [7.4, 1.0], [7.7, 9.6],
                  [8.5, 6.5], [9.0, 4.7], [9.6, 1.6], [9.7, 9.6]])
    z = np.array([[0.4,  0.5], [1.2,  3.6], [1.9,  4.6], [2.9,  5.9],
                  [1.2,  9.1], [5.3,  9.5], [7.7,  9.6], [9.7,  9.6],
                  [9.0,  4.7], [9.6,  1.6], [7.4,  1.0], [4.3,  3.0],
                  [0.4,  0.5]])
    # z = npg.concave(a, 3, True)
    plot_2d([z, a], [False, True], [True, False],
            title='Points and concave hull')
    # plot_2d([a], label_pnts=False, connect=False,
    #         title='Points no closer than... test',
    #         invert_y=False, ax_lbls=None
    #         )
    return


# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    a, plt, ax = _demo()
