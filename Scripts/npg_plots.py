# -*- coding: UTF-8 -*-
"""
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
    2019-12-19

Purpose
-------
Sample scatterplot and line plotting, in 2D and 3D.
alterior motive is to also represent the line and polygon geometries as
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

"""
# ---- imports, formats, constants ----

import sys
import numpy as np
# import npg_create
# import npgeom as npg

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# import matplotlib.lines as lines
from matplotlib.markers import MarkerStyle

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['plot_2d', 'plot_3d', 'plot_polygons', 'plot_mesh']


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
        plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
        plt.grid(True)
        return

    def _scatter(p, plt, color, marker):
        """Do the actual point plotting."""
        X, Y = p[:, 0], p[:, 1]
        plt.scatter(X, Y, s=10, c=color, linewidths=None, marker=marker)

    def _line(p, plt, color, marker, linewdth):
        """Connect the points with lines."""
        X, Y = p[:, 0], p[:, 1]
        plt.plot(X, Y, color=color, marker=marker, linestyle='solid',
                 linewidth=linewdth)

    def _label_pnts(pnts, plt):
        """Label the points."""
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
    `random color generation in matplotlib
    <https://stackoverflow.com/questions/14720331/how-to-generate-random-
    colors-in-matplotlib>`_.

    See module docs for general references.
    """
    def _line(p, plt):  # , arrow=True):  # , color, marker, linewdth):
        """Connect the points."""
        X, Y = p[:, 0], p[:, 1]
        plt.plot(X, Y, color='black', linestyle='solid', linewidth=2)
    # ----
    if hasattr(arr, 'IFT'):
        cw = arr.CW
        shapes = arr.bits
    else:
        shapes = np.copy(arr)
    fig, ax = plt.subplots(1, 1)
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
    ax.set_aspect('equal', adjustable='box')
    # cmap = plt.cm.get_cmap(plt.cm.viridis, 143)  # default colormap
    cmap = plt.cm.get_cmap('hsv', len(shapes))
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


# def preamble(fo):
#     """The SVG preamble and styles.

#     https://scipython.com/blog/depicting-a-torus-as-an-svg-image/
#     """
#     print('<?xml version="1.0" encoding="utf-8"?>\n'
#           '<svg xmlns="http://www.w3.org/2000/svg"\n' + ' '*5 +
#           'xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}" >'.format(width, height), file=fo)

#     print("""
#         <defs>
#         <style type="text/css"><![CDATA[""", file=fo)
#     print('path {stroke-width: 1.5px; stroke: #000;}', file=fo)
#     print("""]]></style>
#     </defs>""", file=fo)
def plot_mesh(x=None, y=None, ax=None):
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
    # fig, ax = plt.subplots(1, 1)
    ax = ax or plt.gca()
    # ax.scatter(x, y)
    ax.add_collection(LineCollection(np.concatenate((segs1, segs2))))
    ax.autoscale(enable=True, axis='both', tight=None)
    # ax.set_aspect('equal', adjustable='box')
    plt.show()


# use hexs to demonstrate
# hexs = npg_create.hex_flat(dx=1, dy=1, cols=4, rows=3)
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
# def generate():
#    plt.show()
#    #plt.close()
# ----------------------------------------------------------------------------
# ---- running script or testing code section ----


def _demo():
    """Plot 20 points which have a minimum 1 unit point spacing."""
    a = np.array([[0.4, 0.5], [1.2, 9.1], [1.2, 3.6], [1.9, 4.6],
                  [2.9, 5.9], [4.2, 5.5], [4.3, 3.0], [5.1, 8.2],
                  [5.3, 9.5], [5.5, 5.7], [6.1, 4.0], [6.5, 6.8],
                  [7.1, 7.6], [7.3, 2.0], [7.4, 1.0], [7.7, 9.6],
                  [8.5, 6.5], [9.0, 4.7], [9.6, 1.6], [9.7, 9.6]])
    plot_2d(a, title='Points no closer than... test',
            ax_lbls=None, pnt_labels=True)
    return a


# ----------------------------------------------------------------------
def svg(self, scale_factor=1, fill_color=None):
    if self.is_empty:
        return '<g />'
    if fill_color is None:
        fill_color = "#66cc99" if self.is_valid else "#ff3333"
    rings = []
    s = ""
    for ring in self['rings']:
        rings = ring
        exterior_coords = [
            ["{},{}".format(*c) for c in rings]]
        path = " ".join([
            "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
            for coords in exterior_coords])
        s += (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
        ).format(2. * scale_factor, path, fill_color)
    return s


'''
_repr_svg_

# r"C:/arc_pro/bin/Python/envs/arcgispro-py3/Lib/site-packages/arcgis/
geometry/_types.py"

line 391


def _repr_svg_(self):
    """SVG representation for iPython notebook"""
    svg_top = '<svg xmlns="http://www.w3.org/2000/svg" ' \
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
    if self.is_empty:
        return svg_top + '/>'
    else:
        # Establish SVG canvas that will fit all the data + small space
        xmin, ymin, xmax, ymax = self.extent
        if xmin == xmax and ymin == ymax:
            # This is a point; buffer using an arbitrary size
            xmin, ymin, xmax, ymax = self.buffer(1).extent
        else:
            # Expand bounds by a fraction of the data ranges
            expand = 0.04  # or 4%, same as R plots
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand
            xmin -= expand_amount
            ymin -= expand_amount
            xmax += expand_amount
            ymax += expand_amount
        dx = xmax - xmin
        dy = ymax - ymin
        width = min([max([100., dx]), 300])
        height = min([max([100., dy]), 300])
        try:
            scale_factor = max([dx, dy]) / max([width, height])
        except ZeroDivisionError:
            scale_factor = 1.
        view_box = "{0} {1} {2} {3}".format(xmin, ymin, dx, dy)
        transform = "matrix(1,0,0,-1,0,{0})".format(ymax + ymin)
        return svg_top + (
            'width="{1}" height="{2}" viewBox="{0}" '
            'preserveAspectRatio="xMinYMin meet">'
            '<g transform="{3}">{4}</g></svg>').format(view_box, width,
                                                       height, transform,
                                                       self.svg(scale_factor)
                                                       )



multipoint svg

#----------------------------------------------------------------------
def svg(self, scale_factor=1., fill_color=None):
    """Returns a group of SVG circle elements for the MultiPoint geometry.

    Parameters
    ==========
    scale_factor : float
        Multiplication factor for the SVG circle diameters.  Default is 1.
    fill_color : str, optional
        Hex string for fill color. Default is to use "#66cc99" if
        geometry is valid, and "#ff3333" if invalid.
    """
    if self.is_empty:
        return '<g />'
    if fill_color is None:
        fill_color = "#66cc99" if self.is_valid else "#ff3333"
    return '<g>' + \
           ''.join(('<circle cx="{0.x}" cy="{0.y}" r="{1}" '
                    'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="0.6" />'
                    ).format(Point({'x': p[0], 'y': p[1]}), 3 * scale_factor, 1 * scale_factor, fill_color) \
                   for p in self['points']) + \
           '</g>'
               
point svg
    def svg(self, scale_factor=1, fill_color=None):
        """Returns SVG circle element for the Point geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameter.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return (
            '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
            'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="0.6" />'
            ).format(self, 3 * scale_factor, 1 * scale_factor, fill_color)

polygon svg
    #----------------------------------------------------------------------
    def svg(self, scale_factor=1,fill_color=None):
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        rings = []
        s = ""
        for ring in self['rings']:
            rings = ring
            exterior_coords = [
                ["{},{}".format(*c) for c in rings]]
            path = " ".join([
                "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
                for coords in exterior_coords])
            s += (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
            ).format(2. * scale_factor, path, fill_color)
        return s


polyline svg
    def svg(self, scale_factor=1, stroke_color=None):
        """Returns SVG polyline element for the LineString geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        stroke_color : str, optional
            Hex string for stroke color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if stroke_color is None:
            stroke_color = "#66cc99" if self.is_valid else "#ff3333"
        paths = []
        for path in self['paths']:
            pnt_format = " ".join(["{0},{1}".format(*c) for c in path])
            s = ('<polyline fill="none" stroke="{2}" stroke-width="{1}" '
                 'points="{0}" opacity="0.8" />').format(pnt_format, 2. * scale_factor, stroke_color)
            paths.append(s)
        return "<g>" + "".join(paths) + "</g>"

'''

# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    a, plt, ax = _demo()
