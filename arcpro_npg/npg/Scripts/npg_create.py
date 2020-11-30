# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""
----------
npg_create
----------

Create geometry shapes.

Script :
    npg_create.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-10-23

Purpose
-------
Tools for creating arrays of various geometric shapes.

Notes
-----
Originally part of the `arraytools` module.

References
----------
`Trigonometric functions
<https://en.wikipedia.org/wiki/Trigonometric_functions>`_.

`List of mathematical shapes
<https://en.wikipedia.org/wiki/List_of_mathematical_shapes>`_.

`Circumscribed circle
<https://en.wikipedia.org/wiki/Circumscribed_circle>`_.

`Circumgon
<https://en.wikipedia.org/wiki/Circumgon>`_.

The vector from the incenter to the area centroid, GA , of a circumgonal
region and the vector from the incenter to the centroid of its boundary, GB,
(outer edge points) , are related by

>>> GB = 3./2. * GA   # where GA is the unit circle radius

`hexagon website
<https://www.redblobgames.com/grids/hexagons>`_.

`hexagonal tiling
<https://en.wikipedia.org/wiki/Hexagonal_tiling>`_.

hexagon :
    1.5, sqrt(3)/2
octagon :
    The coordinates for the vertices of a regular octagon centered at the
    origin and with side length 2 are:

    - (±1, ±(1+√2))
    - (±(1+√2), ±1)

Test::

    s = [3, 4, 5, 6, 8, 9, 10, 12]
    c0 = [(n, np.linspace(-180, 180., n+1, True)) for n in s]
    xs = np.cos(np.radians(c0[0][1]))
    ys = np.sin(np.radians(c0[0][1]))
    xy = np.array(list(zip(xs, ys)))
    out = []
    for i in c0:
        xs = np.cos(np.radians(i[1]))
        ys = np.sin(np.radians(i[1]))
        out.append(np.array(list(zip(xs, ys))))
    npg.plot_2d(out[:N], True, True)  # N, number of sides in range 3-->
    [np.mean(i[:-1], axis=0) for i in out]
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
# from textwrap import dedent
# from functools import wraps
import numpy as np

import npGeo as npg
from npg_plots import plot_mixed  # plot_2d, plot_polygons

np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 7.3f}'.format})
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['code_grid', 'rot_matrix',
           'arc_', 'arc_sector',
           'circle', 'circle_mini', 'circle_ring', 'circ_3pa', 'circ_3p',
           'ellipse',
           'rectangle',
           'triangle',
           'hex_flat', 'hex_pointy',
           'mesh_xy',
           'pyramid',
           'pnt_from_dist_bearing',
           'xy_grid',
           'transect_lines',
           'spiral_archim', 'spiral_sqr', 'spiral_cw', 'spiral_ccw',
           'base_spiral', 'to_spiral', 'from_spiral',
           'repeat', 'mini_weave'
           ]

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS


def code_grid(x_cols=1, y_rows=1,
              zero_based=False,
              shaped=True,
              bottom_up=False
              ):
    """Produce spreadsheet like labelling, either zero or 1 based.

    Parameters
    ----------
    cols, rows: integer
        make sure that the numbers are 1 or more... no checks for this
    zero-based: boolean
        zero yields A0, A1   ones yields A1, A2
    shaped: boolean
        True will shape the output to conform to array shape to match the rows
        and columns of the output
    bottom_up: boolean
        False is the default so that top_down conforms to array shapes

    Notes
    -----
    - list('0123456789')  # string.digits
    - import string .... string.ascii_uppercase

    This use padding A01 to facilitate sorting.
    If you want a different system change
    >>> >>> "{}{}".format(UC[c], r)    # A1 to whatever, no padding
    >>> "{}{:02.0f}".format(UC[c], r)  # A01 to ..99
    >>> "{}{:03.0f}".format(UC[c], r)  # A001 to A999
    >>> # etc

    >>> c0 = code_grid(
            cols=5, rows=3, zero_based=False, shaped=True, bottom_up=False
            )
    [['A01' 'B01' 'C01' 'D01' 'E01']
     ['A02' 'B02' 'C02' 'D02' 'E02']
     ['A03' 'B03' 'C03' 'D03' 'E03']]

    See Also
    --------
    ``code_grid.py`` for more details
    """
    alph = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    UC = [("{}{}").format(alph[i], alph[j]).strip()
          for i in range(27)
          for j in range(1, 27)]
    z = [1, 0][zero_based]
    rc = [1, 0][zero_based]
    c = ["{}{:02.0f}".format(UC[c], r)  # pull in the column heading
         for r in range(z, y_rows + rc)   # label in the row letter
         for c in range(x_cols)]          # label in the row number
    c = np.asarray(c)
    if shaped:
        c = c.reshape(y_rows, x_cols)
        if bottom_up:
            c = np.flipud(c)
    return c


# ---- helpers ---- rot_matrix -----------------------------------------------
#
def rot_matrix(angle=0, nm_3=False):
    """Return the rotation matrix given points and rotation angle.

    Parameters
    ----------
    Rotation angle in degrees and whether the matrix will be used with
    homogenous coordinates.

    Returns
    -------
    rot_m : matrix
        Rotation matrix for 2D transform.

    nm_3 : boolean
        True, is for use with homogenous coordinates, otherwise, leave as
        False

    Notes
    -----
    Rotate Geo array, ``g`` around a point...

    >>>> g.translate(-x, -y).rotate(theta).translate(x, y).
    """
    rad = np.deg2rad(angle)
    c = np.cos(rad)
    s = np.sin(rad)
    rm = np.array([[c, -s, 0.],
                   [s, c, 0.],
                   [0., 0., 1.]])
    if not nm_3:
        rm = rm[:2, :2]
    return rm


# ---- arc_sector, convex hull, circle ellipse, hexagons, rectangles,
#      triangle, xy-grid --
#
def arc_(radius=100, start=0, stop=1, step=0.1, xc=0.0, yc=0.0):
    """Create an arc from a specified radius, centre and start/stop angles.

    Parameters
    ----------
    radius : number
        cirle radius from which the arc is obtained
    start, stop, step : numbers
        angles in degrees
    xc, yc : number
        center coordinates in projected units
    as_list : boolean
        False, returns an array.  True yields a list

    Returns
    -------
      Points on the arc as an array

    >>> # arc from 0 to 90 in 5 degree increments with radius 2 at (0, 0)
    >>> a0 = arc_(radius=2, start=0, stop=90, step=5, xc=0.0, yc=0.0)
    """
    start, stop = sorted([start, stop])
    angle = np.deg2rad(np.arange(start, stop, step))
    x_s = radius * np.cos(angle)         # X values
    y_s = radius * np.sin(angle)         # Y values
    pnts = np.array([x_s, y_s]).T + [xc, yc]
    return pnts


def arc_sector(outer=10, inner=9, start=1, stop=6, step=0.1):
    """Form an arc sector bounded by a distance specified by two radii.

    Parameters
    ----------
    outer : number
        outer radius of the arc sector
    inner : number
        inner radius
    start : number
        start angle of the arc
    stop : number
        end angle of the arc
    step : number
        the angle densification step

    Requires
    --------
    `arc_` is used to produce the arcs, the top arc is rotated clockwise and
    the bottom remains in the order produced to help form closed-polygons.
    """
    s_s = [start, stop]
    s_s.sort()
    start, stop = s_s
    top = arc_(outer, start, stop, step, 0.0, 0.0)
    top = top[::-1]
    bott = arc_(inner, start, stop, step, 0.0, 0.0)
    close = top[0]
    pnts = np.concatenate((top, bott, [close]), axis=0)
    return pnts


def circle(radius=100, clockwise=True, theta=1, rot=0.0, scale=1,
           xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.

    Parameters
    ----------
    radius : number
        In projected units.
    clockwise : boolean
        True for clockwise (outer rings), False for counter-clockwise
        (for inner rings).
    theta : number
        Angle spacing. If theta=1, angles between -180 to 180, are returned
        in 1 degree increments. The endpoint is excluded.
    rot : number
         Rotation angle in degrees... used if scaling is not equal to 1.
    scale : number
         For ellipses, change the scale to <1 or > 1. The resultant
         y-values will favour the x or y-axis depending on the scaling.

    Returns
    -------
    List of coordinates for the circle/ellipse.

    Notes
    -----
    You can also use np.linspace if you want to specify point numbers.

    >>> np.linspace(start, stop, num=50, endpoint=True, retstep=False)
    >>> np.linspace(-180, 180, num=720, endpoint=True, retstep=False)
    """
    if clockwise:
        angles = np.deg2rad(np.arange(180.0, -180.0 - theta, step=-theta))
    else:
        angles = np.deg2rad(np.arange(-180.0, 180.0 + theta, step=theta))
    x_s = radius*np.cos(angles)            # X values
    y_s = radius*np.sin(angles) * scale    # Y values
    pnts = np.array([x_s, y_s]).T
    if rot != 0:
        rot_mat = rot_matrix(angle=rot)
        pnts = (np.dot(rot_mat, pnts.T)).T
    pnts = pnts + [xc, yc]
    return pnts


def circle_mini(radius=1.0, theta=10.0, xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.

    Parameters
    ----------
    radius : number
        Distance from centre
    theta : number
        Angle of densification of the shape around 360 degrees

    """
    angles = np.deg2rad(np.arange(180.0, -180.0 - theta, step=-theta))
    x_s = radius*np.cos(angles) + xc    # X values
    y_s = radius*np.sin(angles) + yc    # Y values
    pnts = np.array([x_s, y_s]).T
    return pnts


def circle_ring(outer=100, inner=0, theta=10, rot=0, scale=1, xc=0.0, yc=0.0):
    """Create a multi-ring buffer around a center point (xc, yc).

    Parameters
    ----------
    outer, inner : number
        Outer and inner radius in planar units
    theta : number
        See below.
    rot : number
        Rotation angle, used for non-circles.
    scale : number
        Used to scale the y-coordinates.

    Notes
    -----
    Angles to use to densify the circle::

    - 360+ circle
    - 120  triangle
    - 90   square
    - 72   pentagon
    - 60   hexagon
    - 45   octagon
    - etc
    """
    top = circle(outer, clockwise=True, theta=theta, rot=rot, scale=scale,
                 xc=xc, yc=yc)
    if inner == 0.0:
        return top
    bott = circle(inner, clockwise=False, theta=theta, rot=rot, scale=scale,
                  xc=xc, yc=yc)
    return np.concatenate((top, bott), axis=0)


def circ_3pa(arr):
    """Return a circle given a 3 point array.

    This is the same as ``circ3p`` but with a 3 pnt arr.
    """
    p, q, r = arr
    cx, cy, radius = circ_3p(p, q, r)
    return cx, cy, radius


def circ_3p(p, q, r):
    """Return a three point circle center and radius.

    A check is made for three points on a line.
    """
    temp = q[0] * q[0] + q[1] * q[1]
    bc = (p[0] * p[0] + p[1] * p[1] - temp) / 2
    cd = (temp - r[0] * r[0] - r[1] * r[1]) / 2
    # three points on a line check
    det = (p[0] - q[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[1] - q[1])
    if abs(det) < 1.0e-6:
        return None, None, np.inf
    # Center of circle
    cx = (bc * (q[1] - r[1]) - cd * (p[1] - q[1])) / det
    cy = ((p[0] - q[0]) * cd - (q[0] - r[0]) * bc) / det
    radius = np.sqrt((cx - p[0])**2 + (cy - p[1])**2)
    return cx, cy, radius


def ellipse(x_radius=1.0, y_radius=1.0,
            theta=10.,
            xc=0.0, yc=0.0,
            kind=2,
            asGeo=True,
            ):
    """Produce an ellipse depending on parameters.

    Parameters
    ----------
    radius : number
        Distance from centre in the X and Y directions.
    theta : number
        Angle of densification of the shape around 360 degrees.
    """
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = x_radius * np.cos(angles) + xc    # X values
    y_s = y_radius * np.sin(angles) + yc    # Y values
    pnts = np.array(list(zip(x_s, y_s)))
    if asGeo:
        if not isinstance(pnts, list):
            pnts = [pnts]
        frmt = "x_rad {}, y_rad {}, theta {}, x_c {}, y_c {}"
        txt = frmt.format(x_radius, y_radius, theta, xc, yc)
        return npg.arrays_to_Geo(pnts, kind=2, info=txt)
    return pnts


# ---- rectangles/squares, triangles, hexagons -------------------------------
#
# The following all share the same parameter list.
# x = cos(2kπ/n),y = sin(2kπ/n),k=1,2,3⋯n where ``n`` is the number of sides.
# general equation
def rectangle(dx=1, dy=-1,
              x_cols=1, y_rows=1,
              orig_x=0, orig_y=1,
              kind=2,
              asGeo=True,
              ):
    """Create a point array to represent a series of rectangles or squares.

    Parameters
    ----------
    dx, dy : number
        x direction increment, +ve moves west to east, left/right.
        y direction increment, -ve moves north to south, top/bottom.
    x_cols, y_rows : ints
        The number of columns and rows to produce.
    orig_x, orig_y : number
        Planar coordinates assumed.  You can alter the location of the origin
        by specifying the correct combination of (dx, dy) and (orig_x, orig_y).
        The defaults produce a clockwise, closed-loop geometry, beginning and
        ending in the upper left.
    kind, asGeo :
        These relate to Geo arrays

    Example
    -------
    Stating the obvious... squares form when dx == dy.

    Cells are constructed clockwise from the bottom-left.  The rectangular grid
    is constructed from the top-left.  Specifying an origin (upper left) of
    (0, 2) yields a bottom-right corner of (3,0) when the following are used.

    >>> z = rectangle(dx=1, dy=1, x_cols=3, y_rows=2, orig_x=0, orig_y=2,
                kind=2, asGeo=False)

    The first `cell` will be in the top-left and the last `cell` in the
    bottom-right.
    """
    X = [0.0, 0.0, dx, dx, 0.0]       # X, Y values for a unit square
    Y = [0.0, dy, dy, 0.0, 0.0]
    seed = np.array(list(zip(X, Y)))  # [dx0, dy0] keep for insets
    a = [seed + [j * dx, i * dy]      # make the shapes
         for i in range(0, y_rows)      # cycle through the rows
         for j in range(0, x_cols)]     # cycle through the columns
    a = np.asarray(a) + [orig_x, orig_y-dy]
    if asGeo:
        frmt = "dx {}, dy {}, x_cols {}, y_rows {}, LB ({},{})"
        txt = frmt.format(dx, dy, x_cols, y_rows, orig_x, orig_y)
        return npg.arrays_to_Geo(a, kind=2, info=txt)
    return a


def triangle(dx=1, dy=1,
             x_cols=1, y_rows=1,
             orig_x=0, orig_y=1,
             kind=2,
             asGeo=True,
             ):
    """Create a row of meshed triangles.

    The triangles are essentially bisected squares and not equalateral.
    The triangles per row will note be terminated in half triangles to
    `square off` the area of coverage.  This is to ensure that all geometries
    have the same area and point construction.

    Parameters
    ----------
    See `rectangles` for shared parameter explanation..
    """
    a, dx, b = dx/2.0, dx, dx*1.5
    Xu = [0.0, a, dx, 0.0]   # X, Y values for a unit triangle, point up
    Yu = [0.0, dy, 0.0, 0.0]
    Xd = [a, b, dx, a]       # X, Y values for a unit triangle, point down
    Yd = [dy, dy, 0.0, dy]   # shifted by dx
    seedU = np.vstack((Xu, Yu)).T  # np.array(list(zip(Xu, Yu)))
    seedD = np.vstack((Xd, Yd)).T  # np.array(list(zip(Xd, Yd)))
    seed = np.array([seedU, seedD])
    a = [seed + [j * dx, i * dy]       # make the shapes
         for i in range(0, y_rows)       # cycle through the rows
         for j in range(0, x_cols)]      # cycle through the columns
    a = np.asarray(a)
    s1, s2, s3, s4 = a.shape
    a = a.reshape(s1 * s2, s3, s4)
    if asGeo:
        frmt = "dx {}, dy {}, x_cols {}, y_rows {}, LB ({},{})"
        txt = frmt.format(dx, dy, x_cols, y_rows, orig_x, orig_y)
        return npg.arrays_to_Geo(a, kind=2, info=txt)
    return a


def hex_flat(dx=1, dy=1,
             x_cols=1, y_rows=1,
             orig_x=0, orig_y=0,
             kind=2,
             asGeo=True):
    """Generate the points for the flat-headed hexagon.

    Parameters
    ----------
    See `rectangles` for shared parameter explanation.
    """
    f_rad = np.deg2rad([180., 120., 60., 0., -60., -120., -180.])
    X = np.cos(f_rad) * dy
    Y = np.sin(f_rad) * dy            # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))  # array of coordinates
    dx = dx * 1.5
    dy = dy * np.sqrt(3.) / 2.0
    hexs = [seed + [dx * i, dy * (i % 2)] for i in range(0, x_cols)]
    m = len(hexs)
    for j in range(1, y_rows):  # create the other rows
        hexs += [hexs[h] + [0, dy * 2 * j] for h in range(m)]
    if asGeo:
        frmt = "dx {}, dy {}, x_cols {}, y_rows {}, LB ({},{})"
        txt = frmt.format(dx, dy, x_cols, y_rows, orig_x, orig_y)
        return npg.arrays_to_Geo(hexs, kind=2, info=txt)
    return hexs


def hex_pointy(dx=1, dy=1,
               x_cols=1, y_rows=1,
               orig_x=0, orig_y=0,
               kind=2,
               asGeo=True):
    """Create pointy hexagons. Also called ``traverse hexagons``.

    Parameters
    ----------
    See `rectangles` for shared parameter explanation.
    """
    p_rad = np.deg2rad([150., 90, 30., -30., -90., -150., 150.])
    X = np.cos(p_rad) * dx
    Y = np.sin(p_rad) * dy      # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))
    dx = dx * np.sqrt(3.)/2.0
    dy = dy * 1.5
    hexs = [seed + [dx * i * 2, 0] for i in range(0, x_cols)]
    m = len(hexs)
    for j in range(1, y_rows):  # create the other rows
        hexs += [hexs[h] + [dx * (j % 2), dy * j] for h in range(m)]
    if asGeo:
        frmt = "dx {}, dy {}, x_cols {}, y_rows {}, LB ({},{})"
        txt = frmt.format(dx, dy, x_cols, y_rows, orig_x, orig_y)
        return npg.arrays_to_Geo(hexs, kind=2, info=txt)
    return hexs


# ---- others ---------------------------------------------------------------
#
def mesh_xy(L=0, B=0, R=5, T=5, dx=1, dy=1, as_rec=True):
    """Create a mesh of coordinates within the specified X, Y ranges.

    Parameters
    ----------
    L(eft), R(ight), dx : number
        Coordinate min, max and delta x for X axis.
    B(ott), T(op), dy  : number
        Same as above for Y axis.
    as_rec : boolean
        Produce a structured array (or convert to a record array).

    Returns
    -------
    -  A list of coordinates of X,Y pairs and an ID if as_rec is True.
    -  A mesh grid X and Y coordinates is also produced.
    """
    dt = [('Pnt_num', '<i4'), ('X', '<f8'), ('Y', '<f8')]
    x = np.arange(L, R + dx, dx, dtype='float64')
    y = np.arange(B, T + dy, dy, dtype='float64')
    mesh = np.meshgrid(x, y, sparse=False)
    if as_rec:
        xs = mesh[0].ravel()
        ys = mesh[1].ravel()
        p = list(zip(np.arange(len(xs)), xs, ys))
        pnts = np.array(p, dtype=dt)
    else:
        p = list(zip(mesh[0].ravel(), mesh[1].ravel()))
        pnts = np.array(p)
    return pnts, mesh


def pyramid(core=9, steps=10, incr=(1, 1), posi=True):
    """Create a pyramid.  See pyramid_demo.py."""
    a = np.array([core])
    a = np.atleast_2d(a)
    for i in range(1, steps):
        val = core - i
        if posi and (val <= 0):
            val = 0
        a = np.lib.pad(a, incr, "constant", constant_values=(val, val))
    return a


def pnt_from_dist_bearing(orig=(0, 0), bearings=None, dists=None, prn=False):
    """Return point locations given distance and bearing from an origin.

    Calculate the point coordinates from distance and angle.

    References
    ----------
    `<https://community.esri.com/thread/66222>`_.

    `<https://community.esri.com/blogs/dan_patterson/2018/01/21/
    origin-distances-and-bearings-geometry-wanderings>`_.

    Notes
    -----
    Planar coordinates are assumed.  Use Vincenty if you wish to work with
    geographic coordinates.

    Sample calculation::

        bearings = np.arange(0, 361, 22.5)  # 17 bearings
        dists = np.random.randint(10, 500, len(bearings)) * 1.0  OR
        dists = np.full(bearings.shape, 100.)
        data = dist_bearing(orig=orig, bearings=bearings, dists=dists)

    Create a featureclass from the results::

        shapeXY = ['X_to', 'Y_to']
        fc_name = 'C:/path/Geodatabase.gdb/featureclassname'
        arcpy.da.NumPyArrayToFeatureClass(
            out, fc_name, ['X_to', 'Y_to'], "2951")
        # ... syntax
        arcpy.da.NumPyArrayToFeatureClass(
            in_array=out, out_table=fc_name, shape_fields=shapeXY,
            spatial_reference=SR)
    """
    error = "An origin with distances and bearings of equal size are required."
    orig = np.array(orig)
    if bearings is None or dists is None:
        raise ValueError(error)
    iterable = np.all([isinstance(i, (list, tuple, np.ndarray))
                       for i in [dists, bearings]])
    if iterable:
        if not (len(dists) == len(bearings)):
            raise ValueError(error)
    else:
        raise ValueError(error)
    rads = np.deg2rad(bearings)
    dx = np.sin(rads) * dists
    dy = np.cos(rads) * dists
    x_t = np.cumsum(dx) + orig[0]
    y_t = np.cumsum(dy) + orig[1]
    stack = (x_t, y_t, dx, dy, dists, bearings)
    names = ["X_to", "Y_to", "orig_dx", "orig_dy", "distance", "bearing"]
    data = np.vstack(stack).T
    N = len(names)
    if prn:  # ---- just print the results ----------------------------------
        frmt = "Origin ({}, {})\n".format(*orig) + "{:>10s}"*N
        print(frmt.format(*names))
        frmt = "{: 10.2f}"*N
        for i in data:
            print(frmt.format(*i))
        return data
    # ---- produce a structured array from the output -----------------------
    names = ", ".join(names)
    kind = ["<f8"]*N
    kind = ", ".join(kind)
    out = data.transpose()
    out = np.core.records.fromarrays(out, names=names, formats=kind)
    return out


def xy_grid(x, y=None, top_left=True):
    """Create a 2D array of locations from x, y values.

    The values need not  be uniformly spaced just sequential.
    Derived from `meshgrid` in References.

    Parameters
    ----------
    x, y : array-like
        To form a mesh, there must at least be 2 values in each sequence
    top_left: boolean
        True, y's are sorted in descending order, x's in ascending

    References
    ----------
    `<https://github.com/numpy/numpy/blob/master/numpy/lib/function_base.py>`_.
    """
    x = np.array(x)
    if y is None:
        y = x
    y = np.array(y)
    if x.ndim != 1:
        return "A 1D array required"
    xs = np.sort(np.asanyarray(x))
    ys = np.asanyarray(y)
    if top_left:
        ys = ys[np.argsort(-ys)]
    xs = np.reshape(xs, newshape=((1,) + xs.shape))
    ys = np.reshape(ys, newshape=(ys.shape + (1,)))
    xy = [xs, ys]
    xy = np.broadcast_arrays(*xy, subok=True)
    shp = np.prod(xy[0].shape)
    final = np.zeros((shp, 2), dtype=xs.dtype)
    final[:, 0] = xy[0].ravel()
    final[:, 1] = xy[1].ravel()
    return final


def transect_lines(N=5, orig=None, dist=1, x_offset=0, y_offset=0,
                   bearing=0, as_ndarray=True):
    """Construct transect lines from origin-destination points.

    The distance and bearings are from the origin point.

    Parameters
    ----------
    N : number
        The number of transect lines.
    orig : array-like
         A single origin.  If None, the cartesian origin (0, 0) is used.
    dist : number or array-like
        The distance(s) from the origin
    x_offset, y_offset : number
        If the `orig` is a single location, you can construct offset lines
        using these values.
    bearing : number or array-like
        If a single number, parallel lines are produced. An array of values
        equal to the `orig` can be used.

    Returns
    -------
    Two outputs are returned, the first depends on the `as_ndarray` setting.

    1. True, a structured array. False - a recarray
    2. An ndarray with the field names in case the raw data are required.

    Notes
    -----
    It is easiest of you pick a `corner`, then use x_offset, y_offset to
    control whether you are moving horizontally and vertically from the origin.
    The bottom left is easiest, and positive offsets move east and north from.

    Use XY to Line tool in ArcGIS Pro to convert the from/to pairs to a line.
    See references.

    Examples
    --------
    >>> out, data = transect_lines(N=5, orig=None,
                                   dist=100, x_offset=10,
                                   y_offset=0, bearing=45, as_ndarray=True)
    >>> data
    array([[  0.  ,   0.  ,  70.71,  70.71],
           [ 10.  ,   0.  ,  80.71,  70.71],
           [ 20.  ,   0.  ,  90.71,  70.71],
           [ 30.  ,   0.  , 100.71,  70.71],
           [ 40.  ,   0.  , 110.71,  70.71]])
    >>> out
    array([( 0., 0.,  70.71, 70.71), (10., 0.,  80.71, 70.71),
    ...    (20., 0.,  90.71, 70.71), (30., 0., 100.71, 70.71),
    ...    (40., 0., 110.71, 70.71)],
    ...   dtype=[('X_from', '<f8'), ('Y_from', '<f8'),
    ...          ('X_to', '<f8'), ('Y_to', '<f8')])
    ...
    ... Create the table and the lines
    >>> tbl = 'c:/folder/your.gdb/table_name'
    >>> # arcpy.da.NumPyArrayToTable(a, tbl)
    >>> # arcpy.XYToLine_management(
    ... #       in_table, out_featureclass,
    ... #       startx_field, starty_field, endx_field, endy_field,
    ... #       {line_type}, {id_field}, {spatial_reference}
    ... This is general syntax, the first two are paths of source and output
    ... files, followed by coordinates and options parameters.
    ...
    ... To create compass lines
    >>> b = np.arange(0, 361, 22.5)
    >>> a, data = transect_lines(N=10, orig=[299000, 4999000],
                                 dist=100, x_offset=0, y_offset=0,
                                 bearing=b, as_ndarray=True)

    References
    ----------
    `<https://community.esri.com/blogs/dan_patterson/2019/01/17/transect-
    lines-parallel-lines-offset-lines>`_.

    `<http://pro.arcgis.com/en/pro-app/tool-reference/data-management
    /xy-to-line.htm>`_.
    """
    def _array_struct_(a, fld_names=['X', 'Y'], kinds=['<f8', '<f8']):
        """Convert an array to a structured array."""
        dts = list(zip(fld_names, kinds))
        z = np.zeros((a.shape[0],), dtype=dts)
        for i in range(a.shape[1]):
            z[fld_names[i]] = a[:, i]
        return z
    #
    if orig is None:
        orig = np.array([0., 0.])
    args = [orig, dist, bearing]
    arrs = [np.atleast_1d(i) for i in args]
    orig, dist, bearing = arrs
    # o_shp, d_shp, b_shp = [i.shape for i in arrs]
    #
    rads = np.deg2rad(bearing)
    dx = np.sin(rads) * dist
    dy = np.cos(rads) * dist
    #
    n = len(bearing)
    N = [N, n][n > 1]  # either the number of lines or bearings
    x_orig = np.arange(N) * x_offset + orig[0]
    y_orig = np.arange(N) * y_offset + orig[1]
    x_dest = x_orig + dx
    y_dest = y_orig + dy
    # ---- create the output array
    names = ['X_from', 'Y_from', 'X_to', 'Y_to']
    x_cols = len(names)
    kind = ['<f8']*x_cols
    data = np.vstack([x_orig, y_orig, x_dest, y_dest]).T
    if as_ndarray:  # **** add this as a flag
        out = _array_struct_(data, fld_names=names, kinds=kind)
    else:
        out = data.transpose()
        out = np.core.records.fromarrays(out, names=names, formats=kind)
    return out, data


def spiral_archim(N, n, inward=False, clockwise=True):
    """Create an Archimedes spiral in the range 0 to N points with 'n' steps.

    Parameters
    ----------
    N : integer
        The range of the spiral as `N` points
    n : integer
        The number of points between steps.
    inward : boolean
        Whether the points radiate inward toward, or outward from, the center.
    clockwise : boolean
        Direction of rotation

    Notes
    -----
    When n is small relative to N, then you begin to form rectangular
    spirals, like rotated rectangles.

    With N = 360, n = 20 yields 360 points with 2n points (40) to complete each
    360 degree loop of the spiral.
    """
    rnge = np.arange(0.0, N)
    if inward:
        rnge = rnge[::-1]
    phi = rnge/n * np.pi
    xs = phi * np.cos(phi)
    ys = phi * np.sin(phi)
    if clockwise:
        xy = np.c_[ys, xs]
    else:
        xy = np.c_[xs, ys]
    return xs, ys, xy


def spiral_sqr(ULx=-10, n_max=100):
    """Create a square spiral from the centre in a clockwise direction

    Parameters
    ----------
    ULx : number
        This is the upper left x coordinate, relative to center (0, 0).
    n-max : number
        The maximum number of iterations should ULx not be reached.

    Notes
    -----
        See spirangle, Ulam spiral.
    """
    def W(x, y, c):
        x -= c[0]
        return x, y, c

    def S(x, y, c):
        y -= c[1]
        return x, y, c

    def E(x, y, c):
        x += c[2]
        return x, y, c

    def N(x, y, c):
        y += c[3]
        return x, y, c

    c = np.array([1, 1, 2, 2])
    pos = [0, 0, c]
    n = 0
    v = [pos]
    cont = True
    while cont:
        p0 = W(*v[-1])
        p1 = S(*p0)
        p2 = E(*p1)
        p3 = N(*p2)
        c = c + 2
        p3 = [p3[0], p3[1], c]
        for i in [p0, p1, p2, p3]:
            v.append(i)
        # --- print(p0, p0[0])  # for testing
        if (p0[0] <= ULx):      # bail option 1
            cont = False
        if n > n_max:           # bail option 2
            cont = False
        n = n+1
    coords = np.asarray([np.array([i[0], i[1]]) for i in v])[:-3]
    return coords


# -------Excellent one-------------------------------------------------------
#  https://stackoverflow.com/questions/36834505/
#        creating-a-spiral-array-in-python
def spiral_cw(A):
    """Docstring"""
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0])        # take first row
        A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def spiral_ccw(A):
    """Docstring"""
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0][::-1])    # first row reversed
        A = A[1:][::-1].T         # cut off first row and rotate clockwise
    return np.concatenate(out)


def base_spiral(nrow, ncol):
    """Docstring"""
    return spiral_ccw(np.arange(nrow*ncol).reshape(nrow, ncol))[::-1]


def to_spiral(A):
    """Docstring"""
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B


def from_spiral(A):
    """Docstring"""
    A = np.array(A)
    return A.flat[base_spiral(*A.shape)].reshape(A.shape)

# ---- end code section--------------------------------------


def repeat(seed=None, corner=[0, 0], x_cols=1, y_rows=1, angle=0):
    """Create the array of pnts to pass to arcpy .

    Numpy magic is used to produce a fishnet of the desired in_shp.

    Parameters
    ----------
    seed : use grid_array, hex_flat or hex_pointy.
        You specify the width and height or its ratio when making the shapes.
    corner : array-like
        Lower left corner of the shape pattern.
    dx, dy : numbers
        Offset of the shapes... this is different.
    x_cols, y_rows : integers
        The number of y_rows and columns to produce.
    angle : number
        Rotation angle in degrees.
    """
    def rotate(pnts, angle=0):
        """Rotate points about the origin in degrees, (+ve for clockwise)."""
        angle = np.deg2rad(angle)                 # convert to radians
        s = np.sin(angle)
        c = np.cos(angle)    # rotation terms
        aff_matrix = np.array([[c, s], [-s, c]])  # rotation matrix
        XY_r = np.dot(pnts, aff_matrix)           # numpy magic to rotate pnts
        return XY_r
    # ----
    if seed is None:
        a = rectangle(dx=1, dy=1, x_cols=3, y_rows=3)
    else:
        a = np.asarray(seed)
    if angle != 0:
        a = [rotate(p, angle) for p in a]        # rotate the scaled points
    pnts = [p + corner for p in a]               # translate them
    return pnts


def mini_weave(n):
    """Inter-weave two arrays of ``n`` segments.

    Parameters
    ----------
    n : segments
       z is sliced to ensure compliance.

    >>> a = mini_weave(11)
    >>> e_leng(a)
    | total 14.142135623730953,
    | segment [1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]
    """
    # root2 = np.full(n, np.sqrt(2.))
    one_s = np.ones(n)
    zero_s = np.zeros(n)
    x = np.arange(n)
    y = np.zeros(n)
    z = np.asarray([*sum(zip(zero_s, one_s), ())])[:n]
    a = np.vstack((x, y, z)).T
    return a


def _test_data(plot=False):
    """Sample test data.

    a, r = _test_data(plot=False)
    """
    a = np.array(
        [[0.4, 0.5], [1.2, 9.1], [1.2, 3.6], [1.9, 4.6], [2.9, 5.9],
         [4.2, 5.5], [4.3, 3.0], [5.1, 8.2], [5.3, 9.5], [5.5, 5.7],
         [6.1, 4.0], [6.5, 6.8], [7.1, 7.6], [7.3, 2.0], [7.4, 1.0],
         [7.7, 9.6], [8.5, 6.5], [9.0, 4.7], [9.6, 1.6], [9.7, 9.6]]
    )
    # rand_state = np.random.RandomState(123)
    b = np.random.random(size=(100, 2))*10
    h, e = np.histogramdd(b, [np.arange(11), np.arange(11)])
    r = rectangle(
        dx=1, dy=-1, x_cols=10, y_rows=10, orig_x=0, orig_y=10, kind=2,
        asGeo=True
    )
    data = [[b, 0, 'black', 'o', False], [r.shapes, 2, 'red', '.', True]]
    if plot:
        plot_mixed(data)
    return a, r, b, h, e


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    print("Script path {}".format(script))
    # a = _test_data(plot=False)
