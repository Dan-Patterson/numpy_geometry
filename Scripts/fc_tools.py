# -*- coding: UTF-8 -*-
"""
==========
_common.py
==========

Script : _common.py
    Common methods for the featureclass tool
Author :
    Dan_Patterson@carleton.ca
Modified : 2019-06-16
    Initial creation between 2016-2018
Purpose :
    Common tools for working with numpy arrays and featureclasses.

Requires
--------
numpy and arcpy

References :


"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=E0611  # stifle the arcgisscripting
# pylint: disable=E1101  # ditto for arcpy
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
#from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = [
    'tweet', 'de_punc',
    '_describe', 'getSR',
    'fc_composition', 'fc_fld_info', 'fc_geom_info', 'fc_info',
    'null_dict', 'tbl_arr'
    ]


def tweet(msg):
    """Print a message for both arcpy and python.
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


def de_punc(s, punc=None, no_spaces=True, char='_'):
    """Remove punctuation and/or spaces in strings and replace with
    underscores or nothing

    Parameters
    ----------
    s : string
        input string to parse
    punc : string
        A string of characters to replace ie. '@ "!\'\\[]'
    no_spaces : boolean
        True, replaces spaces with underscore.  False, leaves spaces
    char : string
        Replacement character
    """
    if (punc is None) or not isinstance(punc, str):
        punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'  # _ removed
    if no_spaces:
        punc = " " + punc
    s = "".join([[i, char][i in punc] for i in s])
    return s

# ----------------------------------------------------------------------------
# ---- Geometry objects and generic geometry/featureclass functions ----------
# ----------------------------------------------------------------------------
def _describe(in_fc):
    """Simply return the arcpy.da.Describe object.

    **desc.keys()** an abbreviated list::

    'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
    'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
    'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
    'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
    'spatialReference'

    """
    return arcpy.da.Describe(in_fc)


def getSR(in_fc, verbose=False):
    """Return the spatial reference of a featureclass"""
    desc = arcpy.da.Describe(in_fc)
    SR = desc['spatialReference']
    if verbose:
        print("SR name: {}  factory code: {}".format(SR.name, SR.factoryCode))
    return SR


def fc_composition(in_fc, SR=None, prn=True, start=0, end=10):
    """Featureclass geometry composition in terms of shapes, shape parts, and
    point counts for each part.
    """
    if SR is None:
        SR = getSR(in_fc)
    with arcpy.da.SearchCursor(
            in_fc, ['OID@', 'SHAPE@'], spatial_reference=SR) as cur:
        len_lst = []
        for _, row in enumerate(cur):
            oid, p = row[0], row[1]
            parts = p.partCount
            num_pnts = np.asarray([p[i].count for i in range(parts)])
            IDs = np.repeat(oid, parts)
            part_count = np.arange(parts)
            too = np.cumsum(num_pnts)
            result = np.stack((IDs, part_count, num_pnts, too), axis=-1)
            len_lst.append(result)
    tmp = np.vstack(len_lst)
    too = np.cumsum(tmp[:, 2])
    frum = np.concatenate(([0], too))
    frum_too = np.array(list(zip(frum, too)))
    fc_comp = np.hstack((tmp[:, :3], frum_too)) #, axis=0)
    dt = np.dtype({'names':['IDs', 'Part', 'Points', 'From_pnt', 'To_pnt'],
                   'formats': ['i4', 'i4', 'i4', 'i4', 'i4']})
    fc = uts(fc_comp, dtype=dt)
    if prn:
        frmt = """\n{}\nShapes :   {}\nParts  :   {:,}\n  max  :   {}\n""" + \
        """Points :   {:,}\n  min  :   {}\n  median : {}\n  max  :   {:,}"""
        uni, cnts = np.unique(fc['IDs'], return_counts=True)
        a0, a1 = [fc['Part'] + 1, fc['Points']]
        args = [in_fc, len(uni), np.sum(cnts), np.max(a0),
                np.sum(a1), np.min(a1), int(np.median(a1)), np.max(a1)]
        msg = dedent(frmt).format(*args)
        print(msg)
        # ---- to structured and print
        frmt = "{:>8} "*5
        start, end = sorted([abs(int(i)) if isinstance(i, (int, float))
                             else 0 for i in [start, end]])
        print(frmt.format(*fc.dtype.names))
        end = min([fc.shape[0], end])
        for i in range(start, end):
            print(frmt.format(*fc[i]))
        return None
    return fc


def fc_fld_info(in_fc, prn=False):
    """Field information for a featureclass (in_fc).

    Parameters
    ----------
    prn : boolean
        True - returns the values

        False - simply prints the results

    Field properties
    ----------------
    'aliasName', 'baseName', 'defaultValue', 'domain', 'editable',
    'isNullable', 'length', 'name', 'precision', 'required', 'scale', 'type'
    """
    flds = arcpy.ListFields(in_fc)
    f_info = [(i.name, i.type, i.length, i.isNullable, i.required)
              for i in flds]
    f = "{!s:<14}{!s:<12}{!s:>7} {!s:<10}{!s:<10}"
    if prn:
        frmt = "FeatureClass:\n   {}\n".format(in_fc)
        args = ["Name", "Type", "Length", "Nullable", "Required"]
        frmt += f.format(*args) + "\n"
        frmt += "\n".join([f.format(*i) for i in f_info])
        tweet(frmt)
        return None
    return f_info


def fc_geom_info(in_fc, SR=None, prn=True, start=0, num=10):
    """Featureclass geometry composition in terms of shapes, shape parts, and
    point counts for each part.

    Parameters
    ----------
    in_fc : geodatabase featureclass
        Tested with file gdbs only
    SR : spatial reference
        Either the object, the name, or code
    prn : boolean
        True, prints the information.  False, returns the values as text.
    start, num : integer
        The start shape and number of shapes to return information for.
        Note:  Arrays are 0 indexed, featureclasses are 1 indexed.
    """
    good = np.all([isinstance(i, int) for i in [start, num]])
    if not good:
        print("Check your parameters:\n\n{}".format(fc_geom_info.__doc__))
        return None
    if SR is None:
        SR = getSR(in_fc)
    with arcpy.da.SearchCursor(
            in_fc, ['OID@', 'SHAPE@'], spatial_reference=SR) as cur:
        len_lst = []
        for _, row in enumerate(cur):
            oid, p = row[0], row[1]
            parts = p.partCount
            num_pnts = np.asarray([p[i].count for i in range(parts)])
            N = np.sum(num_pnts)
            too = np.sum(num_pnts)
            result = np.stack((oid, parts, N, too), axis=-1)
            len_lst.append(result)
    tmp = np.vstack(len_lst)
    too = np.cumsum(tmp[:, 2])
    frum = np.concatenate(([0], too))
    frum_too = np.array(list(zip(frum, too)))
    fc_comp = np.hstack((tmp[:, :3], frum_too)) #, axis=0)
    dt = np.dtype({'names':['Shape', 'Parts', 'Points', 'From_pnt', 'To_pnt'],
                   'formats': ['i4', 'i4', 'i4', 'i4', 'i4']})
    fc = uts(fc_comp, dtype=dt)
    if prn:
        print("\nFeatureclass:\n    {}".format(in_fc))
        frmt = "{:>8} "*5
        print(frmt.format(*fc.dtype.names))
        start, num = [abs(i) for i in [start, num]]
        end = min([fc.shape[0], start + num])
        for i in range(start, end):
            print(frmt.format(*fc[i]))
        return None
    return fc


def fc_info(in_fc, prn=False):
    """Return basic featureclass information, including the following...

    Returns:
    --------
    shp_fld  :
        field name which contains the geometry object
    oid_fld  :
        the object index/id field name
    SR       :
        spatial reference object (use SR.name to get the name)
    shp_type :
        shape type (Point, Polyline, Polygon, Multipoint, Multipatch)

    Notes:
    ------
    Other useful parameters::

    'areaFieldName', 'baseName', 'catalogPath','featureType',
    'fields', 'hasOID', 'hasM', 'hasZ', 'path'

    Derive all field names:

    >>> all_flds = [i.name for i in desc['fields']]
    """
    desc = _describe(in_fc)
    args = ['shapeFieldName', 'OIDFieldName', 'shapeType', 'spatialReference']
    shp_fld, oid_fld, shp_type, SR = [desc[i] for i in args]
    if prn:
        frmt = "FeatureClass:\n   {}".format(in_fc)
        f = "\n{!s:<16}{!s:<14}{!s:<10}{!s:<10}"
        frmt += f.format(*args)
        frmt += f.format(shp_fld, oid_fld, shp_type, SR.name)
        tweet(frmt)
        return None
    return shp_fld, oid_fld, shp_type, SR


def null_dict(flds):
    """Produce a null dictionary from a list of fields
    These must be field objects and not just their name.
    """
    dump_flds = ["OBJECTID", "Shape_Length", "Shape_Area", "Shape"]
    flds_oth = [f for f in flds
                if f.name not in dump_flds]
#    oid_geom = ['OBJECTID', 'SHAPE@X', 'SHAPE@Y']
    nulls = {'Double':np.nan,
             'Single':np.nan,
             'Short':np.iinfo(np.int16).min,
             'SmallInteger':np.iinfo(np.int16).min,
             'Long':np.iinfo(np.int32).min,
             'Float':np.nan,
             'Integer':np.iinfo(np.int32).min,
             'String':str(None),
             'Text':str(None)}
    fld_dict = {i.name: i.type for i in flds_oth}
    nulls = {f.name:nulls[fld_dict[f.name]] for f in flds_oth}
    return nulls


def tbl_arr(pth):
    """Convert featureclass/table to a structured ndarray

    Parameters
    ----------
    pth : string
        path to input featureclass or table

    """
    flds = arcpy.ListFields(pth)
    nulls = null_dict(flds)
    bad = ['OID', 'Geometry', 'Shape_Length', 'Shape_Area']
    f0 = ["OID@"]
    f1 = [i.name for i in flds if i.type not in bad]
    flds = f0 + f1
    a = arcpy.da.TableToNumPyArray(
            pth, field_names=flds, skip_nulls=False, null_value=nulls
            )
    dt = np.array(a.dtype.descr)
    nmes = dt[:, 0]
    sze = dt[:, 1]
    cleaned = []
    for i in nmes:
        i = de_punc(i)  # run de_punc to remove punctuation
        cleaned.append(i)
    a.dtype = list(zip(cleaned, sze))
    return a


#def arr_csv(a):
#    """Format a structured/recarray to csv format
#    """
#    pass
# ---- extras ----------------------------------------------------------------



# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally... print the script source name. run the _demo """
#    print("Script... {}".format(script))
