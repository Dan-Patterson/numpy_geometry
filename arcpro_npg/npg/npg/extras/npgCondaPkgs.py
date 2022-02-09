# -*- coding: utf-8 -*-
r"""
------------
npgCondaPkgs
------------

----

Script :
    npg_table.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-08-12

Purpose
-------
Determine conda package information related to arcgis pro.  There are two
variants.

- pkg_info_json(folder=None)
  This uses json files in a specific folder in the arcgis installation path by
  default, or you can specify a different folder containing json files.

- package_info_conda(folder=None)
  The user profile installation path
Notes
-----
The following can be used if you need an output text file.

out_file : text  *`currently commented out`*
    The filename if an output `*.csv` is desired, e.g. `c:/temp/out.csv'.
    The file's field names are `Package, Filename, Dependencies`.  The
    latter field is delimited by `; ` (semicolon space) if there are
    multiple dependencies.
# if out_file:  # commented out to just return arrays
      hdr = ", ".join(packages.dtype.names)
      frmt = "%s, %s, %s"
      np.savetxt(out_file, packages, fmt=frmt, header=hdr, comments="")
      print("\nFile saved...")

"""

import sys
import os
# from textwrap import dedent
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as uts
from pathlib import Path
import json
from arcpy.da import NumPyArrayToTable

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=300, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def parse_json(f, key="depends"):
    """Parse the json file."""
    with open(f, "r") as f:
        data = json.load(f)
    keys = list(data.keys())
    if key in keys:
        return data[key]
    else:
        return []


def pkg_info_json(folder=None):
    r"""Access package info from `*.json` files in a `folder`.

    Parameters
    ----------
    folder : text
        File path to the `json` file. By default, this can be derived from
    >>> sys.prefix
    ... r"C:\arc_pro\bin\Python\envs\arcgispro-py3"
    >>> # ---- conda-meta is appended to it to yield `folder`, see *`Example`*

    Notes
    -----
    The keyword to search on is **depends**.
    Other options in json files include::

        arch, auth, build, build_number, channel, depends, files, fn,
        has_prefix, license, link, md5, name, noarch, platform, preferred_env,
        priority, requires, schannel, size, subdir, timestamp, url, version,
        with_features_depends

    Example
    -------
    folder = "C:/...install path/bin/Python/envs/arcgispro-py3/conda-meta" ::

        folder = sys.prefix + "/conda-meta"
        packages, dep_counts, required_by = pkg_info_json(folder)
        f0 = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\dep_pkg_info"
        f1 = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\dep_counts"
        f2 = r"C:\Git_Dan\npgeom\Project_npg\npgeom.gdb\dep_required_by"
        arcpy.da.NumPyArrayToTable(packages, f0)
        arcpy.da.NumPyArrayToTable(dep_counts, f1)
        arcpy.da.NumPyArrayToTable(required_by, f2)

    """
    # ---- Checks
    if not folder:
        folder = sys.prefix + "\\conda-meta"
    folder = Path(folder)
    if not folder.is_dir():
        print("\nInvalid path... {}".format(folder))
        return
    files = list(folder.glob("*.json"))
    if not files:
        print("{} doesn't have any json files".format(folder))
        return
    #
    # --- Package, Filename, Dependencies
    packages = []
    m0 = m1 = m2 = 0
    for f in files:
        ret = parse_json(f, key="depends")  # ---- look at dependencies only
        nme = str(f.name).rsplit("-", 2)[0]  # ---- split off the last two
        if len(ret) == 1:
            ret = ret[0]
        elif len(ret) > 1:
            srted = sorted(ret)
            ret = "; ".join([i for i in srted if "py" not in i])  # `; ` used
        else:
            ret = "None"
        m0 = max(m0, len(nme))
        m1 = max(m1, len(str(f.name)))
        m2 = max(m2, len(ret))
        packages.append((nme, f.name, ret))
    dt1 = [("Package", "<U{}".format(m0)), ("Filename", "<U{}".format(m1)),
           ("Dependencies", "<U{}".format(m2))]
    packages = np.asarray(packages, dtype=dt1)
    #
    # ---- Dependency, Counts
    z = []
    for dep in packages['Dependencies']:
        if dep not in ("", " "):
            z += dep.split("; ")  # split on `; ` delimiter
    z = np.asarray(z)
    uniq, idx, cnts = np.unique(z, return_index=True, return_counts=True)
    uniq2 = [[u, u.split(" ")[0]][" " in u] for u in uniq if u != ""]
    m0 = max(np.char.str_len(uniq2))
    m1 = np.max(np.char.str_len(uniq2)) + 5
    dt2 = [("Full_name", "<U{}".format(m0)), ("Counts", "i8"),
           ("Simple_name", "<U{}".format(m1))]
    dep_counts = np.asarray(list(zip(uniq, cnts, uniq2)), dtype=dt2)
    #
    # ---- Package, Required_by
    required_by = []
    names = packages['Package']
    depends = packages['Dependencies']
    max_len = 0
    for nme in names:
        if nme in ('py', 'python'):
            required_by.append([nme, "many"])
            continue
        w = names[[nme in i for i in depends]]
        if np.size(w) > 0:
            v = w.tolist()
            v0 = "; ".join([i.split("; ")[0] for i in v])
            max_len = max(max_len, len(v0))
            required_by.append([nme, v0])
        else:
            required_by.append([nme, "None"])
    r_dt = "<U{}".format(max_len)
    dt = np.dtype([('Package', '<U30'), ('Required_by', r_dt)])
    required_by = uts(np.asarray(required_by), dtype=dt)
    return packages, dep_counts, required_by


def pkg_info_conda(folder=None):
    r"""Access package info.

    Parameters
    ----------
    folder : text
        Path to the `user` conda folder in installed with arcgis pro.

    Requires
    --------
    `os` and `json` modules.

    Example
    -------
    ::

        folder = r'C:\Users\dan_p\AppData\Local\ESRI\conda\pkgs'
        sub_folder="info"
        file_name = "index.json"
        text = "python"
        np.isin(out['Depend'], 'python')

    """
    out = []
    if folder is None:
        arc_pth = r"\AppData\Local\ESRI\conda\pkgs"
        user = os.path.expandvars("%userprofile%")
        folder = "{}{}".format(user, arc_pth)
        if not os.path.isdir(folder):
            print("{} doesn't exist".format(folder))
            return None
    dir_lst = os.listdir(folder)
    max_len = 0
    for d in dir_lst:  # [:20]:
        if d not in ("cache", ".trash"):
            fname = folder + os.sep + d + os.sep + r"info\index.json"
            if os.path.isfile(fname):
                f = open(fname, 'r')
                d = json.loads(f.read())
                depends = " ".join([i.split(" ")[0] for i in d["depends"]])
                depends = "".join([i for i in depends])  # if i != 'python'])
                depends = depends.replace('python', '')
                max_len = max(max_len, len(depends))
                out.append([d["name"], d["version"], d["build"], depends])
                f.close()
    r_dt = "<U{}".format(max_len)
    dt = np.dtype([('Package', '<U30'), ('Version', '<U15'), ('Build', '<U15'),
                   ('Requires', r_dt)])
    packages = uts(np.asarray(out), dtype=dt)
    out = []
    names = packages['Package']
    depends = packages['Requires']
    max_len = 0
    for nme in names:
        f = np.char.find(depends, nme.split("-")[0])
        w = np.where(f != -1)[0]
        if np.size(w) > 0:
            v = names[w].tolist()
            v0 = " ".join([i.split(" ")[0] for i in v])
            max_len = max(max_len, len(v0))
            out.append([nme, v0])
        else:
            out.append([nme, "None"])
    r_dt = "<U{}".format(max_len)
    dt = np.dtype([('Package', '<U30'), ('Required_by', r_dt)])
    out = uts(np.asarray(out), dtype=dt)
    return packages, out


def in_file(pth, sub_folder="info", file_name="index.json",
            text="", dependencies=True, output=False):
    r"""Find string in file in folders

    os, json required
    pth = r'C:\Users\dan_p\AppData\Local\ESRI\conda\pkgs'
    sub_folder="info"
    file_name = "index.json"
    text = "python"
    """
    import json
    out = []
    dir_lst = os.listdir(pth)
    print("module, dependencies....")
    for obj in dir_lst:
        key_fld = pth + os.sep + obj
        if sub_folder:
            key_fld += os.sep + sub_folder
        if os.path.exists(key_fld):
            fname = key_fld + os.sep + file_name
            if os.path.isfile(fname):
                f = open(fname, 'r')
                f_string = f.read()
                if text in f_string:
                    dct = json.loads(f_string)
                    if dependencies:
                        depends = dct['depends']
                        out.append([obj, depends])
                    else:
                        out.append(obj)
                f.close()
    if output:
        return out
    for ln in out:
        lines = "\n".join([f"   {i}" for i in ln[1]])
        print("{}\n{}".format(ln[0], lines))


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    # print the script source name.
    testing = True
    print('\n{} in source script... {}'.format(__name__, script))
    # parameters here
else:
    testing = False
    # parameters here
