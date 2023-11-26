# -*- coding: utf-8 -*-  # noqa

r"""
------------
npgCondaPkgs
------------

Script :
    npgCondaPkgs.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2023-11-22

Purpose
-------
Determine conda package information related to arcgis pro.  There are two
variants.

- pkg_info_json(folder=None)
  This uses json files in a specific folder in the arcgis installation path by
  default, or you can specify a different folder containing json files.

- package_info_conda(folder=None)
  The user profile installation path.

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

To get a dictionary of packages/dependencies::

    a = packages['Filename'].tolist()
    b =  packages['Dependencies'].tolist()
    b0 = [i.split("; ") for i in b]
    z ={a : b for a, b in list(zip(a, b))}

To reverse it use `reverse_dict`:
"""

import sys
import os
from textwrap import dedent, wrap
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as uts
from pathlib import Path
import json
#from arcpy.da import NumPyArrayToTable  # noqa

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=300, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def reverse_dict(d, sort_keys=True):
    """Reverse a dictionary's keys and values."""
    d_inv = {}
    for k, v in d.items():
        if not isinstance(v, (list, tuple)):
            v = [v]
        for i in v:
            if i in d_inv:
                d_inv[i].append(k)
            else:
                d_inv[i] = [k]
    if sort_keys:
        return dict(sorted(d_inv.items()))
    return d_inv


def parse_json(f, key="depends"):
    """Parse the json file."""
    with open(f, "r") as f:
        data = json.load(f)
    keys = list(data.keys())
    if key in keys:
        return data[key]
    else:
        return []


def pkg_dependencies(pkg='arcpy-base',
                     skip=['python', 'python >3.6',
                           'python >3.', 'python >=3.9,<3.10.0a0'],
                     folder=None,
                     sort_keys=True):
    r"""Access specific information for the specified `pkg`.

    Parameters
    ----------
    pkg : text
        The package name.
    skip : None, text or list of text
        `None`, doesn't exclude packages.  Use `text` for a single package
        exclusion, otherwise provide a list of package names.
    folder : text
        Path to the folder containing the `json` files.
        SEE `pkg_info_json` for details.
    sort_keys : boolean
        `True`, returns the packages in sorted order, `False` otherwise.

    """
    packages = pkg_info_json(folder=folder, all_info=False)
    chk = packages[packages['Package'] == pkg]
    if packages is None or chk.size == 0:
        msg = "\nPackage json or Folder is incorrect, or doesn't exist."
        print(msg)
        return msg, None
    frst = chk[0]
    r0, r1, r2 = frst
    deps = [i.strip() for i in r2.split(";")]
    if not isinstance(deps, (list, tuple)):
        deps = [deps]
    msg = dedent("""
    name : {}
    file : {}
    omitting : {}
    dependencies :
    """)
    msg = msg.format(r0, r1, skip)
    uniq = []
    others = []
    uni_dict = {}
    for d in deps:
        sub = d.split(" ")[0]
        if skip is not None:
            if sub == skip or sub in skip:
                msg += "\nSkipping ... " + sub
                continue  # -- skip out
        if d == 'None':
            msg += "\n - {} \n{}".format(d, d)
        else:
            nxt = packages[packages['Package'] == sub]
            others.append(nxt[0][0])
            if nxt.size == 0:
                msg += "\n   - None"
            else:
                r0, r1, r2 = nxt[0]
                r3 = [i.strip() for i in r2.split(";")]
                r3 = [i for i in r3 if i not in skip]
                # deps = "\n    ".join(r3)
                deps = wrap(", ".join(r3), 70)
                deps = "\n    ".join([i for i in deps])
                uniq.append(sub)
                uniq.extend(r3)
                uni_dict[sub] = r3
                msg += "\n - {}\n    {}".format(r0, deps)
    uniq2 = [i.split(" ")[0] for i in uniq]
    # uni = np.unique(np.asarray(uniq2))
    uni, cnts = np.unique(np.asarray(uniq2), return_counts=True)
    row_frmt = "  {!s:<15} {:>3.0f}\n"
    out = ""
    for cnt, u in enumerate(uni):
        out += row_frmt.format(u, cnts[cnt])
    # out = ", ".join([i for i in uni if i != frst[0]])
    # out = wrap(out, 70)
    msg += dedent("""\n
    General dependencies :
      Name          Count
      ----          -----
    """)
    msg += out
    if sort_keys:
        srt_dct = dict(sorted(uni_dict.items()))
        return msg, srt_dct, others
    return msg, uni_dict, others


def pkg_info_json(folder=None, keyword='depends', all_info=False):
    r"""Access package info from `*.json` files in a `folder`.

    Parameters
    ----------
    folder : text
        File path to the `json` file. See below.
    keyword : text
        See the options in `Notes`.  `depends` is the default.
    all_info : boolean
        If true, a list of packages, their dependency counts and the modules
        it requires is returned.  If False, just a dependency list is returned.

    The `folder` parameter can be derived from::

        >>> sys.prefix
        ... r"C:\arc_pro\bin\Python\envs\arcgispro-py3"
        ... r"C:\arc_pro\bin\Python\pkgs\cache"
        >>> # ---- conda-meta is appended to yield `folder`, see *`Example`*

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

        # -- folder examples
        install_path = r"C:\arc_pro"
        sys.prefix
        ... 'C:\\arc_pro\\bin\\Python\\envs\\arcgispro-py3'  # or
        ... install_path + r"\bin\Python\envs\arcgispro-py3
        #
        folder = install_path + r"\bin\Python\envs\arcgispro-py3\conda-meta"
        #
        # packages are unpacked to:
        #     r"C:\Users\...You...\AppData\Local\ESRI\conda\pkgs"

        >>> folder = sys.prefix + r"\conda-meta"
        >>> out_ = pkg_info_json(folder, keyword='depends', all_info=True)
        >>> packages, dep_counts, required_by = out_
        # create the output
        >>> f0 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\dep_pkg_info"
        >>> f1 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\dep_counts"
        >>> f2 = r"C:\arcpro_npg\Project_npg\npgeom.gdb\dep_required_by"
        >>> arcpy.da.NumPyArrayToTable(packages, f0)
        >>> arcpy.da.NumPyArrayToTable(dep_counts, f1)
        >>> arcpy.da.NumPyArrayToTable(required_by, f2)

    print required by::
        msg = ""
        for r in required_by:
            pkg = r[0]
            req = r[1]
            msg += "{} : {}\n".format(pkg, req)
        print(msg)
    """
    # -- Checks
    if not folder:
        folder = sys.prefix + "\\conda-meta"
    folder = Path(folder)
    if not folder.is_dir():
        print("\nInvalid path... {}".format(folder))
        return None
    files = list(folder.glob("*.json"))
    if not files:
        print("{} doesn't have any json files".format(folder))
        return None
    #
    # -- Package, Filename, Dependencies
    packages = []
    m0 = m1 = m2 = 0
    for f in files:
        ret = parse_json(f, key=keyword)  # ---- look at dependencies only
        nme = str(f.name).rsplit("-", 2)[0]  # ---- split off the last two
        if len(ret) == 1:
            ret = ret[0]
        elif len(ret) > 1:
            srted = sorted(ret)
            ret = "; ".join([i for i in srted])  # `; ` used
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
    # -- Dependency, Counts
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
    # -- Package, Required_by
    required_by = []
    names = packages['Package']
    depends = packages['Dependencies']
    max_len = 0
    for nme in names:
        if nme == 'python':
            required_by.append([nme, "many"])
            continue  # -- skip out
        w = names[[nme in i for i in depends]]
        if np.size(w) > 0:
            v = w.tolist()
            v0 = ", ".join([i.split(" ")[0] for i in v])
            max_len = max(max_len, len(v0))
            required_by.append([nme, v0])
        else:  # -- no dependencies, hence `None`
            max_len = max(max_len, 4)
            required_by.append([nme, "None"])
    r_dt = "<U{}".format(max_len)
    dt = np.dtype([('Package', '<U30'), ('Required_by', r_dt)])
    required_by = uts(np.asarray(required_by), dtype=dt)
    if all_info:
        return packages, dep_counts, required_by
    return packages


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

    To find `requires` info for a package in the folder::

        >>> pckg_info[pckg_info['Package'] == 'jupyter_client']
        ... array([
        ...    ('jupyter_client', '6.1.12', 'pyhd3eb1b0_0',
        ...     'jupyter_core  -dateutil pyzmq tornado traitlets'),
        ...    ('jupyter_client', '7.3.5', 'py39haa95532_0',
        ...     'entrypoints jupyter_core nest-asyncio  -dateutil pyzmq
        ...      tornado traitlets')],
        ...    dtype=[('Package', '<U30'), ('Version', '<U15'),
        ...           ('Build', '<U15'), ('Requires', '<U605')])

    To find `required_by` info::

        >>> pckg_deps[pckg_deps['Package'] == 'jupyter_client']
        ... array([
        ...    ('jupyter_client',
        ...     'arcgis ipykernel ipykernel jupyter_console jupyter_server
        ...      nbclient notebook qtconsole spyder-kernels spyder-kernels'),
        ...    ('jupyter_client',
        ...     'arcgis ipykernel ipykernel jupyter_console jupyter_server
        ...      nbclient notebook qtconsole spyder-kernels spyder-kernels')],
        ...    dtype=[('Package', '<U30'), ('Required_by', '<U2225')])
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
                depends = depends.replace('python', '').strip()
                max_len = max(max_len, len(depends))
                out.append([d["name"], d["version"], d["build"], depends])
                f.close()
    r_dt = "<U{}".format(max_len)
    dt = np.dtype([('Package', '<U30'), ('Version', '<U15'), ('Build', '<U15'),
                   ('Requires', r_dt)])
    pckg_info = uts(np.asarray(out), dtype=dt)
    out2 = []
    names = pckg_info['Package']
    depends = pckg_info['Requires']
    max_len = 0
    for nme in names:
        f = np.char.find(depends, nme.split("-")[0])
        w = np.where(f != -1)[0]
        if np.size(w) > 0:
            v = names[w].tolist()
            v0 = " ".join([i.split(" ")[0] for i in v])
            max_len = max(max_len, len(v0))
            out2.append([nme, v0])
        else:
            out2.append([nme, "None"])
    r_dt = "<U{}".format(max_len)
    dt = np.dtype([('Package', '<U30'), ('Required_by', r_dt)])
    pckg_deps = uts(np.asarray(out2), dtype=dt)
    return pckg_info, pckg_deps


def in_file(pth, text="arcpy", sub_folder="info", file_name="index.json",
            dependencies=True, output=False):
    r"""Find string in file in folders.

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


# ---- special
#
def who_needs(this_pkg, look_in=None):
    """Return the packages that depend on `pkg`.

    Parameters
    ----------
    this_pkg : text
        The package name. e.g. `python`.
    look_in : folder
        The folder that contains the installation json files that describe
        dependent packages.

    Returns
    -------
    A list of packages that depend upon `pkg'.
    """
    returned = pkg_info_json(folder=look_in,
                             keyword='depends',
                             all_info=True)
    if returned is not None:
        packages, dep_counts, required_by = returned  # -- from pkg_info_json
    vals = required_by[required_by['Package'] == this_pkg]
    pkg_nme = vals['Package'][0]
    used_by = vals['Required_by'][0]
    return pkg_nme, used_by


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
