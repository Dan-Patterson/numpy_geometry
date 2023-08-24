# -*- coding: utf-8 -*-
# noqa: D205, D400
r"""

npg_utils
---------

Script :
    utils.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2022-07-26

Purpose
-------
Tools for working with numpy arrays.  Originally from arraytools.utils.

Useage
------

**doc_func(func=None)** : see get_func and get_modu

**get_func** :

Retrieve function information::

    get_func(func, line_nums=True, verbose=True)
    print(npg.get_func(art.main))

    Function: .... main ....
    Line number... 1334
    Docs:
    Do nothing
    Defaults: None
    Keyword Defaults: None
    Variable names:
    Source code:
       0  def main():
       1   '''Do nothing'''
       2      pass

get_modu :
    retrieve module info

**info** :

Retrieve array information::

    - array([(0, 1, 2, 3, 4), (5, 6, 7, 8, 9),
             (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)],
      dtype=[('A', '<i8'), ('B', '<i8')... snip ..., ('E', '<i8')])
    ---------------------
    Array information....
    array
      |__shape (4,)
      |__ndim  1
      |__size  4
      |__type  <class 'numpy.ndarray'>
    dtype      [('A', '<i8'), ('B', '<i8') ... , ('E', '<i8')]
      |__kind  V
      |__char  V
      |__num   20
      |__type  <class 'numpy.void'>
      |__name  void320
      |__shape ()
      |__description
         |__name, itemsize
         |__['A', '<i8']
         |__['B', '<i8']
         |__['C', '<i8']
         |__['D', '<i8']
         |__['E', '<i8']

New random number generator protocol

r = np.random.default_rng()

dir(r)
[ ..., '_poisson_lam_max', 'beta', 'binomial', 'bit_generator', 'bytes',
 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric',
 'gumbel', 'hypergeometric', 'integers', 'laplace', 'logistic', 'lognormal',
 'logseries', 'multinomial', 'multivariate_hypergeometric',
 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare',
 'noncentral_f', 'normal', 'pareto', 'permutation', 'poisson', 'power',
 'random', 'rayleigh', 'shuffle', 'standard_cauchy', 'standard_exponential',
 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform',
 'vonmises', 'wald', 'weibull', 'zipf']
"""

# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import os
from pathlib import Path
from textwrap import dedent, indent, wrap
# import warnings
import numpy as np
# from numpy.lib.recfunctions import unstructured_to_structured as uts

# warnings.simplefilter('ignore', FutureWarning)

np.set_printoptions(
    edgeitems=10, linewidth=120, precision=3, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 7.3f}'.format})
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = [
    'time_deco', 'run_deco', 'doc_func', 'get_func', 'get_module_info',
    'find_def', '_wrapper', '_utils_help_'
]


# ---- decorators and helpers ------------------------------------------------
#
def time_deco(func):  # timing originally
    """Use as a timing decorator function.

    Parameters
    ----------
    The following import.  Uncomment the import or move it inside the script.

    >>> from functools import wraps

    Example function::

        @time_deco  # on the line above the function
        def some_func():
            ``do stuff``
            return None

    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t_0 = time.perf_counter() * 1000  # start time
        result = func(*args, **kwargs)    # ... run the function ...
        t_1 = time.perf_counter() * 1000  # end time
        dt = t_1 - t_0
        print("\nTiming function for... {}".format(func.__name__))
        if result is None:
            result = 0
        print("  Time: {: <8.3f}ms for {:,} objects".format(dt, len(result)))
        return result                    # return the result of the function
        # return dt                      # return delta time
    return wrapper


def run_deco(func):
    """Print basic function information and the results of a run.

    Parameters
    ----------
    The following import.  Uncomment the import or move it inside the script.

    >>> from functools import wraps

    Example function::

        @run_deco  # on the line above the function
        def some_func():
            ``do stuff``
            return None
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrap the function."""
        frmt = "\n".join(["Function... {}", "  args.... {}",
                          "  kwargs.. {}", "  docs.... {}"])
        ar = [func.__name__, args, kwargs, func.__doc__]
        print(dedent(frmt).format(*ar))
        result = func(*args, **kwargs)
        print("{!r:}\n".format(result))  # comment out if results not needed
        return result                    # for optional use outside.
    return wrapper


# ----------------------------------------------------------------------------
# ---- (1) doc_func ... code section ... ----
def doc_func(func=None, verbose=True):
    """(doc_func)...Documenting code using `inspect`.

    Parameters
    ----------
    func : function
        Function/method name to document, without quotes.
    verbose : Boolean
        True prints the result, False returns a string of the result.

    Returns
    -------
    A listing of the source code with line numbers.

    Notes
    -----
    Requires the `inspect` module.  Source code for::

        module level
        - inspect.getsourcelines(sys.modules[__name__])[0]

        function level
        - as a list => inspect.getsourcelines(num_41)[0]
        - as a string => inspect.getsource(num_41)

        file level
        - script = sys.argv[0]
    """
    def demo_func():
        """Demonstrate retrieving and documenting module and function info."""
        def sub():
            """Return sub in dummy."""
            print("sub")
        return None
    #
    import inspect
    if func is None:
        func = demo_func
    if not inspect.isfunction(func):
        out = "\nError... `{}` is not a function, but is of type... {}\n"
        print(out.format(func.__name__, type(func)))
        return None
    script2 = sys.argv[0]  # a useful way to get a file's name
    lines, line_num = inspect.getsourcelines(func)
    code = "".join(["{:4d}  {}".format(idx + line_num, line)
                    for idx, line in enumerate(lines)])
    nmes = ['args', 'varargs', 'varkw', 'defaults', 'kwonlyargs',
            'kwonlydefaults', 'annotations']
    f = inspect.getfullargspec(func)
    f_args = "\n".join([str(i) for i in list(zip(nmes, list(f)))])
    args = [line_num, code,
            inspect.getcomments(func),
            inspect.isfunction(func),
            inspect.ismethod(func),
            inspect.getmodulename(script2),
            f_args]
    frmt = """
    :----------------------------------------------------------------------
    :---- doc_func(func) ----
    :Code for a function on line...{}...
    :
    {}
    Comments preceeding function
    {}
    function?... {} ... or method? {}
    Module name... {}
    Full specs....
    {}
    ----------------------------------------------------------------------
    """
    out = (dedent(frmt)).format(*args)
    if verbose:
        print(out)
    else:
        return out


# -----------------------------------------------------------------------
# ---- (2) get_func .... code section ----
def get_func(func, line_nums=True, output=False):
    """Get function information (ie. for a def).

    Parameters
    ----------
    func : function/def
        The object to document.
    line_nums : boolean
        True, provides line numbers.
    output : boolean
        True, returns the formatted report.  False, prints it.

    Required
    --------
    >>> from textwrap import dedent, indent, wrap
    >>> import inspect

    Returns
    -------
    The function information includes arguments and source code.
    A string is optionally returned for deferred printing.

    Notes
    -----
    Import the module containing the function and put the object name in
    without quotes...

    >>> import npgeom as npg
    >>> npg.get_func(get_func)  # returns this source code etc.
    """
    frmt = r"""
    -----------------------------------------------------------------
    File path: ... {}
    Function: .... {} ....
    Signature .... In [1]: {}{}
    Line number... {}
    Defaults:  ... {}
    kwdefaults: .. {}
    Variable names:
    {}

    Source code:
    {}
    -----------------------------------------------------------------
    """
    import inspect  # required if not imported at the top
    # import dis
    from textwrap import dedent, wrap

    if not inspect.isfunction(func):
        out = "\nError... `{}` is not a function, but is of type... {}\n"
        print(out.format(func.__name__, type(func)))
        return None

    lines, ln_num = inspect.getsourcelines(func)
    sig = str(inspect.signature(func))
    co_file = inspect.getfile(func)
    # co_obj = dis._get_code_object(func)  # these two are the same
    # co_obj = func.__code__
    if line_nums:
        code = "".join(["{:4d}  {}".format(idx + ln_num, line)
                        for idx, line in enumerate(lines)])
    else:
        code = "".join(["{}".format(line) for line in lines])

    vars_ = ", ".join([i for i in func.__code__.co_varnames])
    vars_ = wrap(vars_, 50)
    vars_ = "\n".join([i for i in vars_])
    args = [co_file,
            func.__name__, func.__name__, sig,
            ln_num,
            func.__defaults__,
            func.__kwdefaults__, indent(vars_, "    "), code]
    code_mem = dedent(frmt).format(*args)
    if output:
        return code_mem
    print(code_mem)


# ----------------------------------------------------------------------
# ---- (3) get_module info .... code section ----
def get_module_info(obj, max_number=100, verbose=True):
    """Get module (script) information, including source code if needed.

    Parameters
    ----------
    obj : module (script)
        The imported object.  It must be either a whole module or a script
        that you imported.  Import it. It is easier if it is in the same
        folder as the script running this function.
    max_number : integer
        The maximum number of methods to list if there are a large number.
    verbose : boolean
        True, prints the output.  False, returns the string.

    Requires
    --------
      >>> from textwrap import dedent, indent
      >>> import inspect

    Returns
    -------
    A string is returned for printing.  It will be the whole module
    so use with caution.

    Example
    -------
    Importing this function from the following module to inspect the module
    itself.

    >>> from npgeom.npg_utils import get_module_info
    >>> get_module_info(npg, False, True)
    >>> # No quotes around module name, code=True for module code
    """
    def wrapit(_in, counter=None):
        """Wrap a string."""
        nmes = _in[1]
        if len(nmes) > max_number:
            nmes = nmes[:max_number] + [".... snip ...."]
        sub = ", ".join([j for j in nmes])
        sub = wrap(sub, 75)
        txt = "\n".join([j for j in sub])
        if counter is None:
            return "\n{}\n{}".format(i[0], indent(txt, "    "))
        s = "\n({:2.0f}) : {}\n{}".format(counter, i[0], indent(txt, "    "))
        return s
    # ----
    ln = "\n:{}:\n\n".format("-" * 65)
    f1 = "{}Package: {}\nFile:    {}"
    import inspect
    from textwrap import indent, wrap
    if not inspect.ismodule(obj):
        out = "\nError... `{}` is not a module, but is of type... {}\n"
        print(out.format(obj.__name__, type(obj)))
        return None
    # ----
    path_parts = obj.__file__.split("\\")[:-1]
    mod_path = "\\".join([i for i in path_parts])
    out = []
    mem = inspect.getmembers(obj, inspect.ismodule and not inspect.isbuiltin)
    func = inspect.getmembers(obj, inspect.isroutine)  # isfunction)
    clas_ = inspect.getmembers(obj, inspect.isclass)
    _a0 = sorted([i[0] for i in func if i[0].startswith("_")])  # dir(m[1])
    _a1 = sorted([i[0] for i in func if not i[0].startswith("_")])  # dir(m[1])
    _all = _a0 + _a1
    _c0 = sorted([i[0] for i in clas_])
    out.append(["Functions/methods:", _all])
    out.append(["Classes :", _c0])
    out.append(["\nMembers:", ""])
    for m in mem:
        if hasattr(m[1], "__file__"):
            if mod_path in m[1].__file__:
                r = inspect.getmembers(m[1], inspect.isroutine)
                _a = [i[0] for i in r]
                _a0 = sorted([i for i in _a if i.startswith("_")])
                _a1 = sorted([i for i in _a if not i.startswith("_")])
                _all = _a0 + _a1
                out.append([m[0], _all])
            else:
                out.append([m[0], ["package: {}".format(m[1].__package__)]])
    # ----
    s = ""
    for i in out[:3]:
        s += wrapit(i, counter=None)
    cnt = 1
    for i in out[3:]:
        s += wrapit(i, counter=cnt)
        cnt += 1
    args0 = [ln, obj.__name__, obj.__file__]
    p0 = f1.format(*args0)
    if verbose:
        print("{}\n{}".format(p0, s))
        # print(s)
    else:
        return "{}\n{}\n".format(p0, s)


def find_def(defs, module_name):
    """Find occurences of a function in a module.

    np.lookfor(what, module=None, import_modules=True, regenerate=False,
            output=None):

    Parameters
    ----------
    defs : text singleton or list
        The name of a def or a list of them.  These are what is being searched.
    module_name : name, no quotes
        The name of the module that was imported.
    """
    if not isinstance(defs, (list, tuple)):
        defs = [defs]
    for i in defs:
        print("\n{}\n{}".format("=" * 20, i))
        np.lookfor(i, module_name)


# ----------------------------------------------------------------------
# ---- (4) wrapper .... code section ----
def _wrapper(a, wdth=70, verbose=True):
    """Wrap stuff using ``textwrap.wrap``.

    Notes
    -----
    TextWrapper class
    __init__(self, width=70, initial_indent='', subsequent_indent='',
             expand_tabs=True, replace_whitespace=True,
             fix_sentence_endings=False, break_long_words=True,
             drop_whitespace=True, break_on_hyphens=True, tabsize=8,
             *, max_lines=None, placeholder=' [...]')
    """
    if isinstance(a, np.ndarray):
        txt = [str(i) for i in a.tolist()]
        txt = ", ".join(txt)
    elif isinstance(a, (list, tuple)):
        txt = ", ".join([str(i) for i in a])
    txt = "\n".join(wrap(txt, width=wdth))
    if verbose:
        print(txt)
    else:
        return txt


def _utils_help_():
    """arraytools.utils help...

    Function list follows:
    """
    _hf = """
    :-------------------------------------------------------------------:
    : ---- npg functions  (loaded as 'npg') ----
    : ---- from npg_utils.py
    (1)  doc_func(func=None)
         documenting code using inspect
    (2)  get_func(obj, line_nums=True, verbose=True)
         pull in function code
    (3)  get_module(obj)
         pull in module code
    (4)  dirr(a)  object info
    (5)  wrapper(a)  format objects as a string
    :-------------------------------------------------------------------:
    """
    print(dedent(_hf))


def doc_deco(func, doc):
    """Print basic function information and the results of a run.

    Parameters
    ----------
    The following import.  Uncomment the import or move it inside the script.

    >>> from functools import wraps

    Example function::

        @run_deco  # on the line above the function
        def some_func():
            ``do stuff``
            return None

    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrap function."""
        frmt = "\n".join(["Function... {}", "  args.... {}",
                          "  kwargs.. {}", "  docs.... {}",
                          "  extra.... {}"])
        ar = [func.__name__, args, kwargs, func.__doc__, doc]
        print(dedent(frmt).format(*ar))
        result = func(*args, **kwargs)
        print("{!r:}\n".format(result))  # comment out if results not needed
        return result                    # for optional use outside.
    return wrapper


# ---- (5) folder tree -------------------------------------------------------
# `folders` requires `get_dirs`
#
def get_dirs(pth, ignore=['__pycache__']):
    """Get the directory list from a path, excluding geodatabase folders.

    Parameters
    ----------
    pth : string
        The path string being examined.
    ignore : list
        Folders to ignore.

    Returns
    -------
    The folders in the specified `pth`.

    Notes
    -----
    Used by.. folders.

    >>> get_dirs('C:/arcpro_npg/npg')
    ['C:/arcpro_npg/npg/.pylint.d', 'C:/arcpro_npg/npg/data',
     'C:/arcpro_npg/npg/docs', 'C:/arcpro_npg/npg/Extra_scripts',
     'C:/arcpro_npg/npg/images', 'C:/arcpro_npg/npg/npg_2022',
     'C:/arcpro_npg/npg/Project_npg', 'C:/arcpro_npg/npg/tests']

    References
    ----------
    Correspondence to tools in the os module:
        `<https://docs.python.org/3/library/pathlib.html>`_.

    """
    pth = Path(pth)
    if pth.is_file():
        pth = pth.parent  # os.path.dirname(pth)
    p = pth.resolve()  # os.path.normpath(pth)
    dirs = [x.as_posix() for x in p.iterdir()
            if x.is_dir() and x.stem not in ignore]
    return dirs


def folders(path, first=True, initial=0, prefix="",
            ignore=['__init__', '__pycache__'],
            max_num=20
            ):
    r"""Print recursive listing of folders in a path.

    Parameters
    ----------
    path : string
        The folder path to examine.  Make sure you `raw` format the path::

    Requires
    --------
    ``_get_dir`` See its docstring for an example of path common prefix.
    """
    if first:  # Detect outermost call, print a heading
        print("-" * 30 + "\n|.... Path listing for ....|\n|--{}".format(path))
        print("\n... content and sub folders ...")
        prefix = "|-"
        first = False
        initial = len(path)
        cprev = path
    dirlist = get_dirs(path, ignore)
    for d in dirlist:
        fullname = os.path.join(path, d)  # Turn name into full pathname
        if os.path.isdir(fullname):       # If a directory, recurse.
            cprev = path
            pad = ' ' * (len(cprev) - initial)
            # pad = ' ' * 4
            n = d.replace(cprev, pad)
            print(prefix + "-" + n)  # fullname) # os.path.relpath(fullname))
            p = "  "
            folders(fullname, first=False,
                    initial=initial, prefix=p,
                    ignore=ignore, max_num=max_num)
    # ---


def sub_folders(path, combine=False):
    """Print the folders in a path, excluding '.' folders."""
    import pathlib
    print("Path...\n{}".format(path))
    if combine:
        r = " " * len(path)
    else:
        r = ""
    f = "\n".join([(p._str).replace(path, r)
                   for p in pathlib.Path(path).iterdir()
                   if p.is_dir() and "." not in p._str])
    print("{}".format(f))


def env_list(pth, ordered=False):
    """List folders, files in a path, as an array.  Requires ``os`` module."""
    d = []
    for item in os.listdir(pth):
        check = os.path.join(pth, item)
        check = check.replace("\\", "/")
        if os.path.isdir(check) and ("." not in check):
            d.append(check)
    d = np.array(d)
    if ordered:
        d = d[np.argsort(d)]
    return d


# ---- (6) dirr ... code section ... -----------------------------------------
#
def dir_py(obj, colwise=False, cols=3, prn=True):
    """Return the non-numpy version of dirr."""
    from itertools import zip_longest as zl
    a = dir(obj)
    w = max([len(i) for i in a])
    frmt = (("{{!s:<{}}} ".format(w))) * cols
    csze = len(a) / cols  # split it
    csze = int(csze) + (csze % 1 > 0)
    if colwise:
        a_0 = [a[i: i + csze] for i in range(0, len(a), csze)]
        a_0 = list(zl(*a_0, fillvalue=""))
    else:
        a_0 = [a[i: i + cols] for i in range(0, len(a), cols)]
    if hasattr(obj, '__name__'):
        args = ["-" * 70, obj.__name__]
    else:
        args = ["-" * 70, type(obj)]
    txt_out = "\n{}\n| dir_py({}) ...\n|\n-------".format(*args)
    cnt = 0
    for i in a_0:
        cnt += 1
        txt = "\n  ({:>03.0f})  ".format(cnt)
        frmt = (("{{!s:<{}}} ".format(w))) * len(i)
        txt += frmt.format(*i)
        txt_out += txt
    if prn:
        print(txt_out)
    else:
        return txt_out


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
