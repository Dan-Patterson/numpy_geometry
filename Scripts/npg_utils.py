# -*- coding: utf-8 -*-
r"""

npg_utils
---------

Script :
    utils.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2020-01-11

Purpose
-------
Tools for working with numpy arrays.  From arraytools.utils

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.

`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.


Useage
------

**doc_func(func=None)** : see get_func and get_modu

**get_func** :

Retrieve function information::

    get_func(func, line_nums=True, verbose=True)
    print(art.get_func(art.main))

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
"""

# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
from textwrap import dedent, indent, wrap
import warnings
import numpy as np

warnings.simplefilter('ignore', FutureWarning)

# from arcpytools import fc_info, tweet  #, frmt_rec, _col_format
# import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
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
        t_0 = time.perf_counter()        # start time
        result = func(*args, **kwargs)   # ... run the function ...
        t_1 = time.perf_counter()        # end time
        dt = t_1 - t_0
        print("\nTiming function for... {}".format(func.__name__))
        if result is None:
            result = 0
        print("  Time: {: <8.2e}s for {:,} objects".format(dt, result))
        return result                   # return the result of the function
        # return dt                       # return delta time
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
            """Return sub in dummy"""
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
    code = "".join(["{:4d}  {}".format(idx+line_num, line)
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


# ----------------------------------------------------------------------
# ---- (2) get_func .... code section ----
def get_func(func, line_nums=True, verbose=True):
    """Get function information (ie. for a def)

    Parameters
    ----------
    >>> from textwrap import dedent, indent, wrap
    >>> import inspect

    Returns
    -------
    The function information includes arguments and source code.
    A string is returned for printing.

    Notes
    -----
    Import the module containing the function and put the object name in
    without quotes...

    >>> from arraytools.utils import get_func
    >>> get_func(get_func)  # returns this source code etc.
    """
    frmt = """\
    -----------------------------------------------------------------
    File path: ... {}
    Function: .... {} ....
    Signature .... In [1]: {}{}
    Line number... {}
    Defaults:  ... {}
    kwdefaults: .. {}
    Variable names:
    {}\n
    Source code:
    {}
    -----------------------------------------------------------------
    """
    import inspect  # required if not imported at the top
    import dis
    # from textwrap import dedent, wrap

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
    if verbose:
        print(code_mem)
    else:
        return code_mem


# ----------------------------------------------------------------------
# ---- (3) get_module info .... code section ----
def get_module_info(obj, code=False, verbose=True):
    """Get module (script) information, including source code if needed.

    Parameters
    ----------
    obj : module (script)
        The imported object.  It must be either a whole module or a script
        that you imported.  Import it. It is easier if it is in the same
        folder as the script running this function.
    code, verbose : boolean
        Whether to return the code as well and/or return a string output.

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
    ln = "\n:{}:\n\n".format("-"*65)
    frmt = "{}Module: {}\nFile:   {}\nMembers:\n{}\n\nDocs:...\n{}"
    frmt0 = "  {}"
    frmt1 = "{}Module: {}\nFile:   {}\nMembers:\n"
    frmt2 = "\nSource code: .....\n{}Docs:...\n{}"
    import inspect
    # from textwrap import dedent  # required if not immported initially

    if not inspect.ismodule(obj):
        out = "\nError... `{}` is not a module, but is of type... {}\n"
        print(out.format(obj.__name__, type(obj)))
        return None
    if code:
        lines, _ = inspect.getsourcelines(obj)
        frmt = frmt0 + frmt2
        code = "".join(["{:4d}  {}".format(idx + 1, line)
                        for idx, line in enumerate(lines)])
    else:
        lines = code = ""
        frmt = frmt + frmt1
    memb = [i[0] for i in inspect.getmembers(obj) if i[0][:2] != "__"]
    memb.sort()
    memb = ", ".join([i for i in memb])
    w = wrap(memb)
    w0 = "\n".join([i for i in w])
    args0 = [ln, obj.__name__, obj.__file__]
    args1 = [obj.__doc__, code, ln[1:]]
    p0 = frmt1.format(*args0)
    p1 = indent(w0, "  ")
    p2 = frmt2.format(*args1)
    if verbose:
        print("{}\n{}\n{}".format(p0, p1, p2))
    else:
        return "{}\n{}\n{}".format(p0, p1, p2)


def find_def(defs, module_name):
    """
    find_def
    Find occurences of a function in a module

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
        print("\n{}\n".format("="*20, i))
        np.lookfor(i, module_name)


# ----------------------------------------------------------------------
# ---- (4) wrapper .... code section ----
def _wrapper(a, wdth=70):
    """Wrap stuff using textwrap.wrap

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
    return txt


def _utils_help_():
    """arraytools.utils help...

    Function list follows:
    """
    _hf = """
    :-------------------------------------------------------------------:
    : ---- arrtools functions  (loaded as 'art') ----
    : ---- from utils.py
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
