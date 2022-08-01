# -*- coding: utf-8 -*-
# noqa: D205, D400, F403
r"""
------------------------------------------
  clp: segment intersection and clipping
------------------------------------------

Stuff

----

"""
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import repack_fields

from npg.npg_plots import plot_polygons

np.set_printoptions(
    edgeitems=10, linewidth=120, precision=2, suppress=True, threshold=200,
    formatter={"bool": lambda x: repr(x.astype(np.int32)),
               "float_kind": '{: 6.2f}'.format})


# ---- Polygon class, properties and methods
#
class pgon(np.ndarray):
    """
    Poly class for clipping.

    Requires
    --------
    numpy array.
    """

    def __new__(cls,
                arr=None,
                CFT=None,
                Kind=None,
                Info="clipper",
                Extent=None,
                SR=None
                ):
        # --
        """See `Docs` for construction notes."""
        arr = np.ascontiguousarray(arr)
        CFT = np.ascontiguousarray(CFT)
        if (arr.ndim != 2) or (CFT.ndim != 2):
            m = "Input error... arr.ndim != 2 : {} or IFT.dim != 2 : {}"
            print(m.format(arr.ndim, CFT.ndim))
            return None
        if (CFT.shape[-1] < 6) or (Kind not in (1, 2)):
            print("didn't work")
            return None
        # --
        self = arr.view(cls)      # view as Geo class
        self.CFT = CFT            # array id, fr-to, cw, part id
        self.K = Kind             # Points (0), Polylines (1), Polygons (2)
        self.Info = Info          # any useful information
        self.Fr = CFT[:, 0]       # from point id
        self.To = CFT[:, 1]       # to point id
        # self.Bf = CFT[:, 2]       # previous pnt id
        self.eqX = CFT[:, 2]      # pnt equal intersection
        self.eqO = CFT[:, 3]      # pnt equals other
        self.inO = CFT[:, 4]     # pnt in other
        self.CT = CFT[:, 5]       # crossing type
        return self

    def __array_finalize__(self, src_arr):
        """
        Finalize new object....

        See npgGeo for more details
        """
        if src_arr is None:
            return
        self.CFT = getattr(src_arr, 'CFT', None)
        self.K = getattr(src_arr, 'K', None)
        self.Info = getattr(src_arr, 'Info', None)
        self.Fr = getattr(src_arr, 'Fr', None)
        self.To = getattr(src_arr, 'To', None)
        # self.Bf = getattr(src_arr, 'Bf', None)
        self.eqX = getattr(src_arr, 'eqX', None)
        self.eqO = getattr(src_arr, 'eqO', None)
        self.inO = getattr(src_arr, 'inO', None)
        self.CT = getattr(src_arr, 'CT', None)

    def __array_wrap__(self, out_arr, context=None):
        """Wrap it up."""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # @property

    @property
    def CFT_str(self):
        """Clip poly structure.  See self.structure for more information."""
        nmes = ["Fr_pnt", "To_pnt", "Prev_pnt", "CeX", "CeP", "CinP"]
        return uts(self.CFT, names=nmes, align=False)

    def update(self, eqX, eqOther, inOther):
        """Update a column in `CeX`, `CeP`, `CinP`.

        Parameters
        ----------
        eqX, eqOther, inOther : ndarrays or lists of values
          - poly/clipper equal to an intersection point
          - one equals the other point
          - one is in the other
            CFT column names (3, 4, 5 positionally)

        Notes
        -----
        Conversion values are based on binary conversion as shown in the
        `keys` line, then reclassed using a dictionary conversion.

        - keys = eqX * 100 + eqOther * 10 + inOther
        - 0 is outside
        - 1 is inside with no intersection
        -   position before a 1 is an intersection point not at an endpoint
        - 5 endpoint intersects a segment
        - 111 (7) clp, poly and intersection meet at a point
        """
        k = [0, 1, 10, 11, 100, 101, 110, 111]
        v = [0, 1, 2, 3, 4, 5, 6, 7]
        d = dict(zip(k, v))
        self.CFT[:, 2][eqX] = 1
        self.CFT[:, 3][eqOther] = 1
        self.CFT[:, 4][inOther] = 1
        keys = self.CFT[:, 2] * 100 + self.CFT[:, 3] * 10 + self.CFT[:, 4]
        vals = [d[i] for i in keys.tolist()]
        self.CFT[:, 5] = vals
        return


# ----
def array_CFT(in_arrays, shift_to_origin=False):
    """Produce the Geo array.  Construction information in `npgDocs`.

    Parameters
    ----------
    in_arrays : list of ndarrays
    shift_to_origin : boolean
        True, moves the geometry to the origin.  False, uses the existing
        coordinates.
    """
    out_arrays = []
    info_ = ["target", "clipper"]
    # -- create the polygons
    #
    for cnt, arr in enumerate(in_arrays):
        ids = np.arange(len(arr) + 1)
        f = ids[:-1, None]
        t = np.concatenate((ids[1:-1], [ids[0]]))[:, None]
        fr_to = np.concatenate((f, t), axis=1)
        # prev_ = np.concatenate((ids[[-2]], ids[:-2]))[:, None]
        CFT = np.concatenate(
            (fr_to, np.full((arr.shape[0], 4), 0)), axis=1)  # prev_,
        extent = np.array([np.min(arr, axis=0), np.max(arr, axis=0)])
        if shift_to_origin:
            arr = arr - extent[0]
        _type = info_[cnt]
        p = pgon(arr, CFT, 2, _type)
        out_arrays.append([p, CFT, extent])
    out_ = out_arrays
    if len(out_arrays) == 1:
        out_ = out_arrays[0]
    return out_
