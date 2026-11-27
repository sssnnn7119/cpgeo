"""Clamped bicubic patches fitted to a spherical CPGEO domain."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def axial_chart(knots, thresholds):
    """Locate two dense knot clusters for an elongated spherical parameterization."""
    moment = knots.T @ knots / len(knots)
    values, axes = np.linalg.eigh(moment)
    # An isotropic parameter sphere has second moments near 1/3.
    if values[-1] < 0.68:
        return None
    direction = knots @ axes[:, -1]
    poles, radii = [], []
    for mask in (direction >= 0, direction < 0):
        if not np.any(mask):
            return None
        scales = thresholds[mask]
        smallest = scales.min()
        if smallest <= 0 or np.quantile(scales, 0.9) < 16 * smallest:
            return None
        # Inverse support area emphasizes each cap's most compressed region.
        center = (smallest / scales)**2 @ knots[mask]
        poles.append(center / np.linalg.norm(center))
        radii.append(min(0.25, 2 * smallest))
    north, south = poles
    cosine = np.dot(north, south)
    if cosine >= 0:
        return None
    tangent = south - cosine * north
    if np.linalg.norm(tangent) < 1e-12:
        tangent = np.eye(3)[np.argmin(np.abs(north))]
        tangent -= np.dot(tangent, north) * north
    tangent /= np.linalg.norm(tangent)
    inv_pole = np.dot(south, tangent) / (1 - cosine)
    # One side tile covers about a four-fold change in stereographic radius.
    bands = int(np.clip(np.ceil(-sum(np.log(radii)) / np.log(4)), 1, 32))
    return np.r_[north, tangent, np.cross(north, tangent), inv_pole, radii], bands


@dataclass
class BSplinePatches:
    """Poles in (patch, u, v, xyz) order on [0,1]².

    Root face IDs follow (+X,-X,+Y,-Y,+Z,-Z) in chart coordinates.
    Adaptive results store individual pole grids, knot breaks and rectangles
    within these root faces. A legacy banded chart orders four sides per band
    from south to north followed by the north and south caps.
    Each axis has simple interior knots and end multiplicity four.
    Errors are measured at validation samples in model units. The seam-normal
    angle is in degrees. Metrics describe the poles at fitting time.
    """

    poles: np.ndarray | tuple[np.ndarray, ...]
    max_error: float
    rms_error: float
    max_seam_angle_deg: float
    # Unique, normalized knot breaks for each adaptive patch. End multiplicity is four.
    breaks: list | None = None
    domains: list | None = None  # (root face, [[u0,u1],[v0,v1]], assigned knot count)
    frame: np.ndarray | None = None

    def _breaks(self, index):
        if self.breaks is not None:
            return self.breaks[index]
        return tuple(np.linspace(0, 1, n-2) for n in self.poles[index].shape[:2])

    def to_bsp(self):
        """Return the bspmap.BSP surfaces using the optional bspmap package."""
        from bspmap import BSP
        from bspmap.basis import BasisClamped

        from bspmap import capi as bsp_capi

        surfaces = []
        for i, patch in enumerate(self.poles):
            bases = []
            for n, t in zip(patch.shape[:2], self._breaks(i)):
                basis = BasisClamped(n, 3)
                knots = np.r_[[0.]*3, t, [1.]*3]
                if not np.array_equal(knots, basis.knots):
                    basis.knots = knots
                    basis.basis = bsp_capi.deboor_cox(knots, n, 3)
                    basis.basis_d1 = bsp_capi.basis_derivative(basis.basis)
                    basis.basis_d2 = bsp_capi.basis_derivative(basis.basis_d1)
                bases.append(basis)
            surfaces.append(BSP(bases, 3, patch.shape[:2], patch.reshape(-1, 3).copy()))
        return surfaces

    def write_step(self, path):
        """Sew a single outward solid, write STEP in mm and check the imported topology.

        Requires the optional cadquery-ocp Python package. Coordinates are
        interpreted in millimetres. Replace the destination only after validation.
        """
        from OCP.BOPAlgo import BOPAlgo_ArgumentAnalyzer
        from OCP.BRep import BRep_Tool
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_Sewing,
        )
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepLib import BRepLib
        from OCP.Geom import Geom_BSplineSurface
        from OCP.gp import gp_Pnt
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import (
            STEPControl_ManifoldSolidBrep,
            STEPControl_Reader,
            STEPControl_Writer,
        )

        try:
            from OCP.collections import Array2_gp_Pnt, Array1_double, Array1_int
        except ImportError:  # OCCT 7.x bindings
            from OCP.TColgp import TColgp_Array2OfPnt as Array2_gp_Pnt
            from OCP.TColStd import (
                TColStd_Array1OfReal as Array1_double,
                TColStd_Array1OfInteger as Array1_int,
            )
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        sewing = BRepBuilderAPI_Sewing(1e-7)
        for index, patch in enumerate(self.poles):
            vectors, multiplicities = [], []
            for t in self._breaks(index):
                knots = Array1_double(1, len(t))
                mults = Array1_int(1, len(t))
                for i, value in enumerate(t, 1):
                    knots.SetValue(i, value)
                    mults.SetValue(i, 4 if i in (1, len(t)) else 1)
                vectors.append(knots)
                multiplicities.append(mults)
            points = Array2_gp_Pnt(1, patch.shape[0], 1, patch.shape[1])
            for i, row in enumerate(patch, 1):
                for j, xyz in enumerate(row, 1):
                    points.SetValue(i, j, gp_Pnt(*xyz))
            surface = Geom_BSplineSurface(points, *vectors, *multiplicities, 3, 3)
            face = BRepBuilderAPI_MakeFace(surface, 1e-7)
            if not face.IsDone():
                raise ValueError("Could not build a B-spline face")
            sewing.Add(face.Face())
        sewing.Perform()
        shell = sewing.SewedShape()
        if (
            shell.IsNull()
            or shell.ShapeType() != TopAbs_SHELL
            or sewing.NbMultipleEdges()
            or sewing.NbFreeEdges()
            or not BRep_Tool.IsClosed_s(shell)
        ):
            raise ValueError("B-spline patches did not form a single closed shell")
        as_shell = getattr(TopoDS, "Shell_s", None) or TopoDS.Shell
        solid = BRepBuilderAPI_MakeSolid(as_shell(shell)).Solid()
        if (
            not BRepLib.OrientClosedSolid_s(solid)
            or not BRepCheck_Analyzer(solid).IsValid()
        ):
            raise ValueError("Invalid B-spline solid")

        path = Path(path)
        with TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / "surface.stp"
            writer = STEPControl_Writer()
            writer.Model().SetLocalLengthUnit(1.0)
            writer.Model().SetWriteLengthUnit(1.0)
            if (
                writer.Transfer(solid, STEPControl_ManifoldSolidBrep)
                != IFSelect_RetDone
                or writer.Write(str(temporary)) != IFSelect_RetDone
            ):
                raise OSError("STEP export failed")
            reader = STEPControl_Reader()
            if reader.ReadFile(str(temporary)) != IFSelect_RetDone:
                raise ValueError("STEP round-trip failed to read the file")
            reader.SetSystemLengthUnit(1.0)
            if reader.TransferRoots() != 1:
                raise ValueError("STEP round-trip failed")
            imported = reader.OneShape()
            if (
                imported.ShapeType() != TopAbs_SOLID
                or not BRepCheck_Analyzer(imported).IsValid()
            ):
                raise ValueError("STEP round-trip did not return a valid solid")
            shells = TopExp_Explorer(imported, TopAbs_SHELL)
            if not shells.More() or not BRep_Tool.IsClosed_s(shells.Current()):
                raise ValueError("STEP round-trip returned an open shell")
            shells.Next()
            faces = TopExp_Explorer(imported, TopAbs_FACE)
            count = 0
            while faces.More():
                count += 1
                faces.Next()
            if shells.More() or count != len(self.poles):
                raise ValueError("STEP round-trip changed the fitted shell topology")
            check = BOPAlgo_ArgumentAnalyzer()
            check.SetShape1(imported)
            check.SelfInterMode = True
            check.Perform()
            if check.HasErrors() or check.HasFaulty():
                raise ValueError("STEP solid failed the OCCT self-intersection check")
            temporary.replace(path)
