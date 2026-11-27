"""Six clamped bicubic patches on the cubed sphere."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


@dataclass
class BSplinePatches:
    """Poles in (+X,-X,+Y,-Y,+Z,-Z), u, v, xyz order on [0,1]².

    Each axis has uniform simple interior knots and end multiplicity four.
    Errors are measured at validation samples in model units. The seam-normal
    angle is in degrees. Metrics describe the poles at fitting time.
    """

    poles: np.ndarray
    max_error: float
    rms_error: float
    max_seam_angle_deg: float

    def to_bsp(self):
        """Return six bspmap.BSP surfaces using the optional bspmap package."""
        from bspmap import BSP
        from bspmap.basis import BasisClamped

        n = self.poles.shape[1]
        basis = (BasisClamped(n, 3), BasisClamped(n, 3))
        return [
            BSP(basis, 3, (n, n), patch.reshape(-1, 3).copy()) for patch in self.poles
        ]

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

        n = self.poles.shape[1]
        spans = n - 3
        knots = Array1_double(1, spans + 1)
        mults = Array1_int(1, spans + 1)
        for i in range(spans + 1):
            knots.SetValue(i + 1, i / spans)
            mults.SetValue(i + 1, 4 if i in (0, spans) else 1)
        sewing = BRepBuilderAPI_Sewing(1e-7)
        for patch in self.poles:
            points = Array2_gp_Pnt(1, n, 1, n)
            for i, row in enumerate(patch, 1):
                for j, xyz in enumerate(row, 1):
                    points.SetValue(i, j, gp_Pnt(*xyz))
            surface = Geom_BSplineSurface(points, knots, knots, mults, mults, 3, 3)
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
            if shells.More() or count != 6:
                raise ValueError("STEP round-trip changed the six-face shell topology")
            check = BOPAlgo_ArgumentAnalyzer()
            check.SetShape1(imported)
            check.SelfInterMode = True
            check.Perform()
            if check.HasErrors() or check.HasFaulty():
                raise ValueError("STEP solid failed the OCCT self-intersection check")
            temporary.replace(path)
