
import numpy as np

from . import capi, utils
import pyvista as pv
class CPGEO:

    def __init__(self, control_points: np.ndarray, cp_faces: np.ndarray, knot_influence_num: int = 20):
        """Initialize the CPGEO model with control points and connectivity.
        
        Args:
            control_points (np.ndarray): The control points used for mapping, shape `(V, 3)`.
            cp_faces (np.ndarray): The connectivity of the control points, shape `(F, 3)`.
            knot_influence_num (int): The number of control points influenced by each knot point.
        
        """


        self._control_points: np.ndarray
        """the control points used for mapping, shape (V, 3)"""
        self.control_points = control_points

        self._cp_faces: np.ndarray = cp_faces
        """the connectivity of the control points, shape (F, 3)"""

        self._knots: np.ndarray = np.empty((0, 3), dtype=np.float64)
        """the knot points used for mapping, shape (V, 3)"""

        self._knot_influence_num: int = knot_influence_num
        """the number of control points influenced by each knot point"""

        self._thresholds: np.ndarray = None
        """the thresholds of each knot point, shape (V,)"""

        self._space_tree = None
        """the space tree for fast querying"""

        self._symmetry_points_match: np.ndarray = None
        """the matched control points for each symmetry point, shape (V, 2) for axial symmetry, shape (V, n) for Cn rotational symmetry"""

    def __getstate__(self):
        """Custom pickling to exclude C pointers."""
        state = self.__dict__.copy()
        state['_space_tree'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to reset C pointers."""
        self.__dict__.update(state)
        self._space_tree = None
        self.initialize()

    def initialize(self):
        """Initialize the knot points and thresholds based on the control points."""
        
        self._replace_control_mesh(self._control_points, self._cp_faces)

    def _replace_control_mesh(self, points, faces, matching_alpha=None, fixed_points=None):
        knots = capi.initialize_knots(faces, len(points))
        if not 0 < self._knot_influence_num < len(knots):
            raise ValueError("knot_influence_num must be between 1 and the number of knots minus one")
        thresholds = capi.compute_thresholds(knots, self._knot_influence_num)
        tree = capi.space_tree_create(knots, thresholds)
        if not tree:
            raise RuntimeError("Could not create the reference mesh space tree")
        try:
            if matching_alpha is not None:
                points = capi.fit_reconstructed_control_points(tree, points, matching_alpha, fixed_points)
        except BaseException:
            capi.space_tree_destroy(tree)
            raise

        old_tree = self._space_tree
        self._control_points, self._cp_faces = points, faces
        self._knots, self._thresholds, self._space_tree = knots, thresholds, tree
        if old_tree is not None:
            capi.space_tree_destroy(old_tree)

    def get_weights3(self, query_points: np.ndarray, derivative: int = 0):
        """
        Get the weights of control points for given query points.

        Args:
            query_points (np.ndarray): The query points, shape (N, 3).
        Returns:
            indices_cps (np.ndarray): The indices of control points influencing each query point, shape (N, V).
            indices_pts (np.ndarray): The indices of knot points influencing each query point, shape (N, V).
            w (np.ndarray): The weights of control points for each query point, shape (N, V).
            wdu (np.ndarray): The first derivative of weights, shape (2, N, V), optional if derivative >= 1.
            wdu2 (np.ndarray): The second derivative of weights, shape (2, 2, N, V), optional if derivative == 2.
        """
        
        indices_cps, indices_pts = capi.get_space_tree_query(self._space_tree, query_points)

        points_plane = self.reference_to_curvilinear(query_points)

        if derivative == 0:
            w = capi.get_weights(indices_cps=indices_cps,
                                            indices_pts=indices_pts,
                                            knots=self._knots,
                                            thresholds=self._thresholds,
                                            query_points=points_plane)
            return indices_cps, indices_pts, w
        elif derivative == 1:
            w, wdu = capi.get_weights_derivative1(indices_cps=indices_cps,
                                            indices_pts=indices_pts,
                                            knots=self._knots,
                                            thresholds=self._thresholds,
                                            query_points=points_plane)
            return indices_cps, indices_pts, w, wdu.reshape([2, -1])
        
        elif derivative == 2:
            w, wdu, wdu2 = capi.get_weights_derivative2(indices_cps=indices_cps,
                                            indices_pts=indices_pts,
                                            knots=self._knots,
                                            thresholds=self._thresholds,
                                            query_points=points_plane)
            return indices_cps, indices_pts, w, wdu.reshape([2, -1]), wdu2.reshape([2, 2, -1])
    
    def get_weights2(self, query_points_plane: np.ndarray, derivative: int = 0):
        """
        Get the weights of control points for given query points.

        Args:
            query_points (np.ndarray): The query points, shape (N, 3).
        Returns:
            np.ndarray: The weights of control points for each query point, shape (N, V).
        """
        query_points = self.curvilinear_to_reference(query_points_plane)[0]
        indices_cps, indices_pts = capi.get_space_tree_query(self._space_tree, query_points)


        if derivative == 0:
            w = capi.get_weights(indices_cps=indices_cps,
                                            indices_pts=indices_pts,
                                            knots=self._knots,
                                            thresholds=self._thresholds,
                                            query_points=query_points_plane)
            return indices_cps, indices_pts, w
        elif derivative == 1:
            w, wdu = capi.get_weights_derivative1(indices_cps=indices_cps,
                                            indices_pts=indices_pts,
                                            knots=self._knots,
                                            thresholds=self._thresholds,
                                            query_points=query_points_plane)
            return indices_cps, indices_pts, w, wdu.reshape([2, -1])
        
        elif derivative == 2:
            w, wdu, wdu2 = capi.get_weights_derivative2(indices_cps=indices_cps,
                                            indices_pts=indices_pts,
                                            knots=self._knots,
                                            thresholds=self._thresholds,
                                            query_points=query_points_plane)
            return indices_cps, indices_pts, w, wdu.reshape([2, -1]), wdu2.reshape([2, 2, -1])
    
    def map2(self, points_plane: np.ndarray, derivative: int = 0):
        """
        Map the given points in curvilinear coordinates to reference coordinates.

        Args:
            points_plane (np.ndarray): The points in curvilinear coordinates, shape (N, 2).

        Returns:
            np.ndarray: The mapped points in reference coordinates, shape (N, 3).
        """

        results = self.get_weights2(points_plane, derivative=derivative)
        indices_cps, indices_pts, w = results[:3]
        r = capi.get_mapped_points(indices_cps, indices_pts, w, self._control_points, points_plane.shape[0])

        result = []
        if derivative == 0:
            return r
        result.append(r)
        if derivative >= 1:
            wdu = results[3]
            rdu = np.stack([
                capi.get_mapped_points(indices_cps, indices_pts, wdu[0], self._control_points, points_plane.shape[0]),
                capi.get_mapped_points(indices_cps, indices_pts, wdu[1], self._control_points, points_plane.shape[0]),
            ], axis=-1)

            result.append(rdu)
        if derivative == 2:
            wdu2 = results[4]
            rdu2 = np.stack([
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[0,0], self._control_points, points_plane.shape[0]),
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[0,1], self._control_points, points_plane.shape[0]),
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[1,0], self._control_points, points_plane.shape[0]),
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[1,1], self._control_points, points_plane.shape[0]),
            ], axis=-1).reshape([-1, 3, 2, 2])

            result.append(rdu2)

        return tuple(result)
    
    def map3(self, points: np.ndarray, derivative: int = 0):
        """
        Map the given points in reference coordinates to physical coordinates.

        Args:
            points (np.ndarray): The points in reference coordinates, shape (N, 3).
        Returns:
            np.ndarray: The mapped points in physical coordinates, shape (N, 3).
        """
        if derivative == 0:
            if self._space_tree is None:
                raise RuntimeError("Initialize the surface before mapping points")
            return capi.map_points(points, self._space_tree, self._control_points)
        points_plane = self.reference_to_curvilinear(points)
        return self.map2(points_plane, derivative=derivative)
    

    def fit_bspline_patches(self, tolerance: float = 0.01, *, max_spans: int = 64, target_nodes: int = 64):
        """Fit closed bicubic patches to a sampled maximum position error in model units.

        Partition crowded root charts by reference node count and refine their
        spline intervals by measured error, up to max_spans per patch axis.
        target_nodes (default 64) guides the initial partition. Smooth siblings
        may merge and under-resolved tiles may split further. Adaptive results
        include each tile's knot vectors, parameter bounds and source-node count.
        """
        from .adaptive_patches import _locate, adaptive_fit
        from .bspline import BSplinePatches, axial_chart

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not isinstance(max_spans, (int, np.integer)) or not 1 <= max_spans <= 128:
            raise ValueError("max_spans must be an integer between 1 and 128")
        if not isinstance(target_nodes, (int, np.integer)) or target_nodes < 1:
            raise ValueError("target_nodes must be a positive integer")
        if self._space_tree is None:
            self.initialize()
        chart = axial_chart(self._knots, self._thresholds)
        faces, uv = _locate(self._knots, None if chart is None else chart[0])
        # Retain the inexpensive six-face result for simple, sparse surfaces.
        if chart is None and np.bincount(faces, minlength=6).max() <= target_nodes:
            spans = min(4, max_spans)
            poles, metrics = capi.fit_bspline_patches(self._space_tree, self.control_points, spans)
            if metrics[0] <= tolerance:
                return BSplinePatches(poles, *metrics)
        return adaptive_fit(self, chart, tolerance, max_spans, target_nodes, faces, uv)

    def uniformly_mesh(self, init_vertices: np.ndarray = None, seed_size: float = 1.0,
                       max_iterations: int = 10, *, fixed_count: int = 0):
        """
        Run uniform remeshing using the C API wrapper `capi.uniformly_mesh`.

        Args:
            init_vertices (np.ndarray, optional): Initial sphere vertices (N,3). If None, uses current knots as starting vertices.
            seed_size (float): Desired seed size for remeshing.
            max_iterations (int): Zero stops after inserting and deleting seeds. A positive
                budget runs one overlapping-patch smoothing sweep, then up to this many
                edge-flip passes.
            fixed_count (int): Preserve this many leading seeds throughout remeshing.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (vertices (M,3), faces (F,3)) from the remeshing.
        """
        if self._space_tree is None:
            self.initialize()
        return capi.uniformly_mesh(self._knots if init_vertices is None else init_vertices,
                                   self._control_points, self._space_tree, seed_size, max_iterations,
                                   fixed_count=fixed_count, compact_faces=True)

    def refine_surface(self, seed_size: float = 1.0, max_iterations: int = 10,
                       *, matching_alpha: float | None = 0.1, fixed_control_indices=None):
        """Reconstruct the surface and fit its control points to samples of the old surface.

        matching_alpha weights squared control-point displacement relative to squared
        sample mismatch. Zero gives an unregularized fit. None skips the fit.

        fixed_control_indices selects rows of the current control_points array.
        Their XYZ coordinates are copied exactly. Returns their new indices in
        the supplied order for the next reconstruction, or None if omitted.
        The reference knots are reparameterized as usual.
        """
        if self._space_tree is None:
            self.initialize()
        targets, faces, fixed, new_ids = capi.prepare_reconstruction(
            self._space_tree, self._control_points, seed_size, max_iterations,
            fixed_control_indices, matching_alpha,
        )
        self._replace_control_mesh(targets, faces, matching_alpha, fixed)
        self._symmetry_points_match = None
        return new_ids

    def reconstruct_symmetry(self,
                             mode: str = "axial",
                             plane: str = "yz",
                             plane_offset: float = 0.0,
                             keep_positive: bool = True,
                             periods: int = 2,
                             tol: float = 1e-8,
                             inplace: bool = True):
        """
        Reconstruct control mesh with enforced symmetry and store point correspondence.

        Args:
            mode: "axial" or "rotational".
            plane: Symmetry plane for axial mode.
            plane_offset: Plane offset for axial mode.
            keep_positive: Keep positive side for axial mode.
            periods: Cn periods for rotational mode.
            tol: Numerical tolerance used by utils.
            inplace: Update current model when True.

        Returns:
            (new_vertices, new_faces, symmetry_points_match)
        """
        key = mode.lower().strip()
        if key in ("axial", "axis", "planar"):
            new_vertices, new_faces, match = utils.enforce_axial_symmetry(
                self._control_points,
                self._cp_faces,
                plane=plane,
                plane_offset=plane_offset,
                keep_positive=keep_positive,
                tol=tol,
                return_match=True,
            )
        elif key in ("rotational", "rotation", "cn"):
            new_vertices, new_faces, match = utils.enforce_rotational_symmetry_z(
                self._control_points,
                self._cp_faces,
                periods=periods,
                tol=tol,
                return_match=True,
            )
        else:
            raise ValueError("mode must be one of {'axial', 'rotational'}.")

        # Always keep the latest symmetry correspondence on the model.
        self._symmetry_points_match = match

        if inplace:
            self._control_points = new_vertices
            self._cp_faces = new_faces
            self.initialize()

        return new_vertices, new_faces, match

        

    def show(self):
        r = self.map3(self._knots)
        coo = self._cp_faces

        mesh = pv.PolyData(r, np.hstack([np.full((coo.shape[0], 1), 3), coo]))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=1.0)

        def callback(picked_point, picker):
            point_id = picker.GetPointId()
            if point_id < 0: return
            point = mesh.points[point_id]
            print(f"Node Index: {point_id}, Coordinates: {point}")
            plotter.add_point_labels([point], [f"ID: {point_id}"], point_size=20, font_size=18, name="picked_label", always_visible=True)

        plotter.enable_point_picking(callback=callback, show_message=True, use_picker=True, show_point=True, color='red', picker='point')
        plotter.show()

    def show_knots(self):
        r = self._knots
        coo = self._cp_faces

        mesh = pv.PolyData(r, np.hstack([np.full((coo.shape[0], 1), 3), coo]))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=1.0)
        def callback(picked_point, picker):
            point_id = picker.GetPointId()
            if point_id < 0: return
            point = mesh.points[point_id]
            print(f"Node Index: {point_id}, Coordinates: {point}")
            plotter.add_point_labels([point], [f"ID: {point_id}"], point_size=20, font_size=18, name="picked_label", always_visible=True)

        plotter.enable_point_picking(callback=callback, show_message=True, use_picker=True, show_point=True, color='red', picker='point')
        plotter.show()

    def show_control_points(self):
        r = self._control_points
        coo = self._cp_faces

        mesh = pv.PolyData(r, np.hstack([np.full((coo.shape[0], 1), 3), coo]))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=1.0)        
        def callback(picked_point, picker):
            point_id = picker.GetPointId()
            if point_id < 0: return
            point = mesh.points[point_id]
            print(f"Node Index: {point_id}, Coordinates: {point}")
            plotter.add_point_labels([point], [f"ID: {point_id}"], point_size=20, font_size=18, name="picked_label", always_visible=True)

        plotter.enable_point_picking(callback=callback, show_message=True, use_picker=True, show_point=True, color='red', picker='point')
        plotter.show()

    def curvilinear_to_reference(self, curvilinear_points: np.ndarray,
                                 derivative: int = 0):
        """
        transform curvilinear coordinates to reference coordinates

        Args:
            curvilinear_points (np.ndarray): curvilinear coordinates, shape (N, 2)
            derivative (int): order of derivative to compute (0, 1, or 2)

        Returns:
            list: [x] if derivative == 0
                    [x, xdu] if derivative == 1
                    [x, xdu, xdu2] if derivative == 2
                where x is the reference coordinates, shape (N, 3)
                xdu is the first derivative, shape (N, 3, 2)
                xdu2 is the second derivative, shape (N, 3, 2, 2)
        """

        num_points = curvilinear_points.shape[0]

        x = np.zeros([num_points, 3])
        t = 1 / (4 + curvilinear_points[:, 0]**2 + curvilinear_points[:, 1]**2)
        x[:, 0] = 4 * curvilinear_points[:, 0] * t
        x[:, 1] = 4 * curvilinear_points[:, 1] * t
        x[:, 2] = (1 - 8 * t)

        output = [x]
        if derivative >= 1:

            tdu = np.zeros([num_points, 2])
            tdu[:, 0] = -2 * curvilinear_points[:, 0] * t**2
            tdu[:, 1] = -2 * curvilinear_points[:, 1] * t**2
            xdu = np.zeros([num_points, 3, 2])
            xdu[:, 0, 0] = 4 * t + 4 * curvilinear_points[:, 0] * tdu[:, 0]
            xdu[:, 0, 1] = 4 * curvilinear_points[:, 0] * tdu[:, 1]
            xdu[:, 1, 0] = 4 * curvilinear_points[:, 1] * tdu[:, 0]
            xdu[:, 1, 1] = 4 * t + 4 * curvilinear_points[:, 1] * tdu[:, 1]
            xdu[:, 2, 0] = -8 * tdu[:, 0]
            xdu[:, 2, 1] = -8 * tdu[:, 1]

            output.append(xdu)
        if derivative >= 2:
            tdu2 = np.zeros([num_points, 2, 2])
            tdu2[:, 0, 0] = -2 * t**2 - 4 * curvilinear_points[:, 0] * t * tdu[:, 0]
            tdu2[:, 1, 1] = -2 * t**2 - 4 * curvilinear_points[:, 1] * t * tdu[:, 1]
            tdu2[:, 0, 1] = -4 * curvilinear_points[:, 0] * t * tdu[:, 1]
            tdu2[:, 1, 0] = tdu2[:, 0, 1]

            xdu2 = np.zeros([num_points, 3, 2, 2])
            xdu2[:, 0, 0, 0] = 4 * tdu[:, 0] + 4 * curvilinear_points[:, 0] * tdu2[:, 0, 0] + 4 * tdu[:, 0]
            xdu2[:, 0, 0, 1] = 4 * tdu[:, 1] + 4 * curvilinear_points[:, 0] * tdu2[:, 0, 1]
            xdu2[:, 0, 1, 0] = xdu2[:, 0, 0, 1]
            xdu2[:, 0, 1, 1] = 4 * curvilinear_points[:, 0] * tdu2[:, 1, 1]
            xdu2[:, 1, 0, 0] = 4 * curvilinear_points[:, 1] * tdu2[:, 0, 0]
            xdu2[:, 1, 0, 1] = 4 * tdu[:, 0] + 4 * curvilinear_points[:, 1] * tdu2[:, 0, 1]
            xdu2[:, 1, 1, 0] = xdu2[:, 1, 0, 1]
            xdu2[:, 1, 1, 1] = 4 * tdu[:, 1] + 4 * curvilinear_points[:, 1] * tdu2[:, 1, 1] + 4 * tdu[:, 1]
            xdu2[:, 2] = -8 * tdu2

            output.append(xdu2)

        return output

    def reference_to_curvilinear(self, ref_points: np.ndarray):
        """
        transform reference coordinates to curvilinear coordinates

        Args:
            ref_points (np.ndarray): reference coordinates, shape (N, 3)
        Returns:
            np.ndarray: curvilinear coordinates, shape (N, 2)
        """

        num_points = ref_points.shape[0]

        x = np.zeros([num_points, 2])
        x[:, 0] = 2 * ref_points[:, 0] / (1 - ref_points[:, 2] + 1e-15)
        x[:, 1] = 2 * ref_points[:, 1] / (1 - ref_points[:, 2] + 1e-15)

        ind_inf = np.where(ref_points[:, 2] == 1)[0]
        x[ind_inf, 0] = 0
        x[ind_inf, 1] = 0

        return x

    def __del__(self):
        """Destructor: ensure the C space tree is destroyed when the Python object is garbage collected."""
        try:
            st = getattr(self, '_space_tree', None)
            if st is not None:
                capi.space_tree_destroy(st)
                self._space_tree = None
        except Exception:
            # Never raise exceptions from a destructor
            pass

    @property
    def control_points(self) -> np.ndarray:
        return self._control_points

    @property
    def symmetry_points_match(self) -> np.ndarray:
        return self._symmetry_points_match
    
    @control_points.setter
    def control_points(self, points: np.ndarray):
        if not isinstance(points, np.ndarray):
            raise TypeError("Control points must be a numpy ndarray.")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Control points must be of shape (N, 3).")
        if points.dtype != np.float64:
            points = points.astype(np.float64)
        self._control_points = points


    def save(self, file):
        """Save the CPGEO model to a file.

        Args:
            file (str | file-like): Path to the file or a file-like object (e.g., BytesIO).
        """
        np.savez_compressed(
            file,
            control_points=self._control_points,
            cp_faces=self._cp_faces,
            knot_influence_num=self._knot_influence_num
        )

    @staticmethod
    def load(file) -> 'CPGEO':
        """Load a CPGEO model from a file.

        Args:
            file (str | file-like): Path to the file or a file-like object (e.g., BytesIO).
        Returns:
            CPGEO: The loaded CPGEO model.
        """
        data = np.load(file)
        model = CPGEO(
            control_points=data['control_points'],
            cp_faces=data['cp_faces'],
            knot_influence_num=int(data['knot_influence_num'])
        )
        model.initialize()
        return model
    
def show_surf(vertices: np.ndarray, faces: np.ndarray):
    if vertices.shape[1] == 2:
        temp = np.zeros([vertices.shape[0], 3])
        temp[:, :2] = vertices
        vertices = temp
    mesh = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3), faces]))

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=1.0)
    plotter.show()
