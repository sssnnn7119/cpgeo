
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

    def initialize(self):
        """Initialize the knot points and thresholds based on the control points."""
        
        self._knots = self._initialize_knots(faces=self._cp_faces)
        self._thresholds = capi.compute_thresholds(
            knots=self._knots,
            k=self._knot_influence_num,)
        
        if self._space_tree is not None:
            capi.space_tree_destroy(self._space_tree)
        self._space_tree = capi.space_tree_create(knots=self._knots,
                                                  thresholds=self._thresholds)
        
    def get_weights3(self, query_points: np.ndarray, derivative: int = 0):
        """
        Get the weights of control points for given query points.

        Args:
            query_points (np.ndarray): The query points, shape (N, 3).
        Returns:
            np.ndarray: The weights of control points for each query point, shape (N, V).
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

        results = self.get_weights2(points_plane)
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
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[0], self._control_points, points_plane.shape[0]),
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[1], self._control_points, points_plane.shape[0]),
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[2], self._control_points, points_plane.shape[0]),
                capi.get_mapped_points(indices_cps, indices_pts, wdu2[3], self._control_points, points_plane.shape[0]),
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
        points_plane = self.reference_to_curvilinear(points)
        return self.map2(points_plane, derivative=derivative)
    

    def uniformly_mesh(self, init_vertices: np.ndarray = None, seed_size: float = 1.0, max_iterations: int = 10):
        """
        Run uniform remeshing using the C API wrapper `capi.uniformly_mesh`.

        Args:
            init_vertices (np.ndarray, optional): Initial sphere vertices (N,3). If None, uses current knots as starting vertices.
            seed_size (float): Desired seed size for remeshing.
            max_iterations (int): Maximum number of uniforming iterations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (vertices (M,3), faces (F,3)) from the remeshing.
        """
        # Ensure control points are present
        if self._control_points is None or self._control_points.size == 0:
            raise RuntimeError("Control points are not set. Call CPGEO(...) with valid control points.")

        # If no initial vertices provided, use current knots (initialize if needed)
        if init_vertices is None:
            if self._knots is None or self._knots.size == 0:
                self.initialize()
            init_vertices = self._knots

        # Ensure space tree exists
        if self._space_tree is None:
            self.initialize()

        new_vertices, new_faces = capi.uniformly_mesh(
            init_vertices,
            self._control_points,
            self._space_tree,
            seed_size,
            max_iterations
        )

        return new_vertices, new_faces    
    
    def _post_process(self, initial_points: np.ndarray = None):
        """
        Post-process the CPGEO model after remeshing to update control points for error minimization.
        """

        num_points = self.control_points.shape[0]

        # post process

        indices, rdot = self.get_weights2(self._knots, derivative=0)
        r = self.map3(self._knots)

        alpha = 1.0
        loss0 = ((r - initial_points)**2).sum()
        loss1 = ((self.control_points - initial_points)**2).sum()
        loss = loss0 + loss1 * alpha
        print()
        print('before refine P0: loss = %e' % (loss))

        ldr = 2 * (r - initial_points)
        ldr_2_values = 2 * torch.ones([3, num_points]).flatten()
        ldr_2_indices = torch.stack([
            torch.arange(0, 3).reshape([3, 1]).repeat(1, num_points),
            torch.arange(0, num_points).reshape([1, num_points]).repeat(3, 1),
            torch.arange(0, 3).reshape([3, 1]).repeat(1, num_points),
            torch.arange(0, num_points).reshape([1, num_points]).repeat(3, 1),
        ],
                                    dim=0).reshape([4, -1])
        ldr_2 = torch.sparse_coo_tensor(ldr_2_indices,
                                        ldr_2_values,
                                        size=[3, num_points, 3, num_points])

        l0dot = _sparse_methods._from_Adr_to_Adot(indices=indices,
                                                  Adr=ldr,
                                                  rdot=rdot,
                                                  numel_output=num_points)

        l1dot = 2 * (self.control_points - initial_points)

        index_boundary = torch.tensor(sum([
            self.boundary_points_index[i].tolist()
            for i in range(len(self.boundary_points_index))
        ], []))
        index_boundary2 = index_boundary.clone()
        
        for i in range(5):
            index_boundary2 = self.cp_elements[torch.where(torch.isin(self.cp_elements, index_boundary2).sum(dim=1) > 0)[0]].unique()

        index_no_boundary = torch.tensor(
            list(set(range(num_points)) - set(index_boundary2.tolist())))

        ldot = l0dot + l1dot * alpha

        ldot = ldot[:, index_no_boundary]

        l0dot_2 = _sparse_methods._from_Sdr_to_Sdot_2(indices=indices,
                                                      Sdr_2=ldr_2,
                                                      rdot=rdot)

        l1dot_2 = ldr_2

        ldot_2 = l0dot_2 + l1dot_2 * alpha

        ldot_2 = ldot_2.index_select(1, index_no_boundary).index_select(
            3, index_no_boundary)

        ldot_2 = _sparse_methods._sparse_reshape(
            ldot_2, 2 * [3 * ldot_2.shape[1]]).coalesce()

        dP = _sparse_methods._conjugate_gradient(ldot_2.indices(),
                                                 ldot_2.values(),
                                                 -ldot.flatten(),
                                                 tol=1e-7,
                                                 max_iter=30000)
        dP.view(-1)[dP.view(-1).isnan()] = 0
        dP = dP.reshape([3, -1])

        self.cp_vertices[:,
                         index_no_boundary] = self.cp_vertices[:,
                                                               index_no_boundary] + dP

        r1 = self.map(self.knots)

        epsilon = torch.zeros([3, 3, 3])
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

        loss0_new = ((r1 - initial_points)**2).sum()
        loss1_new = ((self.cp_vertices - initial_points)**2).sum() * alpha

        print('after refine P0: loss = %e' % (loss0_new + loss1_new * alpha))

        self.cp_elements = _mesh_methods.refine_triangular_mesh(
        self.cp_vertices.T, self.cp_elements)



    def refine_surface(self, seed_size: float = 1.0, max_iterations: int = 10):
        new_vertices, new_faces = self.uniformly_mesh(init_vertices=self._knots, seed_size=seed_size, max_iterations=max_iterations)
        
        new_control_points = self.map3(new_vertices)

        self._control_points = new_control_points
        self._cp_faces = new_faces

        self.initialize()

        # new_cps = self._post_process()

        # self._control_points = new_cps

        

    def show(self):
        r = self.map3(self._knots)
        coo = self._cp_faces

        mesh = pv.PolyData(r, np.hstack([np.full((coo.shape[0], 1), 3), coo]))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=1.0)
        plotter.show()

    def show_knots(self):
        r = self._knots
        coo = self._cp_faces

        mesh = pv.PolyData(r, np.hstack([np.full((coo.shape[0], 1), 3), coo]))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=1.0)
        plotter.show()

    def show_control_points(self):
        r = self._control_points
        coo = self._cp_faces

        mesh = pv.PolyData(r, np.hstack([np.full((coo.shape[0], 1), 3), coo]))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=1.0)
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

    @staticmethod
    def _split_mesh(faces: np.ndarray):
        num_points = np.unique(faces.flatten()).shape[0]

        

        import networkx as nx

        # using dijkstra algorithm to split the faces into two parts
        edges_with_counts = capi.get_mesh_edges(faces)
        edges0 = edges_with_counts[:, :2]
        edge_count = edges_with_counts[:, 2]
        G = nx.Graph()
        G.add_edges_from(edges0)
        shortest_path = nx.single_source_dijkstra(G, 50)
        keys = np.array(list(shortest_path[0].keys()))
        values = np.array(list(shortest_path[0].values()))
        mid_values = values[round(len(values) / 2)]
        thre = mid_values
        while True:
            index_half1 = np.array(keys[values<thre])

            faces1_index = (np.isin(faces,
                                            index_half1).sum(axis=1) == 3)
            faces1 = faces[faces1_index]
            faces2 = faces[~faces1_index]

            boundary_points_index = capi.extract_boundary_loops(faces1)
            if len(boundary_points_index) == 1:
                break
            thre = int(thre / 3)

        if index_half1.shape[0] < num_points / 2:
            index_half1 = np.array(
                list((set(range(num_points)) - set(index_half1.tolist()))))
        
        faces1_index = (np.isin(faces,
                                        index_half1).sum(axis=1) == 3)
        return index_half1, boundary_points_index[0], faces1, faces2

    def _initialize_knots(self, faces: np.ndarray):

        num_points = np.unique(faces.flatten()).shape[0]

        knots_cur = np.stack([np.linspace(-1, 1, num=int(np.ceil(num_points))),
                              np.linspace(-1, 1, num=int(np.ceil(num_points)))], axis=1)

        index_half1, boundary_points_index, faces1, faces2 = CPGEO._split_mesh(faces)

        index_half1_ = np.array(
            list(
                set(index_half1.tolist()) -
                set(boundary_points_index.tolist())))

        phi0 = np.arccos(1 - 2 * index_half1_.shape[0] / num_points)
        r0 = 2 * np.sin(phi0) / (1 + np.cos(phi0))

        theta = (np.arange(0, boundary_points_index.shape[0]) / boundary_points_index.shape[0]) * 2 * np.pi
        knots_cur[boundary_points_index] = r0 * np.stack(
            [np.cos(theta), np.sin(theta)], axis=1)

        # map from the z+
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_half1_)

        index_extra = np.where(np.linalg.norm(knots_cur, axis=1) > 0.7 * r0)[0]

        index_half2 = np.array(
            list((set(range(num_points)) - set(index_half1_.tolist())).union(
                set(index_extra.tolist()))))

        
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3[:, 2] *= -1
        knots_cur = self.reference_to_curvilinear(knots3)

        # refine the other semi-sphere
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_half2)

        # refine the x+
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3 = knots3[:, [1,2,0]]
        knots_cur = self.reference_to_curvilinear(knots3)
        norms = np.linalg.norm(knots_cur, axis=1)
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 0.9)]
        index_now = np.where(norms < threshold)[0]
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_now)
        
        # refine the x-
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3[:, 2] *= -1
        knots_cur = self.reference_to_curvilinear(knots3)
        norms = np.linalg.norm(knots_cur, axis=1)
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 0.9)]
        index_now = np.where(norms < threshold)[0]
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_now)
        
        # refine the y+
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3[:, 2] *= -1
        knots3 = knots3[:, [1,2,0]]
        knots_cur = self.reference_to_curvilinear(knots3)
        norms = np.linalg.norm(knots_cur, axis=1)
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 0.9)]
        index_now = np.where(norms < threshold)[0]
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_now)
        
        # refine the y-
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3[:, 2] *= -1
        knots_cur = self.reference_to_curvilinear(knots3)
        norms = np.linalg.norm(knots_cur, axis=1)
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 0.9)]
        index_now = np.where(norms < threshold)[0]
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_now)
        
        # finally get the knots
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3 = knots3[:, [1,2,0]]
        
        return knots3

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