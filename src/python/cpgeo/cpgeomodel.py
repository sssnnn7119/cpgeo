
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

        indices_cps, indices_pts, w = self.get_weights2(points_plane)
        mapped_points_cpp = capi.get_mapped_points(indices_cps, indices_pts, w, self._control_points, points_plane.shape[0])

        return mapped_points_cpp
    
    def map3(self, points: np.ndarray):
        """
        Map the given points in reference coordinates to physical coordinates.

        Args:
            points (np.ndarray): The points in reference coordinates, shape (N, 3).
        Returns:
            np.ndarray: The mapped points in physical coordinates, shape (N, 3).
        """
        indices_cps, indices_pts, w = self.get_weights3(points)
        mapped_points_cpp = capi.get_mapped_points(indices_cps, indices_pts, w, self._control_points, points.shape[0])

        return mapped_points_cpp
    
    

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
        x[:, 0] = 2 * ref_points[:, 0] / (1 - ref_points[:, 2])
        x[:, 1] = 2 * ref_points[:, 1] / (1 - ref_points[:, 2])

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
            thre -= 1

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

        while boundary_points_index[0] != 5:
            boundary_points_index = np.roll(boundary_points_index, 1)

        index_half1_ = np.array(
            list(
                set(index_half1.tolist()) -
                set(boundary_points_index.tolist())))

        phi0 = np.arccos(1 - 2 * index_half1_.shape[0] / num_points)
        r0 = 2 * np.sin(phi0) / (1 - np.cos(phi0))

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
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 2 / 3)]
        index_now = np.where(norms < threshold)[0]
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_now)
        
        # refine the x-
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3[:, 2] *= -1
        knots_cur = self.reference_to_curvilinear(knots3)
        norms = np.linalg.norm(knots_cur, axis=1)
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 2 / 3)]
        index_now = np.where(norms < threshold)[0]
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_now)
        
        # refine the y+
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3[:, 2] *= -1
        knots3 = knots3[:, [1,2,0]]
        knots_cur = self.reference_to_curvilinear(knots3)
        norms = np.linalg.norm(knots_cur, axis=1)
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 2 / 3)]
        index_now = np.where(norms < threshold)[0]
        knots_cur = utils.mesh_regulation_2D(
            knots_cur, faces, index_now)
        
        # refine the y-
        knots3 = self.curvilinear_to_reference(knots_cur)[0]
        knots3[:, 2] *= -1
        knots_cur = self.reference_to_curvilinear(knots3)
        norms = np.linalg.norm(knots_cur, axis=1)
        threshold = np.sort(norms)[int(knots_cur.shape[0] * 2 / 3)]
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

