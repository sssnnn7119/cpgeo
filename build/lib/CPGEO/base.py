import ctypes
import sys
import os
import time

import numpy as np
import torch
import scipy

from .utils import sparse as sparse_methods
from .utils import C_api as C_api
from .utils import mlab_visualization as vis

class _CPGEO_base:
    """
    Base class for computational geometry operations using control points.
    
    This class provides the foundation for geometric mappings between reference,
    curvilinear, and configuration spaces, as well as weight calculation methods
    for point interpolation.
    """

    _subclasses: dict[str, '_CPGEO_base'] = {}

    def __init_subclass__(cls):
        """Register subclasses in the class registry for factory method."""
        cls._subclasses[cls.__name__] = cls

    @classmethod
    def factory(cls, name) -> '_CPGEO_base':
        """
        Factory method to create instances of subclasses by name.
        
        Args:
            name (str): Name of the subclass to instantiate
            
        Returns:
            _CPGEO_base: Instance of the requested subclass or base class if not found
        """
        subclass = cls._subclasses.get(name)
        if subclass is None:
            subclass = cls
        return subclass()

    # region: Initialization and core properties
    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize the CPGEO base object.
        
        Args:
            epsilon (float, optional): Numerical tolerance. Defaults to 1e-6.
        """
        # Core geometric properties
        self._cp_vertices: torch.Tensor = None
        """Control points of the geometric information in the configuration space [3, num_vertices]"""

        self._cp_elements: torch.Tensor = None
        """Element connectivity information"""

        self._mask = self.__class__.__name__
        """Identifier for the specific type of geometry"""

        # Pre-computed data for optimization
        self._pre_vertices: torch.Tensor = None
        self._pre_faces: torch.Tensor = None
        self._pre_indices: torch.Tensor = None
        self._pre_rdot: torch.Tensor = None
        self._pre_rdudot: torch.Tensor = None
        self._pre_rdu2dot: torch.Tensor = None

        # Boundary information and interpolation data
        self.boundary_points_index: list[torch.Tensor] = []
        """Indices of boundary points"""
        
        self._knots: torch.Tensor = None
        """Knot points used for interpolation [3, num_knots]"""
        
        self.knots_weight: torch.Tensor = None
        """Weights associated with each knot point [num_knots]"""
        
        # Threshold information for weight calculation
        self.threshold: torch.Tensor = None
        """Influence radius for each knot point [num_knots]"""

        self.threshold_num: int = 12
        """Number of nearest neighbors to consider for threshold calculation"""

        self._weight_tree = None
        """Data structure for efficient weight computation"""

    def initialize(self):
        """
        Initialize the geometry from control points and element connectivity.
        
        This method sets up the internal state based on the provided control points
        and element connectivity data.
        """
        self._initialize_from_connection(self.cp_elements)
        self._reload_threshold()
        self.knots_weight = torch.ones_like(self.knots[0])
        if self._weight_tree is not None:
            self._delete_trees()
        self._build_trees()

    @property
    def cp_vertices(self) -> torch.Tensor:
        """Control points in the configuration space [3, num_vertices]"""
        return self._cp_vertices
    
    @cp_vertices.setter
    def cp_vertices(self, value: torch.Tensor):
        """
        Set control points in the configuration space.
        
        Args:
            value (torch.Tensor): Control points [3, num_vertices]
            
        Raises:
            ValueError: If the input tensor is not 2D or has an incorrect shape
        """
        if value.ndim != 2 or value.shape[0] != 3:
            raise ValueError("Control points must be a 2D tensor with shape [3, num_vertices]")
        self._cp_vertices = value

    @property
    def cp_elements(self) -> torch.Tensor:
        """Element connectivity information [num_elements, vertices_per_element]"""
        return self._cp_elements
    
    @cp_elements.setter
    def cp_elements(self, value: torch.Tensor):
        """
        Set element connectivity information.
        
        Args:
            value (torch.Tensor): Element connectivity [num_elements, vertices_per_element]
            
        Raises:
            ValueError: If the input tensor is not 2D or has an incorrect shape
        """
        if value.ndim != 2 or value.shape[0] < 3:
            raise ValueError("Element connectivity must be a 2D tensor with shape [num_elements, vertices_per_element]")
        self._cp_elements = value

    @property
    def knots(self) -> torch.Tensor:
        """Knot points used for interpolation [3, num_knots]"""
        return self._knots
    
    @knots.setter
    def knots(self, value: torch.Tensor):
        """
        Set knot points for interpolation.
        
        Args:
            value (torch.Tensor): Knot points [3, num_knots]
            
        Raises:
            ValueError: If the input tensor is not 2D or has an incorrect shape
        """
        if value.ndim != 2 or value.shape[0] != 3:
            raise ValueError("Knot points must be a 2D tensor with shape [3, num_knots]")
        self._knots = value

        

    # endregion

    
    
    # region: Mapping methods
    def map_c(self, points: torch.Tensor = None, derivative: int = 0):
        """
        Map points from curvilinear space to the configuration space.
        
        Args:
            points (torch.Tensor, optional): Point coordinates in curvilinear space [v, num_points].
                If None, uses precomputed vertices.
            derivative (int, optional): Number of derivatives to calculate. Defaults to 0.
                0: position only
                1: position and first derivative
                2: position, first and second derivatives
            
        Returns:
            list[torch.Tensor]: Mapped points in the configuration space. List contents depend on derivative parameter:
                - derivative=0: [r]: positions [3, num_points]
                - derivative=1: [r, rdu]: positions and first derivatives [3, num_points], [3, v, num_points]
                - derivative=2: [r, rdu, rdu2]: positions, first and second derivatives
        """
        if points is not None:
            result = self.get_weights(points, derivative=derivative)
            indices = result[0]
            rdot = result[1]
            if derivative >= 1:
                rdudot = result[2]
            if derivative >= 2:
                rdu2dot = result[3]
        else:
            indices = self._pre_indices
            points = self._pre_vertices
            rdot = self._pre_rdot
            rdudot = self._pre_rdudot
            rdu2dot = self._pre_rdu2dot

        control_points_flatten = self.cp_vertices[:, indices[0]]

        r0 = control_points_flatten * rdot
        r = sparse_methods._sparse_sum(indices,
                                       r0,
                                       numel_output=points.shape[1])

        if derivative == 0:
            return r
        output = [r]
        if derivative >= 1:
            rdu0 = control_points_flatten.reshape(
                control_points_flatten.shape[0], 1, -1) * rdudot.reshape(
                    1, rdudot.shape[0], -1)
            rdu = sparse_methods._sparse_sum(indices,
                                             rdu0,
                                             numel_output=points.shape[1])
            output.append(rdu)
        if derivative >= 2:
            rdu20 = control_points_flatten.reshape(
                control_points_flatten.shape[0], 1, 1, -1) * rdu2dot.reshape(
                    1, rdu2dot.shape[0], rdu2dot.shape[1], -1)
            rdu2 = sparse_methods._sparse_sum(indices,
                                              rdu20,
                                              numel_output=points.shape[1])
            output.append(rdu2)
        return output

    def map(self, points: torch.Tensor = None, derivative: int = 0):
        """
        Map points from reference space to the configuration space.
        
        This is a convenience method that first transforms points from reference space
        to curvilinear space and then to configuration space.
        
        Args:
            points (torch.Tensor, optional): Point coordinates in reference space [3, num_points].
                If None, uses precomputed vertices.
            derivative (int, optional): Number of derivatives to calculate. Defaults to 0.
            
        Returns:
            list[torch.Tensor]: Mapped points in the configuration space.
                See map_c() for details on return format.
        """
        if points is not None:
            ptc = self.reference_to_curvilinear(points)
            return self.map_c(ptc, derivative)
        else:
            return self.map_c(derivative=derivative)
    
    def pre_load(self, num_tessellation: int = 0):
        """
        Preload and cache data to accelerate subsequent calculations.
        
        This method pre-computes and caches frequently used values to avoid
        redundant calculations during mapping operations.
        
        Args:
            num_tessellation (int, optional): Number of tessellation subdivisions. Defaults to 0.
            
        Returns:
            None
        """
        self._pre_vertices, self._pre_faces = self.get_mesh(num_tessellation)

        self._pre_indices, self._pre_rdot, self._pre_rdudot, self._pre_rdu2dot = self.get_weights(
            self.reference_to_curvilinear(self._pre_vertices), derivative=2)

    # endregion

    # region: File I/O operations
    def save(self, path):
        """
        Save the geometric information to a file.
        
        Args:
            path (str): Path where the file will be saved
            
        Returns:
            None
        """
        control_points_np = self.cp_vertices.detach().cpu().numpy()
        mesh_elements_np = self.cp_elements.detach().cpu().numpy()
        reference_mask_np = np.array(self._mask)

        np.savez(path,
                 control_points=control_points_np,
                 mesh_elements=mesh_elements_np,
                 reference_mask=reference_mask_np,
                 geo_mask=self._mask)

    @staticmethod
    def load(path) -> '_CPGEO_base':
        """
        Load geometric information from a file.
        
        Args:
            path (str): Path to the file containing geometric information
            
        Returns:
            _CPGEO_base: An instance of the appropriate geometry class with loaded data
        """
        data = np.load(path)
        geo_mask = data['geo_mask'].item()
        geo = _CPGEO_base.factory(geo_mask)
        geo.cp_vertices = torch.tensor(data['control_points'].tolist())
        geo.cp_elements = torch.tensor(data['mesh_elements'].tolist())
        geo.initialize()
        return geo
    # endregion

    # region: Reference space methods
    def get_weights(self,
                    points: torch.Tensor,
                    derivative: int = 0,
                    knots: torch.Tensor = None):
        """
        Calculate interpolation weights for points in curvilinear space.
        
        This method computes the weights needed to interpolate values at the given
        points based on the knot values.
        
        Args:
            points (torch.Tensor): Point coordinates in curvilinear space [v, num_points]
            derivative (int, optional): Order of derivatives to calculate. Defaults to 0.
            knots (torch.Tensor, optional): Custom knot points to use. Defaults to self.knots.
            
        Returns:
            list[torch.Tensor]: Interpolation weights with varying contents based on derivative:
                - indices [2, num_sparse]: Indices of influencing knots for each point [knot_idx, point_idx]
                - rdot [num_sparse]: Weights for each knot-point pair (derivatives >= 0)
                - rdudot [v, num_sparse]: First derivative weights (derivatives >= 1)
                - rdu2dot [v, v, num_sparse]: Second derivative weights (derivatives >= 2)
        """
        if knots is None:
            knots = self.knots

        c2x = self.curvilinear_to_reference(curvilinear_points=points,
                                            derivative=derivative)
        result = self.weight_method(knots=knots,
                                    points=c2x[0],
                                    derivative=derivative)

        indices = result[0]
        weight = result[1] * self.knots_weight[result[0][0]]
        Wsum = torch.zeros(points.shape[1]).scatter_add_(dim=0,
                                                         index=indices[1],
                                                         src=weight)
        Wsum_flatten = Wsum[indices[1]]
        rdot = weight / Wsum_flatten

        output = [indices, rdot]

        if derivative >= 1:
            wdx = result[2] * self.knots_weight[result[0][0]]
            Wsumdx = sparse_methods._sparse_sum(indices,
                                                wdx,
                                                numel_output=points.shape[1])
            Wsumdx_flatten = Wsumdx[:, indices[1]]
            rdxdot = wdx / Wsum_flatten - weight * Wsumdx_flatten / Wsum_flatten**2

            xdu = c2x[1][:, :, indices[1]]
            rdudot = torch.einsum('ip, iup->up', rdxdot, xdu)
            output.append(rdudot)
        if derivative >= 2:
            wdx2 = result[3] * self.knots_weight[result[0][0]]
            Wsumdx2 = sparse_methods._sparse_sum(indices,
                                                 wdx2,
                                                 numel_output=points.shape[1])
            rdx2dot = wdx2 / Wsum_flatten - weight * Wsumdx2[:, :, indices[
                1]] / Wsum_flatten**2

            Adot = torch.einsum('ip, np->inp', wdx, Wsumdx[:, indices[1]])
            Bdot = -Adot - Adot.transpose(0, 1) + 2 * torch.einsum(
                'p,mp,np->mnp', weight, Wsumdx[:, indices[1]],
                Wsumdx[:, indices[1]]) / Wsum_flatten
            rdx2dot = rdx2dot + Bdot / Wsum_flatten**2

            xdu2 = c2x[2][:, :, :, indices[1]]

            rdu2dot = torch.einsum('ip, iuvp->uvp', rdxdot,
                                   xdu2) + torch.einsum(
                                       'ijp, iup, jvp->uvp', rdx2dot, xdu, xdu)

            output.append(rdu2dot)

        return output

    def get_mesh(self, num_tessellation: int = 0):
        """
        Generate a mesh representation of the geometry.
        
        Args:
            num_tessellation (int, optional): Number of tessellation subdivisions. Defaults to 0.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Vertex coordinates [3, num_vertices] and face indices [3, num_faces]
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_mesh()")

    def mesh_knots(self, knots: torch.Tensor):
        """
        Generate a triangular mesh from the given knot points.
        
        Args:
            knots (torch.Tensor): Knot point coordinates [3, num_points]
            
        Returns:
            torch.Tensor: Triangle indices [3, num_triangles]
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement mesh_knots()")

    def _initialize_from_connection(self, connection: torch.Tensor):
        """
        Initialize the geometry from connectivity information.
        
        This method sets up optimal knots based on the provided element connectivity.
        
        Args:
            connection (torch.Tensor): Element connectivity information [num_elements, vertices_per_element]
            vertices (torch.Tensor, optional): Vertex coordinates
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _initialize_from_connection()")

    def reference_to_curvilinear(self, ref_points: torch.Tensor):
        """
        Convert points from reference space to curvilinear space.
        
        Args:
            ref_points (torch.Tensor): Points in reference space [3, num_points]
            
        Returns:
            torch.Tensor: Points in curvilinear space [v, num_points]
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement reference_to_curvilinear()")

    def curvilinear_to_reference(self, curvilinear_points: torch.Tensor,
                                 derivative: int = 0) -> list[torch.Tensor]:
        """
        Convert points from curvilinear space to reference space.
        
        Args:
            curvilinear_points (torch.Tensor): Points in curvilinear space [v, num_points]
            derivative (int, optional): Order of derivatives to calculate. Defaults to 0.
            
        Returns:
            list[torch.Tensor]: Transformed points with derivatives depending on the derivative parameter:
                - x [3, num_points]: Points in reference space
                - xdu [3, v, num_points]: First derivatives (if derivative >= 1)
                - xdu2 [3, v, v, num_points]: Second derivatives (if derivative >= 2)
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement curvilinear_to_reference()")

    @staticmethod
    def _reposition_knots(knots: torch.Tensor) -> torch.Tensor:
        """
        Reposition knots to ensure they lie on the manifold surface.
        
        Args:
            knots (torch.Tensor): Original knot positions
            
        Returns:
            torch.Tensor: Repositioned knots
        """
        return knots

    def knots_r(self):
        """
        Get knots in curvilinear space.
        
        Returns:
            torch.Tensor: Knot coordinates in curvilinear space
        """
        return self.reference_to_curvilinear(self.knots)

    # endregion

    # region: Weight evaluation methods
    def _reload_threshold(self):
        """
        Calculate appropriate threshold (influence radius) for each knot point.
        
        This method sets the threshold for each knot based on:
        1. Distance to the k-th nearest neighbor (k=10)
        2. Maximum distance to connected knots in the control point elements
        
        The threshold determines how far a knot's influence extends during interpolation.
        """
        if self.knots is None:
            return
        
        # Number of knots and k-value for k-th nearest neighbor
        num_knots = self.knots.shape[1]
        k_neighbors = self.threshold_num  # Use 10th nearest neighbor as default
        
        # Convert knots to contiguous numpy arrays for C++ interop
        knots_np = self.knots.T.detach().cpu().numpy().astype(np.float64).copy(order='C')
        
        # Prepare results buffer
        threshold_np = np.zeros(num_knots, dtype=np.float64)
        
        # Get DLL function to calculate thresholds
        get_thresholds_func = C_api.faceApi.get_thresholds
        get_thresholds_func.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # results
            ctypes.POINTER(ctypes.c_double),  # knots
            ctypes.c_int,                     # number of knots
            ctypes.c_int                      # k neighbors
        ]
        get_thresholds_func.restype = None
        
        # Create C-compatible pointers to the data
        threshold_ptr = threshold_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        knots_ptr = knots_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call the C++ function to calculate thresholds
        get_thresholds_func(threshold_ptr, knots_ptr, num_knots, k_neighbors)
        
        # Convert results to torch tensor
        self.threshold = torch.from_numpy(threshold_np).to(self.knots.device)

        # Ensure threshold includes all connected knots in elements
        # for i in range(self.knots.shape[1]):
        #     # Find all knots connected to this knot through control point elements
        #     neighbor_points = self.cp_elements[torch.where((self.cp_elements == i).sum(dim=1) > 0)[0]].unique().cpu()
        #     if neighbor_points.shape[0] == 0:
        #         continue
        #     # Get maximum distance to connected knots
        #     dist = self._distance_measure(self.knots[:, i].unsqueeze(-1), self.knots[:, neighbor_points])[1].max()
            
        #     # Update threshold if needed to include all connected knots
        #     if dist > self.threshold[i]:
        #         self.threshold[i] = dist

    @staticmethod
    def _distance_measure(knots: torch.Tensor, ref_points: torch.Tensor, indices: torch.Tensor = None):
        """
        Calculate Euclidean distances between knots and reference points.
        
        Args:
            knots (torch.Tensor): Knot coordinates [3, num_knots]
            ref_points (torch.Tensor): Reference point coordinates [3, num_points]
            
        Returns:
            torch.Tensor: Distance matrix [num_knots, num_points]
        """
        
        if indices is None:
            # Calculate all pairwise distances
            delta = ref_points.unsqueeze(1) - knots.unsqueeze(2)
            D = torch.sqrt((delta**2).sum(dim=0))
        else:
            # Reshape for broadcasting
            Y = knots[:, indices[0]]  # Knot coordinates
            y = ref_points[:, indices[1]]  # Point coordinates
            
            # Calculate Euclidean distances
            delta = y - Y
            D = torch.sqrt((delta**2).sum(dim=0))
            
        return delta, D

    
    def _build_trees(self):
        """
        Build octree data structure for efficient weight computation.
        
        This method constructs an octree for fast spatial queries to find
        knots that influence each point. It uses the C++ implementation
        from the CPGEO DLL.
        
        Returns:
            None
        """
        # Check if we already have knots and thresholds
        if self.knots is None or self.threshold is None:
            return None
        
        # Free the previous tree if it exists
        if self._weight_tree is not None:
            self._delete_trees()
        
        # Convert to contiguous numpy arrays for C++ interop
        knots_np = self.knots.T.detach().cpu().numpy().astype(np.float64).copy(order='C')
        threshold_np = self.threshold.detach().cpu().numpy().astype(np.float64).copy(order='C')
        knots_count = self.knots.shape[1]
        
        # Get DLL function to build the octree
        build_trees_func = C_api.faceApi.build_trees
        build_trees_func.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # knots
            ctypes.POINTER(ctypes.c_double),  # threshold
            ctypes.c_int                      # knots count
        ]
        build_trees_func.restype = ctypes.POINTER(ctypes.c_void_p)  # Returns a pointer to the tree
        
        # Create C-compatible pointers to the data
        knots_ptr = knots_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        threshold_ptr = threshold_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call the C++ function to build the tree
        tree_ptr = build_trees_func(knots_ptr, threshold_ptr, knots_count)
        
        # Store the tree pointer for later use
        self._weight_tree = tree_ptr

    def _delete_trees(self):
        """
        Delete the octree data structure.
        
        This method frees the memory allocated for the octree in C++.
        """
        if self._weight_tree is None:
            return
            
        # Get DLL function to delete the tree
        delete_trees_func = C_api.faceApi.delete_trees
        delete_trees_func.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        delete_trees_func.restype = None
        
        # Call C++ function to delete the tree
        delete_trees_func(self._weight_tree)
        self._weight_tree = None

    def get_indices(self, points: torch.Tensor):
        """
        Calculate the indices of influencing knots for each point using the octree.
        
        Args:
            points (torch.Tensor): Point coordinates in reference space [3, num_points]
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Two tensors containing:
                - indices [2, num_nonzero]: Indices of influencing knots [knot_idx, point_idx]
                - values [num_nonzero]: Values indicating influence strength (ones)
        """
        # Build the tree if it doesn't exist
        if self._weight_tree is None:
            self._build_trees()
        
        # Ensure the tree exists
        if self._weight_tree is None:
            raise RuntimeError("Failed to build spatial tree for weight calculation")
        
        # Convert points to numpy for C++ interop
        points_np = points.T.detach().cpu().numpy().astype(np.float64).copy()
        num_points = points.shape[1]
        
        # Prepare arrays to hold results from C++
        result_sizes = np.zeros([num_points], dtype=np.int32)

        # Setup the C-API function
        cal_indices_func = C_api.faceApi.cal_indices
        cal_indices_func.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # tree pointer
            ctypes.POINTER(ctypes.c_double),  # points
            ctypes.c_int,                     # num_points
            ctypes.POINTER(ctypes.c_int)      # result sizes
        ]
        cal_indices_func.restype = None
        
        # Create C-compatible pointers
        points_ptr = points_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sizes_ptr = result_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # Call C++ to get indices
        cal_indices_func(self._weight_tree, points_ptr, num_points, sizes_ptr)
        
        # Read the results
        indices = np.zeros([2, result_sizes.sum()], dtype=np.int32)
        
        # Setup the C-API function
        get_indices_func = C_api.faceApi.get_indices
        get_indices_func.argtypes = [
            ctypes.POINTER(ctypes.c_int) ,  # result arrays
        ]
        get_indices_func.restype = None
        
        # Create C-compatible pointers
        indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # Call C++ to get indices
        get_indices_func(indices_ptr)

        return torch.from_numpy(indices).to(points.device).to(torch.int64)

    @staticmethod
    def get_indices_2nd(indices: torch.Tensor):
        """
        Calculate additional influencing knots based on an initial set of indices.
        
        This method expands the set of influencing knots by including secondary connections
        that may be relevant for more accurate interpolation.
        
        Args:
            indices (torch.Tensor): Initial indices of influencing knots [2, num_indices]
            
        Returns:
            torch.Tensor: Expanded indices of influencing knots [2, num_expanded_indices]
        """

        # Check if input indices is empty
        if indices.shape[1] == 0:
            return indices
        
        # Convert indices to numpy for C++ interop
        indices_np = indices.detach().cpu().numpy().astype(np.int32).copy(order='C')
        num_indices = indices.shape[1]
        indices_2_size = np.zeros(1, dtype=np.int32)
        
        # Setup the C-API function
        cal_indices_2nd_func = C_api.faceApi.cal_indices_2nd
        cal_indices_2nd_func.argtypes = [
            ctypes.POINTER(ctypes.c_int),     # tree pointer
            ctypes.c_int,                     # number of indices pairs
            ctypes.POINTER(ctypes.c_int),     # input indices
        ]
        cal_indices_2nd_func.restype = None
        
        # Create C-compatible pointers
        indices_ptr = indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        indices_2_size_ptr = indices_2_size.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        # Call C++ to expand the indices
        cal_indices_2nd_func(indices_ptr, num_indices, indices_2_size_ptr)
        
        # Get the size of the expanded indices array
        get_indices_2nd_func = C_api.faceApi.get_indices_2nd
        get_indices_2nd_func.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_indices_2nd_func.restype = None
        
        # Call C++ to get the expanded indices
        expanded_indices = np.zeros([2, indices_2_size[0]], dtype=np.int32)
        expanded_indices_ptr = expanded_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        get_indices_2nd_func(expanded_indices_ptr)

        # Convert to PyTorch tensor and return
        return torch.from_numpy(expanded_indices).to(indices.device)
    
    def __del__(self):
        """
        Cleanup resources when the object is deleted.
        
        This method ensures that the C++ tree structure is properly deleted
        when the Python object is garbage collected.
        """
        if hasattr(self, '_weight_tree') and self._weight_tree is not None:
            self._delete_trees()
            
        
    def weight_method(self,
                    knots: torch.Tensor,
                    points: torch.Tensor,
                    derivative: int = 0):
        """
        Calculate interpolation weights and derivatives for given points.
        
        This method uses a C2-continuous weight function (quartic polynomial)
        to determine how much each knot influences each point, based on distance
        and threshold values. It uses the octree for efficient spatial queries.
        
        Args:
            knots (torch.Tensor): Knot coordinates [3, num_knots]
            points (torch.Tensor): Point coordinates [3, num_points]
            derivative (int, optional): Order of derivatives to calculate. Defaults to 0.
            
        Returns:
            tuple: Contents depend on derivative value:
                - derivative=0: (indices, weight)
                - derivative=1: (indices, weight, wdx)
                - derivative=2: (indices, weight, wdx, wdx2)
            
            Where:
                - indices [2, num_nonzero]: Indices of influencing knots [knot_idx, point_idx]
                - weight [num_nonzero]: Weight values for each knot-point pair
                - wdx [3, num_nonzero]: First derivatives of weights
                - wdx2 [3, 3, num_nonzero]: Second derivatives of weights
        """
        # Use octree to efficiently find influencing knots
        indices = self.get_indices(points)
        
        if indices.shape[1] == 0:
            # Handle the case where no indices were found
            if derivative == 0:
                return indices, torch.tensor([])
            elif derivative == 1:
                return indices, torch.tensor([]), torch.tensor([])
            else:
                return indices, torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Calculate Euclidean distances
        delta, D_flatten = self._distance_measure(knots, points, indices)
        threshold_flatten = self.threshold[indices[0]]
        
        # Normalized distance
        d = threshold_flatten
        Dd = D_flatten / d
        
        # Calculate weights using a C2-continuous quartic polynomial: (1-r²)²
        # Written as: 20r⁷ - 70r⁶ + 84r⁵ - 35r⁴ + 1
        weight = (20 * Dd**7 - 70 * Dd**6 + 84 * Dd**5 - 35 * Dd**4 + 1)

        if derivative == 0:
            return indices, weight

        # Calculate first derivatives if needed
        wdx = (20 * 7 * Dd**5 - 70 * 6 * Dd**4 + 84 * 5 * Dd**3 -
            35 * 4 * Dd**2) / d**2 * delta

        if derivative == 1:
            return indices, weight, wdx

        # Calculate second derivatives if needed
        wdx2 = (20 * 35 * Dd**3 -
                70 * 24 * Dd**2 +
                84 * 15 * Dd -
                35 * 8) / d**4 * torch.einsum('ip,jp->ijp', delta, delta) + \
                (20 * 7 * Dd**5 -
                70 * 6 * Dd**4 +
                84 * 5 * Dd**3 -
                35 * 4 * Dd**2) / d**2 * torch.eye(3, device=points.device).reshape([3, 3, 1])

        return indices, weight, wdx, wdx2
    
    # def weight_method(self,
    #                  knots: torch.Tensor,
    #                  points: torch.Tensor,
    #                  derivative: int = 0):
        
    #     y = points
    #     Y = knots
    #     threshold = self.threshold
    #     degree = 3
    #     indices = self.get_indices(points)
        
    #     D = ((y[:, indices[1]] - Y[:, indices[0]])**2).sum(dim=0)
    #     delta = y[:, indices[1]] - Y[:, indices[0]]

    #     threshold_flatten = threshold[indices[0]]
        
        
    #     DD = (threshold_flatten - D)
    #     ind_remain = DD > 0
    #     indices = indices[:, ind_remain]
    #     D = D[ind_remain]
    #     delta = delta[:, ind_remain]
    #     threshold_flatten = threshold[indices[0]]

    #     weight0 = (threshold_flatten - D) / (threshold_flatten)

    #     weight = weight0**degree

    #     if derivative == 0:
    #         return indices, weight

    #     Ddx = delta * 2

    #     wdx = -degree / (threshold_flatten) * weight / weight0 * Ddx

    #     if derivative == 1:
    #         return indices, weight, wdx

    #     Ddx2 = torch.eye(3).reshape([3, 3, 1]) * 2

    #     wdx2 = degree * (degree - 1) / (threshold_flatten)**2 * weight / weight0**2 * torch.einsum('ip,jp->ijp',Ddx,Ddx) \
    #         - degree / (threshold_flatten) * weight / weight0 * Ddx2

    #     return indices, weight, wdx, wdx2
    # endregion

    # region: Integration methods
    
    def get_guassion_points(self, num_subdivisions: int, num_gauss_points: int):
        """
        Get Gauss points for numerical integration.
        
        Args:
            num_points (int): Number of Gauss points to generate
            
        Returns:
            tuple[torch.Tensor]: 
            Gauss points at curvilinear space [v, num_points]
            Gauss weights [num_points]
            
        """
        raise NotImplementedError("Subclasses must implement get_guassion_points()")
    
    # endregion