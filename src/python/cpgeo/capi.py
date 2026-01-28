"""
C API bindings for CPGEO using ctypes.
Direct low-level interface to the C library.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Locate the DLL
def _find_dll() -> Path:
    """Find the cpgeo DLL in the package directory."""
    package_dir = Path(__file__).parent
    dll_dir = package_dir / "bin"
    
    if sys.platform == "win32":
        dll_name = "cpgeo.dll"
    elif sys.platform == "darwin":
        dll_name = "libcpgeo.dylib"
    else:
        dll_name = "libcpgeo.so"
    
    dll_path = dll_dir / dll_name
    
    if not dll_path.exists():
        raise FileNotFoundError(
            f"Could not find {dll_name} in {dll_dir}. "
            "Please ensure the library is built and copied to the package directory."
        )
    
    return dll_path


# Load the DLL
_dll_path = _find_dll()
_lib = ctypes.CDLL(str(_dll_path))


# ==================== C API Function Declarations ====================

# Define types
cpgeo_handle_t = ctypes.c_void_p

# triangulation_compute
_lib.triangulation_compute.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # nodes
    ctypes.c_int,                     # num_nodes
    ctypes.POINTER(ctypes.c_int)      # num_triangles
]
_lib.triangulation_compute.restype = cpgeo_handle_t

# triangulation_get_data
_lib.triangulation_get_data.argtypes = [
    cpgeo_handle_t,                   # handle
    ctypes.POINTER(ctypes.c_int)      # triangles
]
_lib.triangulation_get_data.restype = ctypes.c_int

# sphere_triangulation_compute
_lib.sphere_triangulation_compute.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # sphere_points
    ctypes.c_int,                     # num_points
    ctypes.POINTER(ctypes.c_int)      # num_triangles
]
_lib.sphere_triangulation_compute.restype = cpgeo_handle_t

# sphere_triangulation_get_data
_lib.sphere_triangulation_get_data.argtypes = [
    cpgeo_handle_t,                   # handle
    ctypes.POINTER(ctypes.c_int)      # triangles
]
_lib.sphere_triangulation_get_data.restype = ctypes.c_int

# space_tree_create
_lib.space_tree_create.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # knots
    ctypes.c_int,                     # num_knots
    ctypes.POINTER(ctypes.c_double)   # thresholds
]
_lib.space_tree_create.restype = cpgeo_handle_t

# space_tree_query_compute
_lib.space_tree_query_compute.argtypes = [
    cpgeo_handle_t,                   # handle
    ctypes.POINTER(ctypes.c_double),  # query_points
    ctypes.c_int,                     # num_queries
    ctypes.POINTER(ctypes.c_int)      # total_results
]
_lib.space_tree_query_compute.restype = ctypes.c_int

# space_tree_query_get
_lib.space_tree_query_get.argtypes = [
    cpgeo_handle_t,                   # tree
    ctypes.c_int,                     # num_results
    ctypes.POINTER(ctypes.c_int),     # indices_cps
    ctypes.POINTER(ctypes.c_int)      # indices_pts
]
_lib.space_tree_query_get.restype = ctypes.c_int

# space_tree_destroy
_lib.space_tree_destroy.argtypes = [cpgeo_handle_t]  # handle
_lib.space_tree_destroy.restype = None

# cpgeo_compute_thresholds
_lib.cpgeo_compute_thresholds.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # knots
    ctypes.c_int,                     # num_knots
    ctypes.c_int,                     # k
    ctypes.POINTER(ctypes.c_double)   # out_thresholds
]
_lib.cpgeo_compute_thresholds.restype = None

# cpgeo_get_weights
_lib.cpgeo_get_weights.argtypes = [
    ctypes.POINTER(ctypes.c_int),     # indices_cps
    ctypes.POINTER(ctypes.c_int),     # indices_pts
    ctypes.c_int,                     # num_indices
    ctypes.POINTER(ctypes.c_double),  # knots
    ctypes.c_int,                     # num_knots
    ctypes.POINTER(ctypes.c_double),  # thresholds
    ctypes.POINTER(ctypes.c_double),  # query_points
    ctypes.c_int,                     # num_queries
    ctypes.POINTER(ctypes.c_double)   # out_weights
]
_lib.cpgeo_get_weights.restype = None

# cpgeo_get_weights_derivative1
_lib.cpgeo_get_weights_derivative1.argtypes = [
    ctypes.POINTER(ctypes.c_int),     # indices_cps
    ctypes.POINTER(ctypes.c_int),     # indices_pts
    ctypes.c_int,                     # num_indices
    ctypes.POINTER(ctypes.c_double),  # knots
    ctypes.c_int,                     # num_knots
    ctypes.POINTER(ctypes.c_double),  # thresholds
    ctypes.POINTER(ctypes.c_double),  # query_points
    ctypes.c_int,                     # num_queries
    ctypes.POINTER(ctypes.c_double),  # out_weights
    ctypes.POINTER(ctypes.c_double)   # out_weights_du
]
_lib.cpgeo_get_weights_derivative1.restype = None

# cpgeo_get_weights_derivative2
_lib.cpgeo_get_weights_derivative2.argtypes = [
    ctypes.POINTER(ctypes.c_int),     # indices_cps
    ctypes.POINTER(ctypes.c_int),     # indices_pts
    ctypes.c_int,                     # num_indices
    ctypes.POINTER(ctypes.c_double),  # knots
    ctypes.c_int,                     # num_knots
    ctypes.POINTER(ctypes.c_double),  # thresholds
    ctypes.POINTER(ctypes.c_double),  # query_points
    ctypes.c_int,                     # num_queries
    ctypes.POINTER(ctypes.c_double),  # out_weights
    ctypes.POINTER(ctypes.c_double),  # out_weights_du
    ctypes.POINTER(ctypes.c_double)   # out_weights_du2
]
_lib.cpgeo_get_weights_derivative2.restype = None

# cpgeo_get_mapped_points
_lib.cpgeo_get_mapped_points.argtypes = [
    ctypes.POINTER(ctypes.c_int),     # indices_cps
    ctypes.POINTER(ctypes.c_int),     # indices_pts
    ctypes.c_int,                     # num_indices
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.POINTER(ctypes.c_double),  # controlpoints
    ctypes.c_int,                     # num_controlpoints
    ctypes.c_int,                     # num_queries
    ctypes.POINTER(ctypes.c_double)   # out_mapped_points
]
_lib.cpgeo_get_mapped_points.restype = None

# cpgeo_uniformly_mesh_compute/get
_lib.cpgeo_uniformly_mesh_compute.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # init_vertices_sphere
    ctypes.c_int,                     # num_vertices
    ctypes.POINTER(ctypes.c_double),  # control_points
    ctypes.c_int,                     # num_control_points
    cpgeo_handle_t,                   # tree
    ctypes.c_double,                  # seed_size
    ctypes.c_int,                     # max_iterations
    ctypes.POINTER(ctypes.c_int),     # out_num_vertices
    ctypes.POINTER(ctypes.c_int)      # out_num_faces
]
_lib.cpgeo_uniformly_mesh_compute.restype = ctypes.c_int

_lib.cpgeo_uniformly_mesh_get.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # out_vertices
    ctypes.POINTER(ctypes.c_int)      # out_faces
]
_lib.cpgeo_uniformly_mesh_get.restype = ctypes.c_int

# cpgeo_map_points_derivative2
_lib.cpgeo_map_points_derivative2.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # query_points
    cpgeo_handle_t,                   # tree
    ctypes.POINTER(ctypes.c_double),  # controlpoints
    ctypes.c_int,                     # num_controlpoints
    ctypes.c_int,                     # num_queries
    ctypes.POINTER(ctypes.c_double),  # out_r
    ctypes.POINTER(ctypes.c_double),  # out_rdu
    ctypes.POINTER(ctypes.c_double)   # out_rdu2
]
_lib.cpgeo_map_points_derivative2.restype = None

# cpgeo_map_points (batch default)
_lib.cpgeo_map_points.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # query_points
    cpgeo_handle_t,                   # tree
    ctypes.POINTER(ctypes.c_double),  # controlpoints
    ctypes.c_int,                     # num_controlpoints
    ctypes.c_int,                     # num_queries
    ctypes.POINTER(ctypes.c_double)   # out_mapped_points
]
_lib.cpgeo_map_points.restype = None

# mesh_edges_compute
_lib.mesh_edges_compute.argtypes = [
    ctypes.POINTER(ctypes.c_int),     # elements
    ctypes.c_int,                     # num_elements
    ctypes.POINTER(ctypes.c_int)      # out_num_edges
]
_lib.mesh_edges_compute.restype = None

# mesh_edges_get
_lib.mesh_edges_get.argtypes = [
    ctypes.POINTER(ctypes.c_int)      # out_edges
]
_lib.mesh_edges_get.restype = ctypes.c_int

# mesh_extract_boundary_loops_compute
_lib.mesh_extract_boundary_loops_compute.argtypes = [
    ctypes.POINTER(ctypes.c_int),     # triangles
    ctypes.c_int,                     # num_triangles
    ctypes.POINTER(ctypes.c_int),     # num_boundary_vertices
    ctypes.POINTER(ctypes.c_int)      # num_loops
]
_lib.mesh_extract_boundary_loops_compute.restype = None

# mesh_extract_boundary_loops_get
_lib.mesh_extract_boundary_loops_get.argtypes = [
    ctypes.POINTER(ctypes.c_int),     # out_boundary_vertices
    ctypes.POINTER(ctypes.c_int)      # out_loop_indices
]
_lib.mesh_extract_boundary_loops_get.restype = ctypes.c_int

# mesh_closure_edge_length_derivative0
_lib.mesh_closure_edge_length_derivative0.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # vertices
    ctypes.c_int,                     # num_vertices
    ctypes.c_int,                     # vertices_dim
    ctypes.POINTER(ctypes.c_int),     # edges
    ctypes.c_int,                     # num_edges
    ctypes.c_int,                     # order
    ctypes.POINTER(ctypes.c_double)   # out_loss
]
_lib.mesh_closure_edge_length_derivative0.restype = None

# mesh_closure_edge_length_derivative2_compute
_lib.mesh_closure_edge_length_derivative2_compute.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # vertices
    ctypes.c_int,                     # num_vertices
    ctypes.c_int,                     # vertices_dim
    ctypes.POINTER(ctypes.c_int),     # edges
    ctypes.c_int,                     # num_edges
    ctypes.c_int,                     # order
    ctypes.POINTER(ctypes.c_int)      # num_out_ldr2
]
_lib.mesh_closure_edge_length_derivative2_compute.restype = None

# mesh_closure_edge_length_derivative2_get
_lib.mesh_closure_edge_length_derivative2_get.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # out_loss
    ctypes.POINTER(ctypes.c_double),  # out_ldr
    ctypes.POINTER(ctypes.c_int),     # out_ldr2_indices
    ctypes.POINTER(ctypes.c_double)   # out_ldr2_values
]
_lib.mesh_closure_edge_length_derivative2_get.restype = ctypes.c_int

# mesh_optimize_by_edge_flipping
_lib.mesh_optimize_by_edge_flipping.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # vertices
    ctypes.c_int,                     # num_vertices
    ctypes.c_int,                     # vertices_dim
    ctypes.POINTER(ctypes.c_int),     # faces_in
    ctypes.c_int,                     # num_faces
    ctypes.c_int,                     # max_iterations
    ctypes.POINTER(ctypes.c_int)      # out_faces
]
_lib.mesh_optimize_by_edge_flipping.restype = None

# mesh_free_loop_sizes
# _lib.mesh_free_loop_sizes.argtypes = [
#     ctypes.POINTER(ctypes.c_int)      # loop_sizes
# ]
# _lib.mesh_free_loop_sizes.restype = None


# ==================== Python Wrapper Functions ====================

def get_triangulation(nodes: np.ndarray) -> np.ndarray:
    """Compute Delaunay triangulation.
    
    Args:
        nodes: Flat array of node coordinates [x0, y0, x1, y1, ..., xn, yn]
    
    Returns:
        np.ndarray: Triangle indices array (shape: [num_triangles, 3])
    """
    if nodes.ndim != 1:
        raise ValueError("nodes must be a 1D array")
    
    num_nodes = len(nodes) // 2
    nodes_flat = np.ascontiguousarray(nodes, dtype=np.float64)
    nodes_ptr = nodes_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    num_triangles = ctypes.c_int()
    
    handle = _lib.triangulation_compute(
        nodes_ptr,
        ctypes.c_int(num_nodes),
        ctypes.byref(num_triangles)
    )
    
    triangles = np.zeros(num_triangles.value * 3, dtype=np.int32)
    triangles_ptr = triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    result = _lib.triangulation_get_data(handle, triangles_ptr)
    
    return triangles.reshape(-1, 3)


def get_sphere_triangulation(sphere_points: np.ndarray) -> np.ndarray:
    """Compute sphere triangulation.
    
    Args:
        sphere_points: Flat array of 3D point coordinates [x0, y0, z0, ..., xn, yn, zn]
    
    Returns:
        np.ndarray: Triangle indices array (shape: [num_triangles, 3])
    """
    sphere_points = sphere_points.flatten()
    if sphere_points.ndim != 1 or len(sphere_points) % 3 != 0:
        raise ValueError("sphere_points must be a 1D array with length multiple of 3")
    
    num_points = len(sphere_points) // 3
    points_flat = np.ascontiguousarray(sphere_points, dtype=np.float64)
    points_ptr = points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    num_triangles = ctypes.c_int()
    
    handle = _lib.sphere_triangulation_compute(
        points_ptr,
        ctypes.c_int(num_points),
        ctypes.byref(num_triangles)
    )
    
    triangles = np.zeros(num_triangles.value * 3, dtype=np.int32)
    triangles_ptr = triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    result = _lib.sphere_triangulation_get_data(handle, triangles_ptr)
    
    return triangles.reshape(-1, 3)


def space_tree_create(knots: np.ndarray, thresholds: np.ndarray) -> cpgeo_handle_t:
    """Create a SpaceTree for spatial queries.
    
    Args:
        knots (np.ndarray): Array of 3D knot coordinates [x0, y0, z0, x1, y1, z1, ...]
        thresholds (np.ndarray): Array of influence radii for each knot
    
    Returns:
        cpgeo_handle_t: Handle to the SpaceTree
    """

    knots = knots.flatten()

    if len(thresholds) != len(knots) // 3:
        raise ValueError("thresholds length must match number of knots")
    
    

    num_knots = len(knots) // 3
    knots_flat = np.ascontiguousarray(knots, dtype=np.float64)
    knots_ptr = knots_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    thresholds_flat = np.ascontiguousarray(thresholds, dtype=np.float64)
    thresholds_ptr = thresholds_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    handle = _lib.space_tree_create(
        knots_ptr,
        ctypes.c_int(num_knots),
        thresholds_ptr
    )
    
    return handle


def get_space_tree_query(handle: cpgeo_handle_t, query_points_sphere: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SpaceTree queries for points within influence radii.
    
    Args:
        handle: SpaceTree handle
        query_points_sphere: Flat array of 3D query coordinates [x0, y0, z0, x1, y1, z1, ...]
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: indices_cps and indices_pts arrays
    """
    query_points_sphere = query_points_sphere.flatten()
    num_queries = len(query_points_sphere) // 3
    points_flat = np.ascontiguousarray(query_points_sphere, dtype=np.float64)
    points_ptr = points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    total_results = ctypes.c_int()
    
    result = _lib.space_tree_query_compute(
        handle,
        points_ptr,
        ctypes.c_int(num_queries),
        ctypes.byref(total_results)
    )
    
    indices_cps = np.zeros(total_results.value, dtype=np.int32)
    indices_cps_ptr = indices_cps.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    indices_pts = np.zeros(num_queries + 1, dtype=np.int32)  # Note: size is num_queries + 1
    indices_pts_ptr = indices_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    result = _lib.space_tree_query_get(
        handle,
        ctypes.c_int(total_results.value),
        indices_cps_ptr,
        indices_pts_ptr
    )
    
    return indices_cps, indices_pts


def space_tree_destroy(handle: cpgeo_handle_t) -> None:
    """Destroy the SpaceTree and free resources.
    
    Args:
        handle: SpaceTree handle to destroy
    """
    _lib.space_tree_destroy(handle)


def compute_thresholds(knots: np.ndarray, k: int) -> np.ndarray:
    """Compute influence thresholds for knots based on k-nearest neighbors.
    
    Args:
        knots: Flat array of 3D knot coordinates [x0, y0, z0, x1, y1, z1, ...]
        k: The k-nearest neighbor to consider
    
    Returns:
        np.ndarray: Computed thresholds array
    """
    knots = knots.flatten()
    
    num_knots = len(knots) // 3
    knots_flat = np.ascontiguousarray(knots, dtype=np.float64)
    knots_ptr = knots_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    out_thresholds = np.zeros(num_knots, dtype=np.float64)
    out_thresholds_ptr = out_thresholds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    _lib.cpgeo_compute_thresholds(
        knots_ptr,
        ctypes.c_int(num_knots),
        ctypes.c_int(k),
        out_thresholds_ptr
    )
    
    return out_thresholds


def get_weights(indices_cps: np.ndarray, indices_pts: np.ndarray, knots: np.ndarray, 
                thresholds: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """Compute weights for query points based on knot influences.
    
    Args:
        indices_cps: Array of knot point indices
        indices_pts: Array of query point start indices
        knots: Flat array of 3D knot coordinates
        thresholds: Array of influence radii
        query_points: Flat array of 2D query coordinates [u0, v0, u1, v1, ...]
    
    Returns:
        np.ndarray: Computed weights array
    """
    knots = knots.flatten()
    query_points = query_points.flatten()

    num_indices = len(indices_cps)
    num_knots = len(knots) // 3
    num_queries = len(query_points) // 2
    
    # Convert inputs to ctypes
    indices_cps_flat = np.ascontiguousarray(indices_cps, dtype=np.int32)
    indices_cps_ptr = indices_cps_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    indices_pts_flat = np.ascontiguousarray(indices_pts, dtype=np.int32)
    indices_pts_ptr = indices_pts_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    knots_flat = np.ascontiguousarray(knots, dtype=np.float64)
    knots_ptr = knots_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    thresholds_flat = np.ascontiguousarray(thresholds, dtype=np.float64)
    thresholds_ptr = thresholds_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    query_points_flat = np.ascontiguousarray(query_points, dtype=np.float64)
    query_points_ptr = query_points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    out_weights = np.zeros(num_indices, dtype=np.float64)
    out_weights_ptr = out_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    _lib.cpgeo_get_weights(
        indices_cps_ptr,
        indices_pts_ptr,
        ctypes.c_int(num_indices),
        knots_ptr,
        ctypes.c_int(num_knots),
        thresholds_ptr,
        query_points_ptr,
        ctypes.c_int(num_queries),
        out_weights_ptr
    )
    
    return out_weights


def get_weights_derivative1(indices_cps: np.ndarray, indices_pts: np.ndarray, knots: np.ndarray,
                           thresholds: np.ndarray, query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute weights and first derivatives for query points.
    
    Args:
        indices_cps: Array of knot point indices
        indices_pts: Array of query point start indices
        knots: Flat array of 3D knot coordinates
        thresholds: Array of influence radii
        query_points: Flat array of 2D query coordinates [u0, v0, u1, v1, ...]
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Weights and first derivatives arrays
    """
    knots = knots.flatten()
    query_points = query_points.flatten()

    num_indices = len(indices_cps)
    num_knots = len(knots) // 3
    num_queries = len(query_points) // 2
    
    # Convert inputs to ctypes (similar to get_weights)
    indices_cps_flat = np.ascontiguousarray(indices_cps, dtype=np.int32)
    indices_cps_ptr = indices_cps_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    indices_pts_flat = np.ascontiguousarray(indices_pts, dtype=np.int32)
    indices_pts_ptr = indices_pts_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    knots_flat = np.ascontiguousarray(knots, dtype=np.float64)
    knots_ptr = knots_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    thresholds_flat = np.ascontiguousarray(thresholds, dtype=np.float64)
    thresholds_ptr = thresholds_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    query_points_flat = np.ascontiguousarray(query_points, dtype=np.float64)
    query_points_ptr = query_points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    out_weights = np.zeros(num_indices, dtype=np.float64)
    out_weights_ptr = out_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # First derivatives stored with layout [2, num_indices]
    out_weights_du = np.zeros((2, num_indices), dtype=np.float64)
    out_weights_du_ptr = out_weights_du.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    _lib.cpgeo_get_weights_derivative1(
        indices_cps_ptr,
        indices_pts_ptr,
        ctypes.c_int(num_indices),
        knots_ptr,
        ctypes.c_int(num_knots),
        thresholds_ptr,
        query_points_ptr,
        ctypes.c_int(num_queries),
        out_weights_ptr,
        out_weights_du_ptr
    )
    
    return out_weights, out_weights_du


def get_weights_derivative2(indices_cps: np.ndarray, indices_pts: np.ndarray, knots: np.ndarray,
                           thresholds: np.ndarray, query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute weights and first/second derivatives for query points.
    
    Args:
        indices_cps: Array of knot point indices
        indices_pts: Array of query point start indices
        knots: Flat array of 3D knot coordinates
        thresholds: Array of influence radii
        query_points: Flat array of 2D query coordinates [u0, v0, u1, v1, ...]
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Weights, first derivatives, and second derivatives arrays
            - shape of first derivatives: [2, num_indices]
            - shape of second derivatives: [2, 2, num_indices]
    """
    knots = knots.flatten()
    query_points = query_points.flatten()

    num_indices = len(indices_cps)
    num_knots = len(knots) // 3
    num_queries = len(query_points) // 2
    
    # Convert inputs to ctypes (similar to above)
    indices_cps_flat = np.ascontiguousarray(indices_cps, dtype=np.int32)
    indices_cps_ptr = indices_cps_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    indices_pts_flat = np.ascontiguousarray(indices_pts, dtype=np.int32)
    indices_pts_ptr = indices_pts_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    knots_flat = np.ascontiguousarray(knots, dtype=np.float64)
    knots_ptr = knots_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    thresholds_flat = np.ascontiguousarray(thresholds, dtype=np.float64)
    thresholds_ptr = thresholds_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    query_points_flat = np.ascontiguousarray(query_points, dtype=np.float64)
    query_points_ptr = query_points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    out_weights = np.zeros(num_indices, dtype=np.float64)
    out_weights_ptr = out_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # First derivatives stored with layout [2, num_indices]
    out_weights_du = np.zeros((2, num_indices), dtype=np.float64)
    out_weights_du_ptr = out_weights_du.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # Second derivatives stored with layout [2, 2, num_indices]
    out_weights_du2 = np.zeros((2, 2, num_indices), dtype=np.float64)
    out_weights_du2_ptr = out_weights_du2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    _lib.cpgeo_get_weights_derivative2(
        indices_cps_ptr,
        indices_pts_ptr,
        ctypes.c_int(num_indices),
        knots_ptr,
        ctypes.c_int(num_knots),
        thresholds_ptr,
        query_points_ptr,
        ctypes.c_int(num_queries),
        out_weights_ptr,
        out_weights_du_ptr,
        out_weights_du2_ptr
    )
    
    return out_weights, out_weights_du, out_weights_du2


def get_mapped_points(indices_cps: np.ndarray, indices_pts: np.ndarray,
                         weights: np.ndarray, controlpoints: np.ndarray,
                         num_queries: int) -> np.ndarray:
    """Compute mapped points from control points using sparse weights.
    
    Args:
        indices_cps: Array of control point indices for each weight
        indices_pts: Array of starting indices for each query point
        weights: Array of weight values
        controlpoints: 3D control point coordinates (num_controlpoints, 3)
        num_queries: Number of query points
        
    Returns:
        Mapped coordinates array of shape (num_queries, 3)
    """
    num_indices = len(weights)
    num_controlpoints = controlpoints.shape[0]
    
    # Prepare input arrays
    indices_cps = np.ascontiguousarray(indices_cps, dtype=np.int32)
    indices_pts = np.ascontiguousarray(indices_pts, dtype=np.int32)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    controlpoints = np.ascontiguousarray(controlpoints, dtype=np.float64).flatten()
    
    # Prepare output array
    out_mapped_points = np.zeros((num_queries, 3), dtype=np.float64).flatten()
    
    # Get ctypes pointers
    indices_cps_ptr = indices_cps.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    indices_pts_ptr = indices_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    controlpoints_ptr = controlpoints.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_mapped_points_ptr = out_mapped_points.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    _lib.cpgeo_get_mapped_points(
        indices_cps_ptr,
        indices_pts_ptr,
        ctypes.c_int(num_indices),
        weights_ptr,
        controlpoints_ptr,
        ctypes.c_int(num_controlpoints),
        ctypes.c_int(num_queries),
        out_mapped_points_ptr
    )
    
    return out_mapped_points.reshape((num_queries, 3))


def map_points_derivative2(query_points: np.ndarray, tree_handle: cpgeo_handle_t, controlpoints: np.ndarray):
    """Compute mapped points and derivatives for batch of queries using a SpaceTree handle.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (r, rdu, rdu2) with shapes
            - r: (num_queries, 3)
            - rdu: (num_queries, 2, 3)
            - rdu2: (num_queries, 2, 2, 3)
    """
    query_points = query_points.flatten()
    if len(query_points) % 3 != 0:
        raise ValueError("query_points length must be multiple of 3")

    num_queries = len(query_points) // 3
    num_controlpoints = controlpoints.shape[0]

    query_points_flat = np.ascontiguousarray(query_points, dtype=np.float64)
    query_points_ptr = query_points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    controlpoints_flat = np.ascontiguousarray(controlpoints, dtype=np.float64).flatten()
    controlpoints_ptr = controlpoints_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out_r = np.zeros(num_queries * 3, dtype=np.float64)
    out_rdu = np.zeros(num_queries * 2 * 3, dtype=np.float64)
    out_rdu2 = np.zeros(num_queries * 2 * 2 * 3, dtype=np.float64)

    _lib.cpgeo_map_points_derivative2(
        query_points_ptr,
        tree_handle,
        controlpoints_ptr,
        ctypes.c_int(num_controlpoints),
        ctypes.c_int(num_queries),
        out_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_rdu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_rdu2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    return out_r.reshape((num_queries, 3)), out_rdu.reshape((num_queries, 2, 3)), out_rdu2.reshape((num_queries, 2, 2, 3))


def map_points(query_points: np.ndarray, tree_handle: cpgeo_handle_t, controlpoints: np.ndarray):
    """Compute mapped points for batch queries using a SpaceTree handle.

    Returns:
        np.ndarray: mapped points with shape (num_queries, 3)
    """
    query_points = query_points.flatten()
    if len(query_points) % 3 != 0:
        raise ValueError("query_points length must be multiple of 3")

    num_queries = len(query_points) // 3
    num_controlpoints = controlpoints.shape[0]

    query_points_flat = np.ascontiguousarray(query_points, dtype=np.float64)
    query_points_ptr = query_points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    controlpoints_flat = np.ascontiguousarray(controlpoints, dtype=np.float64).flatten()
    controlpoints_ptr = controlpoints_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out_r = np.zeros(num_queries * 3, dtype=np.float64)

    _lib.cpgeo_map_points(
        query_points_ptr,
        tree_handle,
        controlpoints_ptr,
        ctypes.c_int(num_controlpoints),
        ctypes.c_int(num_queries),
        out_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    return out_r.reshape((num_queries, 3))


def get_mesh_edges(elements: np.ndarray) -> np.ndarray:
    """Compute mesh edges from triangular elements.
    
    Args:
        elements: Array of triangle vertex indices (shape: [num_elements, 3])
    
    Returns:
        np.ndarray: Array of edge vertex indices and counts (shape: [num_edges, 3])
                   Each edge is represented by (vertex1, vertex2, count)
    """
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("elements must be a 2D array with shape [num_elements, 3]")
    
    elements_flat = np.ascontiguousarray(elements.flatten(), dtype=np.int32)
    elements_ptr = elements_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    num_elements = elements.shape[0]
    out_num_edges = ctypes.c_int()
    
    _lib.mesh_edges_compute(
        elements_ptr,
        ctypes.c_int(num_elements),
        ctypes.byref(out_num_edges)
    )
    
    out_edges = np.zeros(out_num_edges.value * 3, dtype=np.int32)
    out_edges_ptr = out_edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    result = _lib.mesh_edges_get(out_edges_ptr)
    
    return out_edges.reshape(-1, 3)


def uniformly_mesh(init_vertices: np.ndarray, control_points: np.ndarray, tree_handle: cpgeo_handle_t, seed_size: float = 1.0, max_iterations: int = 10):
    """Uniformly remesh a sphere surface starting from initial vertices.

    Args:
        init_vertices: Initial sphere vertices as array shape (n, 3) or flat (3n,)
        control_points: Control points array shape (m, 3)
        tree_handle: SpaceTree handle
        seed_size: desired seed size
        max_iterations: max uniforming iterations

    Returns:
        Tuple[np.ndarray, np.ndarray]: (vertices (num_vertices,3), faces (num_faces,3))
    """
    init_flat = np.ascontiguousarray(init_vertices.flatten(), dtype=np.float64)
    control_flat = np.ascontiguousarray(control_points.flatten(), dtype=np.float64)

    num_vertices = len(init_flat) // 3
    num_control = len(control_flat) // 3

    # First call compute to get output sizes
    out_num_vertices = ctypes.c_int()
    out_num_faces = ctypes.c_int()

    _lib.cpgeo_uniformly_mesh_compute(
        init_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(num_vertices),
        control_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(num_control),
        tree_handle,
        ctypes.c_double(seed_size),
        ctypes.c_int(max_iterations),
        ctypes.byref(out_num_vertices),
        ctypes.byref(out_num_faces)
    )

    if out_num_vertices.value <= 0 or out_num_faces.value <= 0:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.int32)

    out_vertices = np.zeros(out_num_vertices.value * 3, dtype=np.float64)
    out_faces = np.zeros(out_num_faces.value * 3, dtype=np.int32)

    # Call get to fetch data
    _lib.cpgeo_uniformly_mesh_get(
        out_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    return out_vertices.reshape((out_num_vertices.value, 3)), out_faces.reshape((out_num_faces.value, 3))


def extract_boundary_loops(triangles: np.ndarray) -> List[np.ndarray]:
    """Extract all boundary loops from a triangular mesh.
    
    A boundary edge is an edge that belongs to only one triangle.
    This function finds all closed boundary loops and returns them as ordered vertex sequences.
    
    Args:
        triangles: Array of triangle vertex indices (shape: [num_triangles, 3])
    
    Returns:
        List[np.ndarray]: List of boundary loops, where each loop is a 1D array of vertex indices
                         forming a closed path (first and last vertices are not duplicated)
    """
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be a 2D array with shape [num_triangles, 3]")
    
    triangles_flat = np.ascontiguousarray(triangles.flatten(), dtype=np.int32)
    triangles_ptr = triangles_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    num_triangles = triangles.shape[0]
    out_num_boundary_vertices = ctypes.c_int()
    out_num_loops = ctypes.c_int()
    
    _lib.mesh_extract_boundary_loops_compute(
        triangles_ptr,
        ctypes.c_int(num_triangles),
        ctypes.byref(out_num_boundary_vertices),
        ctypes.byref(out_num_loops)
    )
    
    if out_num_loops.value == 0:
        return []
    
    # Get all loop vertices and indices
    all_vertices = np.zeros(out_num_boundary_vertices.value, dtype=np.int32)
    all_vertices_ptr = all_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    out_loop_indices = np.zeros(out_num_loops.value + 1, dtype=np.int32)
    out_loop_indices_ptr = out_loop_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    result = _lib.mesh_extract_boundary_loops_get(
        all_vertices_ptr,
        out_loop_indices_ptr
    )
    
    # Parse loops
    loops = []
    for i in range(out_num_loops.value):
        start = out_loop_indices[i]
        end = out_loop_indices[i + 1]
        loops.append(all_vertices[start:end])
    
    return loops


def get_mesh_closure_edge_length_derivative0(vertices: np.ndarray, edges: np.ndarray, order: int = 2) -> float:
    """Compute closure edge length loss.
    
    Args:
        vertices: Array of vertex coordinates (shape: [num_vertices, vertices_dim])
        edges: Array of edge vertex indices (shape: [num_edges, 2])
    
    Returns:
        float: The computed loss value
    """
    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D array with shape [num_vertices, vertices_dim]")
    
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be a 2D array with shape [num_edges, 2]")
    
    num_vertices, vertices_dim = vertices.shape
    num_edges = edges.shape[0]
    
    vertices_flat = np.ascontiguousarray(vertices.flatten(), dtype=np.float64)
    vertices_ptr = vertices_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    edges_flat = np.ascontiguousarray(edges.flatten(), dtype=np.int32)
    edges_ptr = edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    out_loss = np.zeros(1, dtype=np.float64)
    out_loss_ptr = out_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    _lib.mesh_closure_edge_length_derivative0(
        vertices_ptr,
        ctypes.c_int(num_vertices),
        ctypes.c_int(vertices_dim),
        edges_ptr,
        ctypes.c_int(num_edges),
        ctypes.c_int(order),
        out_loss_ptr
    )
    
    return out_loss[0]


def get_mesh_closure_edge_length_derivative2(vertices: np.ndarray, edges: np.ndarray, order: int = 2) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute closure edge length loss and its second derivative.
    
    Args:
        vertices: Array of vertex coordinates (shape: [num_vertices, vertices_dim])
        edges: Array of edge vertex indices (shape: [num_edges, 2])
    
    Returns:
        Tuple[float, np.ndarray, np.ndarray, np.ndarray]: 
            - loss: The computed loss value
            - ldr: First derivative array (shape: [num_vertices, vertices_dim])
            - ldr2_indices: Second derivative indices in COO format (shape: [num_out_ldr2, 4])
            - ldr2_values: Second derivative values in COO format (shape: [num_out_ldr2])
    """
    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D array with shape [num_vertices, vertices_dim]")
    
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be a 2D array with shape [num_edges, 2]")
    
    num_vertices, vertices_dim = vertices.shape
    num_edges = edges.shape[0]
    
    vertices_flat = np.ascontiguousarray(vertices.flatten(), dtype=np.float64)
    vertices_ptr = vertices_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    edges_flat = np.ascontiguousarray(edges.flatten(), dtype=np.int32)
    edges_ptr = edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    num_out_ldr2 = ctypes.c_int()
    
    _lib.mesh_closure_edge_length_derivative2_compute(
        vertices_ptr,
        ctypes.c_int(num_vertices),
        ctypes.c_int(vertices_dim),
        edges_ptr,
        ctypes.c_int(num_edges),
        ctypes.c_int(order),
        ctypes.byref(num_out_ldr2)
    )
    
    out_loss = np.zeros(1, dtype=np.float64)
    out_loss_ptr = out_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    out_ldr = np.zeros(num_vertices * vertices_dim, dtype=np.float64)
    out_ldr_ptr = out_ldr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    out_ldr2_indices = np.zeros(num_out_ldr2.value * 4, dtype=np.int32)
    out_ldr2_indices_ptr = out_ldr2_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    out_ldr2_values = np.zeros(num_out_ldr2.value, dtype=np.float64)
    out_ldr2_values_ptr = out_ldr2_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    result = _lib.mesh_closure_edge_length_derivative2_get(
        out_loss_ptr,
        out_ldr_ptr,
        out_ldr2_indices_ptr,
        out_ldr2_values_ptr
    )
    
    return out_loss[0], out_ldr.reshape(num_vertices, vertices_dim), out_ldr2_indices.reshape(-1, 4), out_ldr2_values


def optimize_mesh_by_edge_flipping(vertices: np.ndarray, faces: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Optimize mesh by edge flipping to make triangles more equilateral.
    
    Args:
        vertices: Vertex coordinates array (shape: [num_vertices, vertices_dim])
        faces: Triangle faces array (shape: [num_faces, 3])
        max_iterations: Maximum number of optimization iterations (default: 100)
    
    Returns:
        np.ndarray: Optimized triangle faces array (shape: [num_faces, 3])
    """
    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D array with shape [num_vertices, vertices_dim]")
    
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be a 2D array with shape [num_faces, 3]")
    
    num_vertices, vertices_dim = vertices.shape
    num_faces = faces.shape[0]
    
    vertices_flat = np.ascontiguousarray(vertices.flatten(), dtype=np.float64)
    vertices_ptr = vertices_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    faces_flat = np.ascontiguousarray(faces.flatten(), dtype=np.int32)
    faces_ptr = faces_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    out_faces = np.zeros(num_faces * 3, dtype=np.int32)
    out_faces_ptr = out_faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    _lib.mesh_optimize_by_edge_flipping(
        vertices_ptr,
        ctypes.c_int(num_vertices),
        ctypes.c_int(vertices_dim),
        faces_ptr,
        ctypes.c_int(num_faces),
        ctypes.c_int(max_iterations),
        out_faces_ptr
    )
    
    return out_faces.reshape(-1, 3)


__all__ = [
    'get_triangulation',
    'get_sphere_triangulation',
    'space_tree_create',
    'get_space_tree_query',
    'space_tree_destroy',
    'compute_thresholds',
    'get_weights',
    'get_weights_derivative1',
    'get_weights_derivative2',
    'get_mapped_points',
    'map_points',
    'map_points_derivative2',
    'get_mesh_edges',
    'extract_boundary_loops',
    'get_mesh_closure_edge_length_derivative0',
    'get_mesh_closure_edge_length_derivative2',
    'optimize_mesh_by_edge_flipping',
]
