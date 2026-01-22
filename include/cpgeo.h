#pragma once

#ifdef _WIN32
    #ifdef cpgeo_EXPORTS
        #define CPGEO_API __declspec(dllexport)
    #else
        #define CPGEO_API __declspec(dllimport)
    #endif
#else
    #define CPGEO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Handle type for opaque CPGEO objects
 */
typedef void* cpgeo_handle_t;

/**
 * @brief Phase 1: Compute Delaunay triangulation
 * 
 * Performs the triangulation and returns a handle to the results.
 * 
 * @param nodes Input array of node coordinates [x0, y0, x1, y1, ..., xn, yn]
 * @param num_nodes Number of nodes (points)
 * @param num_triangles Output pointer to receive the number of generated triangles
 * @return cpgeo_handle_t Opaque handle to the triangulation result, or NULL on failure
 */
CPGEO_API cpgeo_handle_t triangulation_compute(
    const double* nodes,
    int num_nodes,
    int* num_triangles
);

/**
 * @brief Phase 2: Retrieve triangulation results
 * 
 * Copies the triangle indices from the handle to the provided buffer.
 * The handle is automatically destroyed after successful data retrieval.
 * 
 * @param handle The handle returned by triangulation_compute
 * @param triangles Output buffer for triangle indices (must be size num_triangles * 3)
 * @return 0 on success, non-zero on error
 */
CPGEO_API int triangulation_get_data(
    cpgeo_handle_t handle,
    int* triangles
);

/**
 * @brief Phase 1: Compute Sphere triangulation
 * 
 * Performs the triangulation on a sphere and returns a handle to the results.
 * 
 * @param sphere_points Input array of 3D point coordinates [x0, y0, z0, ..., xn, yn, zn]
 * @param num_points Number of points
 * @param num_triangles Output pointer to receive the number of generated triangles
 * @return cpgeo_handle_t Opaque handle to the triangulation result, or NULL on failure
 */
CPGEO_API cpgeo_handle_t sphere_triangulation_compute(
    const double* sphere_points,
    int num_points,
    int* num_triangles
);

/**
 * @brief Phase 2: Retrieve sphere triangulation results
 * 
 * Copies the triangle indices from the handle to the provided buffer.
 * The handle is automatically destroyed after successful data retrieval.
 * 
 * @param handle The handle returned by sphere_triangulation_compute
 * @param triangles Output buffer for triangle indices (must be size num_triangles * 3)
 * @return 0 on success, non-zero on error
 */
CPGEO_API int sphere_triangulation_get_data(
    cpgeo_handle_t handle,
    int* triangles
);

/**
 * @brief Create a SpaceTree for spatial queries
 * 
 * @param knots Flat array of 3D coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_knots Number of knots (points)
 * @param thresholds Array of influence radii for each knot
 * @return cpgeo_handle_t Opaque handle to the SpaceTree, or NULL on failure
 */
CPGEO_API cpgeo_handle_t space_tree_create(
    const double* knots,
    int num_knots,
    const double* thresholds
);

/**
 * @brief Compute SpaceTree queries for points within influence radii
 * 
 * @param handle The SpaceTree handle
 * @param query_points Flat array of 3D query coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_queries Number of query points
 * @param num_results Output array to store the number of results for each query (size: num_queries)
 * @param total_results Output pointer to store the total number of results
 * @return 0 on success, non-zero on error
 */
CPGEO_API int space_tree_query_compute(
    cpgeo_handle_t handle,
    const double* query_points,
    int num_queries,
    int* total_results
);

/**
 * @brief Retrieve SpaceTree query results
 * 
 * @param handle The SpaceTree handle
 * @param num_results Total number of results from query_compute
 * @param indices_cps Output array for knot point indices (size: num_results)
 * @param indices_pts Output array for query point indices (size: num_pts + 1), where indices_pts[i] gives the start index in indices_cps for query i
 * @return 0 on success, non-zero on error
 */
CPGEO_API int space_tree_query_get(
    cpgeo_handle_t tree,
    int num_results,
    int* indices_cps,
    int* indices_pts
);

/**
 * @brief Destroy the SpaceTree and free resources
 * 
 * @param handle The SpaceTree handle to destroy
 */
CPGEO_API void space_tree_destroy(
    cpgeo_handle_t handle
);

/**
 * @brief Compute influence thresholds for knots based on k-nearest neighbors
 * 
 * @param knots Flat array of 3D knot coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_knots Number of knots (points)
 * @param k The k-nearest neighbor to consider
 * @param out_thresholds Output array to store computed thresholds (size: num_knots)
 */
CPGEO_API void cpgeo_compute_thresholds(
    const double* knots,
    int num_knots,
    int k,
    double* out_thresholds
);

/**
 * @brief Compute weights for query points based on knot influences
 * 
 * @param indices Flat array of (query_idx, knot_idx) pairs (size: num_indices * 2)
 * @param num_indices Number of index pairs
 * @param knots Flat array of 3D knot coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_knots Number of knots (points)
 * @param thresholds Array of influence radii for each knot
 * @param query_points Flat array of 3D query coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_queries Number of query points
 * @param out_weights Output array to store computed weights (size: num_indices)
 */
CPGEO_API void cpgeo_get_weights(
    const int* indices_cps,
    const int* indices_pts,
    int num_indices,
    const double* knots,
    int num_knots,
    const double* thresholds,
    const double* query_points,
    int num_queries,
    double* out_weights
);

/**
 * @brief Compute weights and first derivatives for query points based on knot influences
 * 
 * @param indices Flat array of (query_idx, knot_idx) pairs (size: num_indices * 2)
 * @param num_indices Number of index pairs
 * @param knots Flat array of 3D knot coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_knots Number of knots (points)
 * @param thresholds Array of influence radii for each knot
 * @param query_points Flat array of 3D query coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_queries Number of query points
 * @param out_weights Output array to store computed weights (size: num_indices)
 * @param out_weights_du Output array to store first derivatives (size: num_indices * 2)
 */
CPGEO_API void cpgeo_get_weights_derivative1(
    const int* indices_cps,
    const int* indices_pts,
    int num_indices,
    const double* knots,
    int num_knots,
    const double* thresholds,
    const double* query_points,
    int num_queries,
    double* out_weights,
    double* out_weights_du
);

/**
 * @brief Compute weights and first derivatives for query points based on knot influences
 *
 * @param indices Flat array of (query_idx, knot_idx) pairs (size: num_indices * 2)
 * @param num_indices Number of index pairs
 * @param knots Flat array of 3D knot coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_knots Number of knots (points)
 * @param thresholds Array of influence radii for each knot
 * @param query_points Flat array of 3D query coordinates [x0, y0, z0, x1, y1, z1, ...]
 * @param num_queries Number of query points
 * @param out_weights Output array to store computed weights (size: num_indices)
 * @param out_weights_du Output array to store first derivatives (size: num_indices * 2)
 * @param out_weights_du2 Output array to store second derivatives (size: num_indices * 2 * 2)
 */
CPGEO_API void cpgeo_get_weights_derivative2(
    const int* indices_cps,
    const int* indices_pts,
    int num_indices,
    const double* knots,
    int num_knots,
    const double* thresholds,
    const double* query_points,
    int num_queries,
    double* out_weights,
    double* out_weights_du,
    double* out_weights_du2
);

/**
 * @brief Compute mapped points from control points using sparse weights
 *
 * This function computes the final mapped coordinates by applying weights to control points.
 * It performs the sparse matrix multiplication: mapped_points[i] = sum(weights[j] * controlpoints[indices_cps[j]])
 * for all j where indices_pts[i] <= j < indices_pts[i+1]
 *
 * @param indices_cps Array of control point indices for each weight (size: num_indices)
 * @param indices_pts Array of starting indices for each query point (size: num_queries + 1)
 * @param num_indices Total number of weight entries
 * @param weights Array of weight values (size: num_indices)
 * @param controlpoints Flat array of 3D control point coordinates [x0,y0,z0,x1,y1,z1,...]
 * @param num_controlpoints Number of control points
 * @param num_queries Number of query points
 * @param out_mapped_points Output array for mapped coordinates (size: num_queries * 3)
 */
CPGEO_API void cpgeo_get_mapped_points(
    const int* indices_cps,
    const int* indices_pts,
    int num_indices,
    const double* weights,
    const double* controlpoints,
    int num_controlpoints,
    int num_queries,
    double* out_mapped_points
);


// ==============================================================================================================
// ================== Mesh Utilities ============================================================================
// ==============================================================================================================


/**
 * @brief Compute mesh edges from triangular elements
 * 
 * @param elements Input array of triangle vertex indices (size: num_elements * 3)
 * @param num_elements Number of triangular elements
 * @param out_num_edges Output pointer to receive the number of unique edges
 */
CPGEO_API void mesh_edges_compute(
        const int* elements,
        int num_elements,
        int* out_num_edges
);

/**
 * @brief Retrieve computed mesh edges
 * 
 * @param out_edges Output array to store edge vertex indices and counts (size: out_num_edges * 3)
 *                   Each edge is represented by (vertex1, vertex2, count)
 * @return 0 on success, non-zero on error
 */
CPGEO_API int mesh_edges_get(
    int* out_edges
);

/**
 * @brief Partition a sphere-like mesh into two disk-like hemispheres
 * 
 * Divides a closed triangular mesh (topologically equivalent to a sphere) into two regions,
 * each topologically equivalent to a disk. The partition attempts to balance the number of
 * faces in each hemisphere.
 * 
 * @param triangles Input array of triangle vertex indices (size: num_triangles * 3)
 * @param num_triangles Number of triangular faces
 * @param num_vertices Total number of vertices in the mesh
 * @param out_num_hemisphere1 Output pointer to receive number of triangles in first hemisphere
 * @param out_num_hemisphere2 Output pointer to receive number of triangles in second hemisphere
 * @param out_num_cut_vertices Output pointer to receive number of vertices on cutting boundary
 */

/**
 * @brief Extract all boundary loops from a triangular mesh
 * 
 * A boundary edge is an edge that belongs to only one triangle.
 * This function finds all closed boundary loops and returns them as ordered vertex sequences.
 * 
 * @param triangles Input array of triangle vertex indices (size: num_triangles * 3)
 * @param num_triangles Number of triangular faces
 * @param num_boundary_vertices Output pointer to receive total number of boundary vertices across all loops
 * @param num_loops Output pointer to receive number of boundary loops found
 */
CPGEO_API void mesh_extract_boundary_loops_compute(
        const int* triangles,
        int num_triangles,
        int* num_boundary_vertices,
        int* num_loops

    );

/**
 * @brief Retrieve the extracted boundary loops
 * 
 * @param out_boundary_vertices Output array for boundary vertex indices (size: num_boundary_vertices)
 * @param out_loop_indices Output array for loop starting indices (size: num_loops + 1)
 * @return 0 on success, non-zero on error
 */
CPGEO_API int mesh_extract_boundary_loops_get(
        int* out_boundary_vertices,
        int* out_loop_indices
    );

/**
 * @brief Compute closure edge length loss and its first derivative
 * @param vertices Input array of vertex coordinates (size: num_vertices * vertices_dim)
 * @param num_vertices Number of vertices
 * @param vertices_dim Dimension of each vertex (e.g., 2 for 2D, 3 for 3D)
 * @param edges Input array of edge vertex indices (size: num_edges * 2)
 * @param num_edges Number of edges
 * @param order Order of the length penalty (e.g., 2 for squared length)
 * @param out_loss Output pointer to receive the computed loss value
 */
CPGEO_API void mesh_closure_edge_length_derivative0(
        const double* vertices,
        int num_vertices,
        int vertices_dim,
        const int* edges,
        int num_edges,
        int order,
        double* out_loss
);

/**
 * @brief Compute closure edge length loss and its second derivative
 * 
 * @param vertices Input array of vertex coordinates (size: num_vertices * vertices_dim)
 * @param num_vertices Number of vertices
 * @param vertices_dim Dimension of each vertex (e.g., 2 for 2D, 3 for 3D)
 * @param edges Input array of edge vertex indices (size: num_edges * 2)
 * @param num_edges Number of edges
 * @param order Order of the length penalty (e.g., 2 for squared length)
 * @param out_loss Output pointer to receive the computed loss value
 */
CPGEO_API void mesh_closure_edge_length_derivative2_compute(
        const double* vertices,
        int num_vertices,
        int vertices_dim,
        const int* edges,
        int num_edges,
        int order,
        int* num_out_ldr2
);

/**
 * @brief Retrieve closure edge length loss and its second derivative
 * 
 * @param out_loss Output pointer to receive the computed loss value
 * @param out_ldr Output array for first derivative (size: num_vertices * vertices_dim)
 * @param out_ldr2_indices Output array for second derivative indices in COO format (size: num_out_ldr2 * 4)
 * @param out_ldr2_values Output array for second derivative values in COO format (size: num_out_ldr2)
 * @return 0 on success, non-zero on error
 */
CPGEO_API int mesh_closure_edge_length_derivative2_get(
        double* out_loss,
        double* out_ldr,
        int* out_ldr2_indices,
        double* out_ldr2_values
);

/**
 * @brief Optimize mesh by edge flipping to make triangles more equilateral
 * 
 * This function performs iterative edge flipping on a manifold triangle mesh to improve
 * triangle quality by maximizing minimum angles. Boundary edges are never flipped.
 * 
 * @param vertices Vertex coordinates (size: num_vertices * vertices_dim)
 * @param num_vertices Number of vertices
 * @param vertices_dim Dimension of each vertex (2 for 2D, 3 for 3D)
 * @param faces_in Input triangle faces (size: num_faces * 3)
 * @param num_faces Number of triangular faces
 * @param max_iterations Maximum number of optimization iterations
 * @param out_faces Output buffer for optimized faces (size: num_faces * 3)
 */
CPGEO_API void mesh_optimize_by_edge_flipping(
        const double* vertices,
        int num_vertices,
        int vertices_dim,
        const int* faces_in,
        int num_faces,
        int max_iterations,
        int* out_faces
);

#ifdef __cplusplus
}
#endif
