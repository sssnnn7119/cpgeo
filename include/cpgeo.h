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


#ifdef __cplusplus
}
#endif
