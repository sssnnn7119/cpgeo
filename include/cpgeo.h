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
 * @brief Perform Delaunay triangulation on a set of 2D points
 * 
 * @param nodes Input array of node coordinates [x0, y0, x1, y1, ..., xn, yn]
 * @param num_nodes Number of nodes (not the array length)
 * @param triangles Output array for triangle indices [v0, v1, v2, ...] 
 *                  Must be pre-allocated with at least num_nodes*2*3 integers
 * @param num_triangles Output: number of triangles generated
 * @return 0 on success, non-zero on error
 */
CPGEO_API int cpgeo_triangulate(
    const double* nodes,
    int num_nodes,
    int* triangles,
    int* num_triangles
);

#ifdef __cplusplus
}
#endif
