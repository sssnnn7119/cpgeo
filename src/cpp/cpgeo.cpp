#include "cpgeo.h"
#include "triangular_mesh.h"
#include <memory>

extern "C" {

CPGEO_API int cpgeo_triangulate(
    const double* nodes,
    int num_nodes,
    int* triangles,
    int* num_triangles
) {
    if (!nodes || !triangles || !num_triangles || num_nodes < 3) {
        return -1;  // Invalid parameters
    }

    try {
        std::span<const double> nodes_span(nodes, num_nodes * 2);
        cpgeo::DelaunayTriangulation triangulation(nodes_span);
        
        triangulation.triangulate();
        
        *num_triangles = static_cast<int>(triangulation.size());
        triangulation.getTriangleIndices(triangles);
        
        return 0;  // Success
    } catch (...) {
        return -2;  // Error during triangulation
    }
}

}  // extern "C"
