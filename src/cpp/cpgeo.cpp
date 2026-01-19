#include "cpgeo.h"
#include "triangulation.h"
#include "sphere_triangulation.h"
#include "space_tree.h"
#include <memory>
#include <algorithm>

extern "C" {

// Triangulation
CPGEO_API cpgeo_handle_t triangulation_compute(
    const double* nodes,
    int num_nodes,
    int* num_triangles
) {
    if (!nodes || !num_triangles || num_nodes < 3) {
        return nullptr;
    }

    try {
        std::span<const double> nodes_span(nodes, num_nodes * 2);
        auto* tri = new cpgeo::DelaunayTriangulation(nodes_span);
        
        tri->triangulate();
        
        *num_triangles = static_cast<int>(tri->size());
        return static_cast<cpgeo_handle_t>(tri);
    } catch (...) {
        return nullptr;
    }
}

CPGEO_API int triangulation_get_data(
    cpgeo_handle_t handle,
    int* triangles
) {
    if (!handle || !triangles) {
        return -1;
    }

    try {
        auto* tri = static_cast<cpgeo::DelaunayTriangulation*>(handle);
        tri->getTriangleIndices(triangles);
        delete tri;
        return 0;
    } catch (...) {
        return -1;
    }
}

// Sphere Triangulation
CPGEO_API cpgeo_handle_t sphere_triangulation_compute(
    const double* sphere_points,
    int num_points,
    int* num_triangles
) {
    if (!sphere_points || !num_triangles || num_points < 4) {
        return nullptr;
    }

    try {
        std::span<const double> points_span(sphere_points, num_points * 3);
        auto* tri = new cpgeo::SphereTriangulation(points_span);
        
        tri->triangulate();
        
        *num_triangles = static_cast<int>(tri->size());
        return static_cast<cpgeo_handle_t>(tri);
    } catch (...) {
        return nullptr;
    }
}

CPGEO_API int sphere_triangulation_get_data(
    cpgeo_handle_t handle,
    int* triangles
) {
    if (!handle || !triangles) {
        return -1;
    }

    try {
        auto* tri = static_cast<cpgeo::SphereTriangulation*>(handle);
        tri->getTriangleIndices(std::span<int>(triangles, tri->size() * 3));
        delete tri;
        return 0;
    } catch (...) {
        return -1;
    }
}

// SpaceTree
CPGEO_API cpgeo_handle_t space_tree_create(
    const double* knots,
    int num_knots,
    const double* thresholds
) {
    if (!knots || !thresholds || num_knots <= 0) {
        return nullptr;
    }

    try {
        std::span<const double> knots_span(knots, num_knots * 3);
        std::span<const double> thresholds_span(thresholds, num_knots);
        auto* tree = new cpgeo::SpaceTree(knots_span, thresholds_span);
        return static_cast<cpgeo_handle_t>(tree);
    } catch (...) {
        return nullptr;
    }
}

CPGEO_API int space_tree_query_compute(
    cpgeo_handle_t tree,
    const double* query_points,
    int num_queries,
    int* num_results,
    int* total_results
) {
    if (!tree || !query_points || !num_results || !total_results || num_queries <= 0) {
        return -1;
    }

    try {
        auto* tree_ptr = static_cast<cpgeo::SpaceTree*>(tree);
        std::span<const double> query_span(query_points, num_queries * 3);
        tree_ptr->compute_indices(query_span);

        const auto& results = tree_ptr->get_query_results();

        // Fill num_results and total_results
        *total_results = 0;
        for (int i = 0; i < num_queries; ++i) {
            num_results[i] = static_cast<int>(results[i].size());
            *total_results += num_results[i];
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

CPGEO_API int space_tree_query_get(
    cpgeo_handle_t tree,
    int num_results,
    int* results
) {
    if (!tree) {
        return -1;
    }
    // Allow num_results 0 (no results found)
    if (num_results == 0) {
        return 0;
    }
    if (!results) {
        return -1;
    }

    try {
        auto* tree_ptr = static_cast<cpgeo::SpaceTree*>(tree);
        const auto& query_results = tree_ptr->get_query_results();

        // Safety check: ensure provided buffer size matches internal count
        size_t total_count = 0;
        for (const auto& res : query_results) {
            total_count += res.size();
        }

        if (static_cast<size_t>(num_results) < total_count) {
             return -2; // Buffer too small
        }

        int offset = 0;
        for (int i = 0; i < static_cast<int>(query_results.size()); ++i) {
            const auto& res_vec = query_results[i];
            for (size_t j = 0; j < res_vec.size(); ++j) {
                results[offset] = i;
                results[offset + num_results] = res_vec[j];
                offset++;
            }
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

CPGEO_API void space_tree_destroy(cpgeo_handle_t handle) {
    if (handle) {
        delete static_cast<cpgeo::SpaceTree*>(handle);
    }
}

}  // extern "C"
