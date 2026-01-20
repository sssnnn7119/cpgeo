#include "cpgeo.h"
#include "triangulation.h"
#include "sphere_triangulation.h"
#include "space_tree.h"
#include "cpgeo_mapping.h"
#include <memory>
#include <algorithm>
#include "tensor.h"

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
    int* total_results
) {
    if (!tree || !query_points || !total_results || num_queries <= 0) {
        return -1;
    }

    try {
        auto* tree_ptr = static_cast<cpgeo::SpaceTree*>(tree);
        std::span<const double> query_span(query_points, num_queries * 3);
        tree_ptr->compute_indices_batch(query_span);

        const auto& results = tree_ptr->get_query_results();

        // Fill num_results and total_results
        *total_results = 0;
        for (int i = 0; i < num_queries; ++i) {
            *total_results += static_cast<int>(results[i].size());
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

CPGEO_API int space_tree_query_get(
    cpgeo_handle_t tree,
    int num_results,
    int* indices_cps,
    int* indices_pts
) {
    if (!tree) {
        return -1;
    }
    // Allow num_results 0 (no results found)
    if (num_results == 0) {
        return 0;
    }
    if (!indices_cps || !indices_pts) {
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
        
        uint32_t offset = 0;
        indices_pts[0] = 0;  // First query starts at 0
        for (int i = 0; i < static_cast<int>(query_results.size()); ++i) {
            const auto& res_vec = query_results[i];
            for (size_t j = 0; j < res_vec.size(); ++j) {
                indices_cps[offset] = res_vec[j];
                offset++;
            }
            indices_pts[i + 1] = static_cast<int>(offset);  // Next query starts after this one
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



CPGEO_API void cpgeo_compute_thresholds(
    const double* knots,
    int num_knots,
    int k,
    double* out_thresholds
) {
    if (!knots || !out_thresholds || num_knots <= 0 || k <= 0) {
        return;
    }

    try {
        std::span<const double> knots_span(knots, num_knots * 3);
        auto thresholds = cpgeo::compute_thresholds(knots_span, k);
        std::copy(thresholds.begin(), thresholds.end(), out_thresholds);
    } catch (...) {
        return;
    }
}

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
) {
    if (!indices_cps || !indices_pts || !knots || !thresholds || !query_points || !out_weights ||
        num_indices <= 0 || num_knots <= 0 || num_queries <= 0) {
        return;
    }

    try {
        std::span<const int> indices_cps_span(indices_cps, num_indices);
        std::span<const int> indices_pts_span(indices_pts, num_queries + 1);
        std::span<const double> knots_span(knots, num_knots * 3);
        std::span<const double> thresholds_span(thresholds, num_knots);
        std::span<const double> query_span(query_points, num_queries * 3);

        for(int ptidx=0;ptidx<num_queries;ptidx++){
            int start = indices_pts[ptidx];
            int end = indices_pts[ptidx+1];
            if(end - start <= 0){
                continue;
            }
            auto weights = cpgeo::get_weights(
                std::span<const int>(indices_cps + start, end - start),
                knots_span,
                thresholds_span,
                std::span<const double, 2>(query_points + ptidx * 3, 2)
            );
            std::copy(weights.begin(), weights.end(), out_weights + start);
        }
    } catch (...) {
        return;
    }
}

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
){

    if (!indices_cps || !indices_pts || !knots || !thresholds || !query_points || !out_weights || !out_weights_du ||
        num_indices <= 0 || num_knots <= 0 || num_queries <= 0) {
        return;
    }

    try {
        std::span<const int> indices_cps_span(indices_cps, num_indices);
        std::span<const int> indices_pts_span(indices_pts, num_queries + 1);
        std::span<const double> knots_span(knots, num_knots * 3);
        std::span<const double> thresholds_span(thresholds, num_knots);
        std::span<const double> query_span(query_points, num_queries * 3);

        //#pragma omp parallel for
        for(int ptidx=0;ptidx<num_queries;ptidx++){
            int start = indices_pts[ptidx];
            int end = indices_pts[ptidx+1];
            if(end - start <= 0){
                continue;
            }
            auto [weights, weightsdu] = cpgeo::get_weights_derivative1(
                std::span<const int>(indices_cps + start, end - start),
                knots_span,
                thresholds_span,
                std::span<const double, 2>(query_points + ptidx * 3, 2)
            );
            std::copy(weights.begin(), weights.end(), out_weights + start);
            for(int i=0;i<end - start;i++){
                out_weights_du[(start + i)*2 + 0] = weightsdu.at(i, 0);
                out_weights_du[(start + i)*2 + 1] = weightsdu.at(i, 1);
            }
        }
    } catch (...) {
        return;
    }
}

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
){

    if (!indices_cps || !indices_pts || !knots || !thresholds || !query_points || !out_weights || !out_weights_du || !out_weights_du2 ||
        num_indices <= 0 || num_knots <= 0 || num_queries <= 0) {
        return;
    }

    try {
        std::span<const int> indices_cps_span(indices_cps, num_indices);
        std::span<const int> indices_pts_span(indices_pts, num_queries + 1);
        std::span<const double> knots_span(knots, num_knots * 3);
        std::span<const double> thresholds_span(thresholds, num_knots);
        std::span<const double> query_span(query_points, num_queries * 3);

        //#pragma omp parallel for
        for(int ptidx=0;ptidx<num_queries;ptidx++){
            int start = indices_pts[ptidx];
            int end = indices_pts[ptidx+1];
            if(end - start <= 0){
                continue;
            }
            auto [weights, weightsdu, weightsdu2] = cpgeo::get_weights_derivative2(
                std::span<const int>(indices_cps + start, end - start),
                knots_span,
                thresholds_span,
                std::span<const double, 2>(query_points + ptidx * 3, 2)
            );
            std::copy(weights.begin(), weights.end(), out_weights + start);
            for(int i=0;i<end - start;i++){
                out_weights_du[(start + i)*2 + 0] = weightsdu.at(i, 0);
                out_weights_du[(start + i)*2 + 1] = weightsdu.at(i, 1);
            }

            for(int i=0;i<end - start;i++){
                for(int j=0;j<2;j++){
                    for(int k=0;k<2;k++){
                        out_weights_du2[((start + i)*2 + j)*2 + k] = weightsdu2.at(i, j, k);
                    }
                }
            }
        }
    } catch (...) {
        return;
    }
}

}  // extern "C"
