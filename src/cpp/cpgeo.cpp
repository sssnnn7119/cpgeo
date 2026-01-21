#include "cpgeo.h"
#include "triangulation.h"
#include "sphere_triangulation.h"
#include "space_tree.h"
#include "cpgeo_mapping.h"
#include <memory>
#include <algorithm>
#include "tensor.h"
#include "mesh_utils.h"

// cpgeo method implementations
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

        #pragma omp parallel for
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
                std::span<const double, 2>(query_points + ptidx * 2, 2)
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

        #pragma omp parallel for
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
                std::span<const double, 2>(query_points + ptidx * 2, 2)
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

        #pragma omp parallel for
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
                std::span<const double, 2>(query_points + ptidx * 2, 2)
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

CPGEO_API void cpgeo_get_mapped_points(
    const int* indices_cps,
    const int* indices_pts,
    int num_indices,
    const double* weights,
    const double* controlpoints,
    int num_controlpoints,
    int num_queries,
    double* out_mapped_points
) {
    if (!indices_cps || !indices_pts || !weights || !controlpoints || !out_mapped_points ||
        num_indices <= 0 || num_controlpoints <= 0 || num_queries <= 0) {
        return;
    }

    try {
        // 初始化输出数组为0
        std::fill_n(out_mapped_points, num_queries * 3, 0.0);

        // 循环计算每个查询点的映射坐标
        #pragma omp parallel for
        for (int i = 0; i < num_queries; ++i) {
            int start = indices_pts[i];
            int end = indices_pts[i + 1];

            for (int j = start; j < end; ++j) {
                int cp_idx = indices_cps[j];
                double weight = weights[j];

                // 确保控制点索引有效
                if (cp_idx >= 0 && cp_idx < num_controlpoints) {
                    for (int k = 0; k < 3; ++k) {
                        out_mapped_points[i * 3 + k] += weight * controlpoints[cp_idx * 3 + k];
                    }
                }
            }
        }
    } catch (...) {
        return;
    }
}

}  // extern "C"


// mesh utility implementations
extern "C" {

    static std::unordered_map<std::pair<int, int>, int, cpgeo::EdgeHash> edges;

    CPGEO_API void mesh_edges_compute(
        const int* elements,
        int num_elements,
        int* out_num_edges
    ) {
        if (!elements || !out_num_edges || num_elements <= 0) {
            return;
        }

        try {
            std::span<const int> elements_span(elements, num_elements * 3);
            edges = cpgeo::extractEdgesWithNumber(elements_span);
            *out_num_edges = static_cast<int>(edges.size());
        } catch (...) {
            return;
        }
    }

    CPGEO_API int mesh_edges_get(
        int* out_edges
    ) {
        if (!out_edges) {
            return -1;
        }

        try {
            for (const auto& [edge, count] : edges) {
                *out_edges++ = edge.first;
                *out_edges++ = edge.second;
                *out_edges++ = count;
            }
            edges.clear();
            return 0;
        } catch (...) {
            return -1;
        }
    }

    // Global storage for mesh partition results
    // static cpgeo::MeshPartition partition_result;

    // CPGEO_API void mesh_partition_sphere_compute(
    //     const int* triangles,
    //     int num_triangles,
    //     int num_vertices,
    //     int* out_num_hemisphere1,
    //     int* out_num_hemisphere2,
    //     int* out_num_cut_vertices
    // ) {
    //     if (!triangles || !out_num_hemisphere1 || !out_num_hemisphere2 || !out_num_cut_vertices) {
    //         return;
    //     }

    //     try {
    //         std::span<const int> triangles_span(triangles, num_triangles * 3);
    //         partition_result = cpgeo::partitionSphereMesh(triangles_span, num_vertices);
            
    //         *out_num_hemisphere1 = static_cast<int>(partition_result.hemisphere1_faces.size() / 3);
    //         *out_num_hemisphere2 = static_cast<int>(partition_result.hemisphere2_faces.size() / 3);
    //         *out_num_cut_vertices = static_cast<int>(partition_result.cut_vertices.size());
    //     } catch (...) {
    //         *out_num_hemisphere1 = 0;
    //         *out_num_hemisphere2 = 0;
    //         *out_num_cut_vertices = 0;
    //     }
    // }

    // CPGEO_API int mesh_partition_sphere_get(
    //     int* hemisphere1_triangles,
    //     int* hemisphere2_triangles,
    //     int* cut_vertices
    // ) {
    //     if (!hemisphere1_triangles || !hemisphere2_triangles || !cut_vertices) {
    //         return -1;
    //     }

    //     try {
    //         std::copy(partition_result.hemisphere1_faces.begin(), 
    //                  partition_result.hemisphere1_faces.end(), 
    //                  hemisphere1_triangles);
    //         std::copy(partition_result.hemisphere2_faces.begin(), 
    //                  partition_result.hemisphere2_faces.end(), 
    //                  hemisphere2_triangles);
    //         std::copy(partition_result.cut_vertices.begin(), 
    //                  partition_result.cut_vertices.end(), 
    //                  cut_vertices);
            
    //         // Clear the results after retrieval
    //         partition_result = cpgeo::MeshPartition{};
    //         return 0;
    //     } catch (...) {
    //         return -1;
    //     }
    // }




    // Global storage for boundary loops results
    static std::vector<std::vector<int>> boundary_loops;

    CPGEO_API void mesh_extract_boundary_loops_compute(
        const int* triangles,
        int num_triangles,
        int* num_boundary_vertices,
        int* num_loops

    ) {
        if (!triangles || !num_boundary_vertices || !num_loops) {
            return;
        }

        try {
            std::span<const int> triangles_span(triangles, num_triangles * 3);
            boundary_loops = cpgeo::extractBoundaryLoops(triangles_span);
            
            *num_boundary_vertices = 0;
            for (const auto& loop : boundary_loops) {
                *num_boundary_vertices += static_cast<int>(loop.size());
            }
            *num_loops = static_cast<int>(boundary_loops.size());

        } catch (...) {
            *num_boundary_vertices = 0;
            *num_loops = 0;
        }
    }

    CPGEO_API int mesh_extract_boundary_loops_get(
        int* out_boundary_vertices,
        int* out_loop_indices
    ) {
        if (!out_boundary_vertices || !out_loop_indices) {
            return -1;
        }

        try {
            int offset = 0;
            out_loop_indices[0] = 0;  // First loop starts at 0
            for (int i = 0; i < static_cast<int>(boundary_loops.size()); ++i) {
                const auto& loop = boundary_loops[i];
                out_loop_indices[i+1] = out_loop_indices[i] + static_cast<int>(loop.size());
                for (int v : loop) {
                    out_boundary_vertices[offset++] = v;
                }
            }

            
            boundary_loops.clear();
            return 0;
        } catch (...) {
            return -1;
        }
    }


}  // extern "C"