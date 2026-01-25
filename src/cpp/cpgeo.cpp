#include "cpgeo.h"
#include "triangulation.h"
#include "sphere_triangulation.h"
#include "space_tree.h"
#include "cpgeo_mapping.h"
#include "mesh_edge_flip.h"
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

static std::vector<std::vector<int>> temp_query_results;
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
        temp_query_results = tree_ptr->query_point_batch(query_span);

        // Fill num_results and total_results
        *total_results = 0;
        for (int i = 0; i < num_queries; ++i) {
            *total_results += static_cast<int>(temp_query_results[i].size());
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
        const auto& query_results = temp_query_results;

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

        temp_query_results.clear(); // Clear temporary storage

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
			int num_indices_now = end - start;
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
            // copy first derivatives to out_weights_du with layout [2, num_indices]
            std::copy(weightsdu.begin(), weightsdu.begin() + num_indices_now, out_weights_du + start);
            std::copy(weightsdu.begin() + num_indices_now, weightsdu.begin() + num_indices_now * 2, out_weights_du + num_indices + start);
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

        //#pragma omp parallel for
        for(int ptidx=0;ptidx<num_queries;ptidx++){
            int start = indices_pts[ptidx];
            int end = indices_pts[ptidx+1];
			int num_indices_now = end - start;
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
            // copy first derivatives to out_weights_du with layout [2, num_indices]
            std::copy(weightsdu.begin(), weightsdu.begin() + num_indices_now, out_weights_du + start);
            std::copy(weightsdu.begin() + num_indices_now, weightsdu.begin() + num_indices_now * 2, out_weights_du + num_indices + start);

            // copy second derivatives to out_weights_du2 with layout [2,2,num_indices]
            // weightsdu2 organized in blocks (k, j) each of length num_indices_now
            for (int k = 0; k < 2; ++k) {
                for (int j = 0; j < 2; ++j) {
                    std::copy(
                        weightsdu2.begin() + (k * 2 + j) * num_indices_now,
                        weightsdu2.begin() + (k * 2 + j + 1) * num_indices_now,
                        out_weights_du2 + ((k * 2 + j) * num_indices) + start
                    );
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
        std::span<const int> indices_cps_span(indices_cps, num_indices);
        std::span<const int> indices_pts_span(indices_pts, num_queries + 1);
        std::span<const double> weights_span(weights, num_indices);
        std::span<const double> controlpoints_span(controlpoints, num_controlpoints * 3);

        auto mapped = cpgeo::get_mapped_points_batch(
            indices_cps_span,
            indices_pts_span,
            weights_span,
            controlpoints_span
        );

        std::copy(mapped.begin(), mapped.end(), out_mapped_points);

    } catch (...) {
        return;
    }
}

CPGEO_API void cpgeo_map_points(
    const double* query_points,
    cpgeo_handle_t tree,
    const double* controlpoints,
    int num_controlpoints,
    int num_queries,
    double* out_mapped_points
) {
    if (!query_points || !tree || !controlpoints || !out_mapped_points || num_controlpoints <= 0 || num_queries <= 0) {
        return;
    }

    try {
        auto* tree_ptr = static_cast<cpgeo::SpaceTree*>(tree);
        std::span<const double> query_span(query_points, num_queries * 3);
        std::span<const double> cp_span(controlpoints, num_controlpoints * 3);

        auto mapped = cpgeo::map_points_batch(query_span, *tree_ptr, cp_span);
        std::copy(mapped.begin(), mapped.end(), out_mapped_points);

    } catch (...) {
        return;
    }
}

CPGEO_API void cpgeo_map_points_derivative2(
    const double* query_points,
    cpgeo_handle_t tree,
    const double* controlpoints,
    int num_controlpoints,
    int num_queries,
    double* out_r,
    double* out_rdu,
    double* out_rdu2
) {
    if (!query_points || !tree || !controlpoints || !out_r || !out_rdu || !out_rdu2 || num_controlpoints <= 0 || num_queries <= 0) {
        return;
    }

    try {
        auto* tree_ptr = static_cast<cpgeo::SpaceTree*>(tree);
        std::span<const double> query_span(query_points, num_queries * 3);
        std::span<const double> cp_span(controlpoints, num_controlpoints * 3);

        auto result = cpgeo::map_points_batch_derivative2(query_span, *tree_ptr, cp_span);
        const auto& r = result[0];
        const auto& rdu = result[1];
        const auto& rdu2 = result[2];

        // copy results to outputs
        std::copy(r.begin(), r.end(), out_r);
        std::copy(rdu.begin(), rdu.end(), out_rdu);
        std::copy(rdu2.begin(), rdu2.end(), out_rdu2);

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


    CPGEO_API void mesh_closure_edge_length_derivative0(
        const double* vertices,
        int num_vertices,
        int vertices_dim,
        const int* edges,
        int num_edges,
        int order,
        double* out_loss
    ) {
        if (!vertices || !edges || !out_loss || vertices_dim <= 0 || num_edges <= 0) {
            return;
        }

        try {
            std::span<const double> vertices_span(vertices, vertices_dim * num_vertices);
            std::span<const int> edges_span(edges, num_edges * 2);
            double loss = cpgeo::closure_edge_length_derivative0(vertices_span, vertices_dim, edges_span, order);
            *out_loss = loss;
        } catch (...) {
            return;
        }
    }

    static std::tuple<double, std::vector<double>, std::vector<int>, std::vector<double>> closure_result;
    CPGEO_API void mesh_closure_edge_length_derivative2_compute(
        const double* vertices,
        int num_vertices,
        int vertices_dim,
        const int* edges,
        int num_edges,
        int order,
        int* num_out_ldr2
    ) {
        if (!vertices || !edges || !num_out_ldr2 ||
            vertices_dim <= 0 || num_edges <= 0) {
            return;
        }

        try {
            std::span<const double> vertices_span(vertices, vertices_dim * num_vertices);
            std::span<const int> edges_span(edges, num_edges * 2);
            closure_result = cpgeo::closure_edge_length_derivative2(vertices_span, vertices_dim, edges_span, order);
            *num_out_ldr2 = static_cast<int>(std::get<3>(closure_result).size());
        } catch (...) {
            return;
        }
    }

    CPGEO_API int mesh_closure_edge_length_derivative2_get(
        double* out_loss,
        double* out_ldr,
        int* out_ldr2_indices,
        double* out_ldr2_values
    ) {
        if (!out_loss || !out_ldr || !out_ldr2_indices || !out_ldr2_values) {
            return -1;
        }

        try {
            *out_loss = std::get<0>(closure_result);
            const auto& ldr = std::get<1>(closure_result);
            const auto& ldr2_indices = std::get<2>(closure_result);
            const auto& ldr2_values = std::get<3>(closure_result);

            std::copy(ldr.begin(), ldr.end(), out_ldr);
            std::copy(ldr2_indices.begin(), ldr2_indices.end(), out_ldr2_indices);
            std::copy(ldr2_values.begin(), ldr2_values.end(), out_ldr2_values);

            // Clear stored result
            closure_result = {};

            return 0;
        } catch (...) {
            return -1;
        }
    }

    CPGEO_API void mesh_optimize_by_edge_flipping(
        const double* vertices,
        int num_vertices,
        int vertices_dim,
        const int* faces_in,
        int num_faces,
        int max_iterations,
        int* out_faces
    ) {
        if (!vertices || !faces_in || !out_faces || num_vertices <= 0 || num_faces <= 0) {
            return;
        }

        try {
            std::span<const double> vertices_span(vertices, num_vertices * vertices_dim);
            std::span<const int> faces_span(faces_in, num_faces * 3);
            
            auto optimized_faces = cpgeo::mesh_optimize_by_edge_flipping(
                vertices_span, vertices_dim, faces_span, max_iterations
            );
            
            std::copy(optimized_faces.begin(), optimized_faces.end(), out_faces);
        } catch (...) {
            // On error, copy input to output
            std::copy(faces_in, faces_in + num_faces * 3, out_faces);
        }
    }

}  // extern "C"