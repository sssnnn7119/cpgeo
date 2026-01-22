#include "mesh_utils.h"


// edges method
namespace cpgeo {
std::unordered_map<std::pair<int, int>, int, EdgeHash> extractEdgesWithNumber(const std::span<const int>& triangles) {
    std::unordered_map<std::pair<int, int>, int, EdgeHash> edge_count;
    int num_triangles = static_cast<int>(triangles.size() / 3);
    for (int tri_idx = 0; tri_idx < num_triangles; ++tri_idx) {
        for (int j = 0; j < 3; ++j) {
            int a = triangles[tri_idx * 3 + j];
            int b = triangles[tri_idx * 3 + ((j + 1) % 3)];
            auto edge = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
            ++edge_count[edge];
        }
    }
    
    return edge_count;
}

std::vector<std::vector<int>> extractBoundaryLoops(const std::span<const int>& triangles) {
    const int num_triangles = static_cast<int>(triangles.size() / 3);
    
    // Find all boundary edges (edges that appear only once)
    auto edge_count = extractEdgesWithNumber(triangles);
    
    // Build adjacency for boundary edges (directed)
    std::unordered_map<int, std::vector<int>> boundary_adj;
    for (const auto& [edge, count] : edge_count) {
        if (count == 1) {
            // Add both directions since we need to traverse the boundary
            boundary_adj[edge.first].push_back(edge.second);
            boundary_adj[edge.second].push_back(edge.first);
        }
    }
    
    // Extract all boundary loops
    std::vector<std::vector<int>> loops;
    std::unordered_set<int> visited;
    
    for (const auto& [start_vertex, neighbors] : boundary_adj) {
        if (visited.count(start_vertex)) {
            continue;
        }
        
        // Trace a loop starting from this vertex
        std::vector<int> loop;
        int current = start_vertex;
        int prev = -1;
        
        while (true) {
            loop.push_back(current);
            visited.insert(current);
            
            // Find next vertex (not the one we came from)
            int next = -1;
            for (int neighbor : boundary_adj[current]) {
                if (neighbor != prev) {
                    next = neighbor;
                    break;
                }
            }
            
            if (next == -1 || next == start_vertex) {
                break;
            }
            
            prev = current;
            current = next;
        }
        
        if (loop.size() > 2) {
            loops.push_back(loop);
        }
    }
    
    return loops;
}



}

// mesh fineturing method
namespace cpgeo {

    double closure_edge_length_derivative0(std::span<const double> vertices, int vertices_dim, std::span<const int> edges, int order) {

        int num_edges = edges.size() / 2;

        double loss = 0.;

        for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            int v0_idx = edges[edge_idx * 2];
            int v1_idx = edges[edge_idx * 2 + 1];

            double edge_length = 0.;
            for (int d = 0; d < vertices_dim; d++) {
                double diff = vertices[v0_idx * vertices_dim + d] - vertices[v1_idx * vertices_dim + d];
                edge_length += diff * diff;
            }
            edge_length = std::sqrt(edge_length) + 1e-8;
            loss += std::pow(edge_length, order);
        }

        return loss;
    }

    std::tuple<double, std::vector<double>, std::vector<int>, std::vector<double>> closure_edge_length_derivative2(std::span<const double> vertices, int vertices_dim, std::span<const int> edges, int order) {

        int num_edges = edges.size() / 2;
        int num_vertices = vertices.size() / vertices_dim;

        double loss = 0.;
        std::vector<double> Ldr(vertices.size(), 0.);
        std::unordered_map<std::array<int, 4>, double, Array4IntHash> Ldr2_map; // (v0_idx, v0_dim, v1_idx, v1_dim) -> value

        for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            int v0_idx = edges[edge_idx * 2];
            int v1_idx = edges[edge_idx * 2 + 1];

            double edge_length = 0.;
            auto diffs = std::vector<double>(vertices_dim, 0.);
            for (int dim_idx = 0; dim_idx < vertices_dim; dim_idx++) {
                double diff = vertices[v0_idx * vertices_dim + dim_idx] - vertices[v1_idx * vertices_dim + dim_idx];
                diffs[dim_idx] = diff;
                edge_length += diff * diff;
            }
            edge_length = std::sqrt(edge_length);
            loss += std::pow(edge_length, order);

            double Ldl = order * std::pow(edge_length, order - 1);
            double Ldl2 = order * (order - 1) * std::pow(edge_length, order - 2);

            for (int dim1_idx = 0; dim1_idx < vertices_dim; dim1_idx++) {
                // the first derivative
                double ldre = diffs[dim1_idx] / edge_length;
                double Ldre = Ldl * ldre;
                Ldr[v0_idx * vertices_dim + dim1_idx] += Ldre;
                Ldr[v1_idx * vertices_dim + dim1_idx] -= Ldre;

                // the second derivative
                for (int dim2_idx = 0; dim2_idx < vertices_dim; dim2_idx++) {
                    double delta = (dim1_idx == dim2_idx) ? 1.0 : 0.0;
                    double ldre2 = (delta * edge_length - ldre * diffs[dim2_idx]) / (edge_length * edge_length);
                    double Ldre2 = Ldl2 * diffs[dim2_idx] / edge_length * ldre + Ldl * ldre2;

                    auto key_v0 = std::array<int, 4>{ v0_idx, dim1_idx, v0_idx, dim2_idx };
                    Ldr2_map[key_v0] += Ldre2;

                    auto key_v1 = std::array<int, 4>{ v1_idx, dim1_idx, v1_idx, dim2_idx };
                    Ldr2_map[key_v1] += Ldre2;

                    auto key_v0v1 = std::array<int, 4>{ v0_idx, dim1_idx, v1_idx, dim2_idx };
                    Ldr2_map[key_v0v1] -= Ldre2;

                    auto key_v1v0 = std::array<int, 4>{ v1_idx, dim1_idx, v0_idx, dim2_idx };
                    Ldr2_map[key_v1v0] -= Ldre2;
                }
            }

        }

        // convert Ldr2_map to vectors
        std::vector<int> Ldr2_indices; // (v0_idx, v0_dim, v1_idx, v1_dim, flattened)
        std::vector<double> Ldr2_values;

        Ldr2_indices.reserve(Ldr2_map.size() * 4);
        Ldr2_values.reserve(Ldr2_map.size());

        for (const auto& [key, value] : Ldr2_map) {
            Ldr2_indices.push_back(key[0]);
            Ldr2_indices.push_back(key[1]);
            Ldr2_indices.push_back(key[2]);
            Ldr2_indices.push_back(key[3]);
            Ldr2_values.push_back(value);
        }

		return { loss, Ldr, Ldr2_indices, Ldr2_values };

    }

    

}