#include "mesh_edge_flip.h"
#include "mesh_utils.h"
#include <limits>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cpgeo {

// Helper structure to represent an edge with its adjacent triangles
struct EdgeInfo {
    int tri0 = -1;  // First adjacent triangle index (-1 if boundary)
    int tri1 = -1;  // Second adjacent triangle index (-1 if boundary)
    int opp0 = -1;  // Index (0-2) of the vertex opposite to this edge in tri0
    int opp1 = -1;  // Index (0-2) of the vertex opposite to this edge in tri1
};

struct TriangleKeyHash {
    std::size_t operator()(const std::array<int, 3>& k) const noexcept {
        return std::hash<int>{}(k[0]) ^ (std::hash<int>{}(k[1]) << 1) ^ (std::hash<int>{}(k[2]) << 2);
    }
};

inline std::array<int, 3> make_triangle_key(int a, int b, int c) {
    std::array<int, 3> key{a, b, c};
    std::sort(key.begin(), key.end());
    return key;
}

// Compute the minimum angle in a triangle (in radians)
inline double compute_min_angle(
    const double* v0, const double* v1, const double* v2, int dim)
{
    // Compute edge vectors
    std::array<double, 3> e01{}, e12{}, e20{};
    for (int d = 0; d < dim; ++d) {
        e01[d] = v1[d] - v0[d];
        e12[d] = v2[d] - v1[d];
        e20[d] = v0[d] - v2[d];
    }
    
    // Compute edge lengths
    double len01 = 0.0, len12 = 0.0, len20 = 0.0;
    for (int d = 0; d < dim; ++d) {
        len01 += e01[d] * e01[d];
        len12 += e12[d] * e12[d];
        len20 += e20[d] * e20[d];
    }
    len01 = std::sqrt(len01);
    len12 = std::sqrt(len12);
    len20 = std::sqrt(len20);
    
    if (len01 < 1e-12 || len12 < 1e-12 || len20 < 1e-12) {
        return 0.0;  // Degenerate triangle
    }
    
    // Compute angles using dot products
    double dot0 = 0.0, dot1 = 0.0, dot2 = 0.0;
    for (int d = 0; d < dim; ++d) {
        dot0 += e01[d] * (-e20[d]);
        dot1 += e12[d] * (-e01[d]);
        dot2 += e20[d] * (-e12[d]);
    }
    
    double cos0 = dot0 / (len01 * len20);
    double cos1 = dot1 / (len12 * len01);
    double cos2 = dot2 / (len20 * len12);
    
    // Clamp to [-1, 1] to avoid numerical errors
    cos0 = std::clamp(cos0, -1.0, 1.0);
    cos1 = std::clamp(cos1, -1.0, 1.0);
    cos2 = std::clamp(cos2, -1.0, 1.0);
    
    double angle0 = std::acos(cos0);
    double angle1 = std::acos(cos1);
    double angle2 = std::acos(cos2);
    
    return std::min({angle0, angle1, angle2});
}

// Compute normal vector of a triangle
inline std::array<double, 3> compute_normal(
    const double* v0, const double* v1, const double* v2, int dim)
{
    std::array<double, 3> e01{}, e02{};
    for (int d = 0; d < std::min(dim, 3); ++d) {
        e01[d] = v1[d] - v0[d];
        e02[d] = v2[d] - v0[d];
    }
    
    // Cross product: e01 x e02
    std::array<double, 3> normal{
        e01[1] * e02[2] - e01[2] * e02[1],
        e01[2] * e02[0] - e01[0] * e02[2],
        e01[0] * e02[1] - e01[1] * e02[0]
    };
    
    return normal;
}

// Normalize a vector
inline std::array<double, 3> normalize(const std::array<double, 3>& v) {
    double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-12) {
        return {0.0, 0.0, 0.0};
    }
    return {v[0] / len, v[1] / len, v[2] / len};
}

// Compute dot product of two vectors
inline double dot_product(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline bool is_valid_index(int idx, int max_size) {
    return idx >= 0 && idx < max_size;
}

// Compute normal from triangle indices
inline std::array<double, 3> compute_face_normal(
    const int* tri,
    std::span<const double> vertices,
    int dim)
{
    const double* v0 = &vertices[tri[0] * dim];
    const double* v1 = &vertices[tri[1] * dim];
    const double* v2 = &vertices[tri[2] * dim];
    return compute_normal(v0, v1, v2, dim);
}

// Orient triangle so its normal matches target_normal (keeps winding consistent)
inline void orient_triangle_to_match_normal(
    int* tri,
    const std::array<double, 3>& target_normal,
    std::span<const double> vertices,
    int dim)
{
    auto current_normal = normalize(compute_face_normal(tri, vertices, dim));
    if (dot_product(current_normal, target_normal) < 0.0) {
        std::swap(tri[1], tri[2]);
    }
}

inline int directed_edge_direction(const int* tri, int a, int b) {
    for (int i = 0; i < 3; ++i) {
        int v0 = tri[i];
        int v1 = tri[(i + 1) % 3];
        if (v0 == a && v1 == b) return 1;
        if (v0 == b && v1 == a) return -1;
    }
    return 0;
}

// Check if flipping an edge would improve mesh quality
// Returns true if flip is beneficial, false otherwise
inline bool should_flip_edge(
    const std::vector<int>& faces,
    const EdgeInfo& edge_info,
    std::span<const double> vertices,
    int vertices_dim,
    const std::unordered_map<std::pair<int, int>, EdgeInfo, EdgeHash>& edge_map,
    const std::unordered_set<std::array<int, 3>, TriangleKeyHash>& triangle_set)
{
    if (edge_info.tri0 == -1 || edge_info.tri1 == -1) {
        return false;  // Boundary edge, don't flip
    }

    if (faces.size() % 3 != 0 || vertices_dim <= 0) {
        return false;
    }
    const int vertex_count = static_cast<int>(vertices.size()) / vertices_dim;
    const int triangle_count = static_cast<int>(faces.size()) / 3;
    if (vertex_count <= 0) {
        return false;
    }
    if (edge_info.tri0 < 0 || edge_info.tri0 >= triangle_count ||
        edge_info.tri1 < 0 || edge_info.tri1 >= triangle_count) {
        return false;
    }
    if (edge_info.opp0 < 0 || edge_info.opp0 > 2 ||
        edge_info.opp1 < 0 || edge_info.opp1 > 2) {
        return false;
    }
    
    const int* tri0 = &faces[edge_info.tri0 * 3];
    const int* tri1 = &faces[edge_info.tri1 * 3];

    for (int i = 0; i < 3; ++i) {
        if (!is_valid_index(tri0[i], vertex_count) || !is_valid_index(tri1[i], vertex_count)) {
            return false;
        }
    }
    
    // Get the four vertices involved
    int v_opp0 = tri0[edge_info.opp0];
    int v_opp1 = tri1[edge_info.opp1];
    int v_shared0 = tri0[(edge_info.opp0 + 1) % 3];
    int v_shared1 = tri0[(edge_info.opp0 + 2) % 3];

    // Ensure all vertices are distinct
    if (v_opp0 == v_opp1 || v_opp0 == v_shared0 || v_opp0 == v_shared1 ||
        v_opp1 == v_shared0 || v_opp1 == v_shared1 || v_shared0 == v_shared1) {
        return false;
    }
    
    const double* p_opp0 = &vertices[v_opp0 * vertices_dim];
    const double* p_opp1 = &vertices[v_opp1 * vertices_dim];
    const double* p_shared0 = &vertices[v_shared0 * vertices_dim];
    const double* p_shared1 = &vertices[v_shared1 * vertices_dim];
    
    // Compute normals before flip
    auto normal_before_0 = normalize(compute_face_normal(tri0, vertices, vertices_dim));
    auto normal_before_1 = normalize(compute_face_normal(tri1, vertices, vertices_dim));
    
    // Build flipped triangles and orient to match original normals
    std::array<int, 3> new_tri0 = {v_opp0, v_opp1, v_shared0};
    std::array<int, 3> new_tri1 = {v_opp1, v_opp0, v_shared1};
    orient_triangle_to_match_normal(new_tri0.data(), normal_before_0, vertices, vertices_dim);
    orient_triangle_to_match_normal(new_tri1.data(), normal_before_1, vertices, vertices_dim);

    int dir0 = directed_edge_direction(new_tri0.data(), v_opp0, v_opp1);
    int dir1 = directed_edge_direction(new_tri1.data(), v_opp0, v_opp1);
    if (dir0 == 0 || dir1 == 0) {
        return false;
    }
    if (dir0 == dir1) {
        std::array<int, 3> alt_tri1 = {new_tri1[0], new_tri1[2], new_tri1[1]};
        auto alt_normal = normalize(compute_face_normal(alt_tri1.data(), vertices, vertices_dim));
        if (dot_product(alt_normal, normal_before_1) >= 0.0) {
            new_tri1 = alt_tri1;
        } else {
            return false;
        }
    }

    // Reject if new triangles already exist elsewhere (duplicate triangles)
    auto old_key0 = make_triangle_key(tri0[0], tri0[1], tri0[2]);
    auto old_key1 = make_triangle_key(tri1[0], tri1[1], tri1[2]);
    auto new_key0 = make_triangle_key(new_tri0[0], new_tri0[1], new_tri0[2]);
    auto new_key1 = make_triangle_key(new_tri1[0], new_tri1[1], new_tri1[2]);

    if (new_key0 == new_key1) {
        return false;
    }

    if ((new_key0 != old_key0 && new_key0 != old_key1 && triangle_set.find(new_key0) != triangle_set.end()) ||
        (new_key1 != old_key0 && new_key1 != old_key1 && triangle_set.find(new_key1) != triangle_set.end())) {
        return false;
    }

    // Reject if new edge already has two adjacent triangles
    auto new_edge = std::make_pair(std::min(v_opp0, v_opp1), std::max(v_opp0, v_opp1));
    auto it_new_edge = edge_map.find(new_edge);
    if (it_new_edge != edge_map.end()) {
        const EdgeInfo& info = it_new_edge->second;
        bool has_two = (info.tri0 != -1 && info.tri1 != -1);
        bool is_same_pair =
            (info.tri0 == edge_info.tri0 || info.tri0 == edge_info.tri1 || info.tri0 == -1) &&
            (info.tri1 == edge_info.tri0 || info.tri1 == edge_info.tri1 || info.tri1 == -1);
        if (has_two && !is_same_pair) {
            return false;
        }
    }
    
    auto normal_after_0 = normalize(compute_face_normal(new_tri0.data(), vertices, vertices_dim));
    auto normal_after_1 = normalize(compute_face_normal(new_tri1.data(), vertices, vertices_dim));
    
    // Dihedral angle threshold: reject if it gets significantly worse or too sharp
    const double dihedral_dot_threshold = 0.2;  // ~78° max dihedral angle
    const double dihedral_tolerance = 0.05;  // Allow small degradation
    
    // Check dihedral angles with neighboring triangles
    // For tri0, check its other two edges (not the shared edge)
    for (int i = 0; i < 3; ++i) {
        if (i == edge_info.opp0) continue;  // Skip the shared edge
        
        int va = tri0[i];
        int vb = tri0[(i + 1) % 3];
        auto edge = std::make_pair(std::min(va, vb), std::max(va, vb));
        
        auto it = edge_map.find(edge);
        if (it != edge_map.end()) {
            const EdgeInfo& neighbor_info = it->second;
            int neighbor_tri = (neighbor_info.tri0 == edge_info.tri0) ? neighbor_info.tri1 : neighbor_info.tri0;
            
            if (neighbor_tri != -1 && neighbor_tri != edge_info.tri1) {
                if (neighbor_tri < 0 || neighbor_tri >= triangle_count) {
                    continue;
                }
                const int* neighbor = &faces[neighbor_tri * 3];
                const double* pn0 = &vertices[neighbor[0] * vertices_dim];
                const double* pn1 = &vertices[neighbor[1] * vertices_dim];
                const double* pn2 = &vertices[neighbor[2] * vertices_dim];
                auto neighbor_normal = normalize(compute_normal(pn0, pn1, pn2, vertices_dim));
                
                double dot_before = dot_product(normal_before_0, neighbor_normal);
                double dot_after = dot_product(normal_after_0, neighbor_normal);
                
                // Reject if dihedral angle gets significantly worse or becomes too sharp
                if (dot_after < dot_before - dihedral_tolerance || dot_after < dihedral_dot_threshold) {
                    return false;
                }
            }
        }
    }
    
    // Check dihedral angles for tri1
    for (int i = 0; i < 3; ++i) {
        if (i == edge_info.opp1) continue;  // Skip the shared edge
        
        int va = tri1[i];
        int vb = tri1[(i + 1) % 3];
        auto edge = std::make_pair(std::min(va, vb), std::max(va, vb));
        
        auto it = edge_map.find(edge);
        if (it != edge_map.end()) {
            const EdgeInfo& neighbor_info = it->second;
            int neighbor_tri = (neighbor_info.tri0 == edge_info.tri1) ? neighbor_info.tri1 : neighbor_info.tri0;
            
            if (neighbor_tri != -1 && neighbor_tri != edge_info.tri0) {
                if (neighbor_tri < 0 || neighbor_tri >= triangle_count) {
                    continue;
                }
                const int* neighbor = &faces[neighbor_tri * 3];
                const double* pn0 = &vertices[neighbor[0] * vertices_dim];
                const double* pn1 = &vertices[neighbor[1] * vertices_dim];
                const double* pn2 = &vertices[neighbor[2] * vertices_dim];
                auto neighbor_normal = normalize(compute_normal(pn0, pn1, pn2, vertices_dim));
                
                double dot_before = dot_product(normal_before_1, neighbor_normal);
                double dot_after = dot_product(normal_after_1, neighbor_normal);
                
                // Reject if dihedral angle gets significantly worse or becomes too sharp
                if (dot_after < dot_before - dihedral_tolerance || dot_after < dihedral_dot_threshold) {
                    return false;
                }
            }
        }
    }

    // Ensure the two new triangles are not too sharp relative to each other
    if (dot_product(normal_after_0, normal_after_1) < dihedral_dot_threshold) {
        return false;
    }
    
    // Now check if the flip would minimize the maximum angle
    // Define lambda to compute maximum angle in a triangle
    auto compute_max_angle_in_tri = [&](const double* v0, const double* v1, const double* v2) -> double {
        for (int d = 0; d < vertices_dim; ++d) {
            if (!std::isfinite(v0[d]) || !std::isfinite(v1[d]) || !std::isfinite(v2[d])) {
                return M_PI;
            }
        }
        std::array<double, 3> e01{}, e12{}, e20{};
        for (int d = 0; d < vertices_dim; ++d) {
            e01[d] = v1[d] - v0[d];
            e12[d] = v2[d] - v1[d];
            e20[d] = v0[d] - v2[d];
        }
        
        double len01 = 0.0, len12 = 0.0, len20 = 0.0;
        for (int d = 0; d < vertices_dim; ++d) {
            len01 += e01[d] * e01[d];
            len12 += e12[d] * e12[d];
            len20 += e20[d] * e20[d];
        }
        len01 = std::sqrt(len01);
        len12 = std::sqrt(len12);
        len20 = std::sqrt(len20);
        
        if (len01 < 1e-12 || len12 < 1e-12 || len20 < 1e-12) {
            return M_PI;  // Degenerate triangle
        }
        
        double dot0 = 0.0, dot1 = 0.0, dot2 = 0.0;
        for (int d = 0; d < vertices_dim; ++d) {
            dot0 += e01[d] * (-e20[d]);
            dot1 += e12[d] * (-e01[d]);
            dot2 += e20[d] * (-e12[d]);
        }
        
        double denom0 = len01 * len20;
        double denom1 = len12 * len01;
        double denom2 = len20 * len12;
        if (denom0 < 1e-12 || denom1 < 1e-12 || denom2 < 1e-12) {
            return M_PI;
        }

        double cos0 = std::clamp(dot0 / denom0, -1.0, 1.0);
        double cos1 = std::clamp(dot1 / denom1, -1.0, 1.0);
        double cos2 = std::clamp(dot2 / denom2, -1.0, 1.0);

        if (!std::isfinite(cos0)) cos0 = 1.0;
        if (!std::isfinite(cos1)) cos1 = 1.0;
        if (!std::isfinite(cos2)) cos2 = 1.0;
        
        double angle0 = std::acos(cos0);
        double angle1 = std::acos(cos1);
        double angle2 = std::acos(cos2);
        return std::max({angle0, angle1, angle2});
    };
    
    double max_angle_before_0 = compute_max_angle_in_tri(p_opp0, p_shared0, p_shared1);
    double max_angle_before_1 = compute_max_angle_in_tri(p_opp1, p_shared1, p_shared0);
    double max_angle_before = std::max(max_angle_before_0, max_angle_before_1);
    
    // Compute maximum angles after flip
    double max_angle_after_0 = compute_max_angle_in_tri(
        &vertices[new_tri0[0] * vertices_dim],
        &vertices[new_tri0[1] * vertices_dim],
        &vertices[new_tri0[2] * vertices_dim]);
    double max_angle_after_1 = compute_max_angle_in_tri(
        &vertices[new_tri1[0] * vertices_dim],
        &vertices[new_tri1[1] * vertices_dim],
        &vertices[new_tri1[2] * vertices_dim]);
    double max_angle_after = std::max(max_angle_after_0, max_angle_after_1);
    
    // Flip if it reduces the maximum angle (with small threshold to avoid oscillation)
    const double threshold = 1e-6;
    return (max_angle_after < max_angle_before - threshold);
}

// Perform the edge flip operation
inline void flip_edge(
    std::vector<int>& faces,
    std::unordered_map<std::pair<int, int>, EdgeInfo, EdgeHash>& edge_map,
    const std::pair<int, int>& edge,
    std::span<const double> vertices,
    int vertices_dim,
    std::unordered_set<std::array<int, 3>, TriangleKeyHash>& triangle_set)
{
    auto it = edge_map.find(edge);
    if (it == edge_map.end()) {
        return;  // Edge not found
    }
    
    EdgeInfo& info = it->second;
    if (info.tri0 == -1 || info.tri1 == -1) {
        return;  // Boundary edge
    }
    const int tri0_idx = info.tri0;
    const int tri1_idx = info.tri1;
    const int opp0_idx = info.opp0;
    const int opp1_idx = info.opp1;

    if (faces.size() % 3 != 0 || vertices_dim <= 0) {
        return;
    }
    const int vertex_count = static_cast<int>(vertices.size()) / vertices_dim;
    const int triangle_count = static_cast<int>(faces.size()) / 3;
    if (vertex_count <= 0) {
        return;
    }
    if (tri0_idx < 0 || tri0_idx >= triangle_count ||
        tri1_idx < 0 || tri1_idx >= triangle_count) {
        return;
    }
    if (opp0_idx < 0 || opp0_idx > 2 ||
        opp1_idx < 0 || opp1_idx > 2) {
        return;
    }
    
    int* tri0 = &faces[tri0_idx * 3];
    int* tri1 = &faces[tri1_idx * 3];

    for (int i = 0; i < 3; ++i) {
        if (!is_valid_index(tri0[i], vertex_count) || !is_valid_index(tri1[i], vertex_count)) {
            return;
        }
    }
    
    // Get vertices
    int v_opp0 = tri0[opp0_idx];
    int v_opp1 = tri1[opp1_idx];
    int v_shared0 = tri0[(opp0_idx + 1) % 3];
    int v_shared1 = tri0[(opp0_idx + 2) % 3];
    
    // Store old triangles (before flip)
    std::array<int, 3> old_tri0{tri0[0], tri0[1], tri0[2]};
    std::array<int, 3> old_tri1{tri1[0], tri1[1], tri1[2]};

    // Remove old triangle references from edge map safely
    auto remove_triangle_from_edge = [&](const std::pair<int, int>& e, int tri_idx) {
        auto it_edge = edge_map.find(e);
        if (it_edge == edge_map.end()) {
            return;
        }
        if (it_edge->second.tri0 == tri_idx) {
            it_edge->second.tri0 = -1;
            it_edge->second.opp0 = -1;
        } else if (it_edge->second.tri1 == tri_idx) {
            it_edge->second.tri1 = -1;
            it_edge->second.opp1 = -1;
        }
        if (it_edge->second.tri0 == -1 && it_edge->second.tri1 == -1) {
            edge_map.erase(it_edge);
        }
    };

    for (int i = 0; i < 3; ++i) {
        int va = old_tri0[i];
        int vb = old_tri0[(i + 1) % 3];
        remove_triangle_from_edge({std::min(va, vb), std::max(va, vb)}, tri0_idx);
    }
    for (int i = 0; i < 3; ++i) {
        int va = old_tri1[i];
        int vb = old_tri1[(i + 1) % 3];
        remove_triangle_from_edge({std::min(va, vb), std::max(va, vb)}, tri1_idx);
    }
    
    // Preserve original triangle normals
    auto normal_before_0 = normalize(compute_face_normal(tri0, vertices, vertices_dim));
    auto normal_before_1 = normalize(compute_face_normal(tri1, vertices, vertices_dim));

    // Update triangles (flip the edge)
    // New tri0: v_opp0, v_opp1, v_shared0
    // New tri1: v_opp1, v_opp0, v_shared1
    tri0[0] = v_opp0;
    tri0[1] = v_opp1;
    tri0[2] = v_shared0;
    
    tri1[0] = v_opp1;
    tri1[1] = v_opp0;
    tri1[2] = v_shared1;


    // Fix winding to match original normals
    orient_triangle_to_match_normal(tri0, normal_before_0, vertices, vertices_dim);
    orient_triangle_to_match_normal(tri1, normal_before_1, vertices, vertices_dim);

    int dir0 = directed_edge_direction(tri0, v_opp0, v_opp1);
    int dir1 = directed_edge_direction(tri1, v_opp0, v_opp1);
    if (dir0 == 0 || dir1 == 0) {
        return;
    }
    if (dir0 == dir1) {
        std::swap(tri1[1], tri1[2]);
        auto n1 = normalize(compute_face_normal(tri1, vertices, vertices_dim));
        if (dot_product(n1, normal_before_1) < 0.0) {
            // Revert if normal would flip
            std::swap(tri1[1], tri1[2]);
            return;
        }
    }

    // Update triangle set
    triangle_set.erase(make_triangle_key(old_tri0[0], old_tri0[1], old_tri0[2]));
    triangle_set.erase(make_triangle_key(old_tri1[0], old_tri1[1], old_tri1[2]));
    triangle_set.insert(make_triangle_key(tri0[0], tri0[1], tri0[2]));
    triangle_set.insert(make_triangle_key(tri1[0], tri1[1], tri1[2]));
    
    // Rebuild edge map for affected triangles
    // We need to be careful: some edges might be shared with other triangles
    for (int tri_idx : {tri0_idx, tri1_idx}) {
        const int* tri = &faces[tri_idx * 3];
        for (int i = 0; i < 3; ++i) {
            int va = tri[i];
            int vb = tri[(i + 1) % 3];
            if (!is_valid_index(va, vertex_count) || !is_valid_index(vb, vertex_count)) {
                continue;
            }
            auto e = std::make_pair(std::min(va, vb), std::max(va, vb));
            
            // Check if this edge already exists in the map (from another triangle)
            auto it_edge = edge_map.find(e);
            if (it_edge == edge_map.end()) {
                // New edge, create entry
                EdgeInfo new_info;
                new_info.tri0 = tri_idx;
                new_info.opp0 = (i + 2) % 3;
                edge_map[e] = new_info;
            } else {
                // Edge exists, add this triangle as the second adjacent
                if (it_edge->second.tri0 == -1) {
                    it_edge->second.tri0 = tri_idx;
                    it_edge->second.opp0 = (i + 2) % 3;
                } else if (it_edge->second.tri1 == -1) {
                    it_edge->second.tri1 = tri_idx;
                    it_edge->second.opp1 = (i + 2) % 3;
                }
                // If both tri0 and tri1 are already set, this is a non-manifold situation
                // which shouldn't happen in a valid manifold mesh
            }
        }
    }
}

// Build edge-to-triangle adjacency map
inline std::unordered_map<std::pair<int, int>, EdgeInfo, EdgeHash> build_edge_map(
    const std::vector<int>& faces)
{
    std::unordered_map<std::pair<int, int>, EdgeInfo, EdgeHash> edge_map;
    int num_triangles = static_cast<int>(faces.size() / 3);
    
    for (int tri_idx = 0; tri_idx < num_triangles; ++tri_idx) {
        const int* tri = &faces[tri_idx * 3];
        for (int i = 0; i < 3; ++i) {
            int va = tri[i];
            int vb = tri[(i + 1) % 3];
            auto edge = std::make_pair(std::min(va, vb), std::max(va, vb));
            
            auto& info = edge_map[edge];
            if (info.tri0 == -1) {
                info.tri0 = tri_idx;
                info.opp0 = (i + 2) % 3;  // Opposite vertex index
            } else if (info.tri1 == -1) {
                info.tri1 = tri_idx;
                info.opp1 = (i + 2) % 3;
            } else {
                // Non-manifold edge (appears in more than 2 triangles)
                // Skip this edge - don't print warnings as they may be false positives
                // during edge flipping operations
            }
        }
    }
    
    return edge_map;
}

std::vector<int> mesh_optimize_by_edge_flipping(
    std::span<const double> vertices,
    int vertices_dim,
    std::span<const int> faces_in,
    int max_iterations)
{
    if (vertices_dim <= 0 || vertices.size() % vertices_dim != 0) {
        return std::vector<int>(faces_in.begin(), faces_in.end());
    }
    const int vertex_count = static_cast<int>(vertices.size()) / vertices_dim;

    if (faces_in.size() % 3 != 0 || vertex_count <= 0) {
        return std::vector<int>(faces_in.begin(), faces_in.end());
    }

    for (size_t i = 0; i < faces_in.size(); ++i) {
        if (!is_valid_index(faces_in[i], vertex_count)) {
            return std::vector<int>(faces_in.begin(), faces_in.end());
        }
    }

    // Copy input faces to working buffer
    std::vector<int> faces(faces_in.begin(), faces_in.end());

    // Track triangle set to avoid duplicates
    std::unordered_set<std::array<int, 3>, TriangleKeyHash> triangle_set;
    triangle_set.reserve(faces.size() / 3 * 2);
    for (size_t i = 0; i + 2 < faces.size(); i += 3) {
        triangle_set.insert(make_triangle_key(faces[i], faces[i + 1], faces[i + 2]));
    }
    
    // Build initial edge map
    auto edge_map = build_edge_map(faces);
    
    int iteration = 0;
    int total_flips = 0;
    
    // Main optimization loop: repeat until no more flips are possible
    while (iteration < max_iterations) {
        // Rebuild candidate edges from current edge map
        std::vector<std::pair<int, int>> candidate_edges;
        candidate_edges.reserve(edge_map.size());
        for (const auto& [edge, info] : edge_map) {
            if (info.tri0 != -1 && info.tri1 != -1) {
                candidate_edges.push_back(edge);
            }
        }
        
        int flips_this_iteration = 0;
        
        // Try to flip each candidate edge
        for (const auto& edge : candidate_edges) {
            auto it_edge = edge_map.find(edge);
            if (it_edge == edge_map.end()) {
                continue;
            }

            const EdgeInfo& info = it_edge->second;
            if (should_flip_edge(faces, info, vertices, vertices_dim, edge_map, triangle_set)) {
                flip_edge(faces, edge_map, edge, vertices, vertices_dim, triangle_set);
                ++flips_this_iteration;
            }
        }
        
        total_flips += flips_this_iteration;
        ++iteration;
        
        // If no flips happened, we've converged - stop iteration
        if (flips_this_iteration == 0) {
            break;
        }
    }
    
    return faces;
}

} // namespace cpgeo
