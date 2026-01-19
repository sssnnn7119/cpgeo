#include "triangle.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <array>
#include <span>
#include <limits>

using lf = double;

// Custom hash for edge pairs
struct EdgeHash {
    std::size_t operator()(const std::pair<int, int>& edge) const noexcept {
        return std::hash<int>{}(edge.first) ^ (std::hash<int>{}(edge.second) << 1);
    }
};

class DelaunayTriangulation {
private:
    std::vector<Triangle> triangles;
    std::span<const double> nodes;
    std::array<double, 8> super_triangle_nodes;  // 4 super triangle vertices (x,y) * 4
    int num_original_nodes;
    
    // Get node coordinates with super triangle support
    inline std::pair<double, double> getNode(int idx) const {
        if (idx < num_original_nodes) {
            return {nodes[idx * 2], nodes[idx * 2 + 1]};
        } else {
            int super_idx = idx - num_original_nodes;
            return {super_triangle_nodes[super_idx * 2], super_triangle_nodes[super_idx * 2 + 1]};
        }
    }
    
    // Create a span that includes both original nodes and super triangle nodes
    std::span<const double> getExtendedNodes() const {
        // This is a workaround: we'll create a temporary view
        // In practice, Triangle will use getNode through the extended index
        static thread_local std::vector<double> extended;
        extended.clear();
        extended.reserve((num_original_nodes + 4) * 2);
        extended.insert(extended.end(), nodes.begin(), nodes.end());
        extended.insert(extended.end(), super_triangle_nodes.begin(), super_triangle_nodes.end());
        return std::span<const double>(extended.data(), extended.size());
    }

    void bowyerWatsonStep(int node_idx) {
        // Find triangles whose circumcircle contains the node
        std::vector<int> bad_triangles;
        bad_triangles.reserve(triangles.size() / 4);  // Heuristic reserve
        
        for (size_t i = 0; i < triangles.size(); ++i) {
            if (triangles[i].isInCircumcircle(node_idx)) {
                bad_triangles.push_back(static_cast<int>(i));
            }
        }
        
        if (bad_triangles.empty()) return;

        // Find the boundary edges (edges that appear only once)
        std::unordered_map<std::pair<int, int>, int, EdgeHash> edge_count;
        edge_count.reserve(bad_triangles.size() * 3);
        
        for (int tri_idx : bad_triangles) {
            for (int j = 0; j < 3; ++j) {
                int a = triangles[tri_idx].ind[j];
                int b = triangles[tri_idx].ind[(j + 1) % 3];
                auto edge = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
                ++edge_count[edge];
            }
        }

        // Remove bad triangles (in reverse order to maintain indices)
        std::sort(bad_triangles.begin(), bad_triangles.end(), std::greater<int>());
        for (int idx : bad_triangles) {
            triangles.erase(triangles.begin() + idx);
        }

        // Add new triangles for each boundary edge
        auto extended_nodes = getExtendedNodes();
        for (const auto& [edge, count] : edge_count) {
            if (count == 1) {
                triangles.emplace_back(edge.first, edge.second, node_idx, extended_nodes);
            }
        }
    }

    void removeSuperTriangleVertices() {
        // Remove all triangles that include any super triangle vertex
        triangles.erase(
            std::remove_if(triangles.begin(), triangles.end(),
                [this](const Triangle& tri) {
                    return tri.ind[0] >= num_original_nodes ||
                           tri.ind[1] >= num_original_nodes ||
                           tri.ind[2] >= num_original_nodes;
                }),
            triangles.end()
        );
    }

public:
    DelaunayTriangulation(std::span<const double> nodes_span) 
        : nodes(nodes_span), num_original_nodes(static_cast<int>(nodes_span.size() / 2)) {
        triangles.reserve(num_original_nodes * 2);  // Heuristic: ~2 triangles per node
    }

    void triangulate() {
        triangles.clear();
        
        if (num_original_nodes < 3) return;

        // Calculate bounding box with margin
        double minX = std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double maxY = std::numeric_limits<double>::lowest();
        
        for (int i = 0; i < num_original_nodes; ++i) {
            double x = nodes[i * 2];
            double y = nodes[i * 2 + 1];
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
        }

        // Expand bounding box
        constexpr double margin = 1.0;
        minX -= margin;
        minY -= margin;
        maxX += margin;
        maxY += margin;

        // Create super triangle vertices (no copy, just store in array)
        super_triangle_nodes = {
            maxX, maxY,  // vertex 0
            minX, maxY,  // vertex 1
            minX, minY,  // vertex 2
            maxX, minY   // vertex 3
        };

        // Create initial super triangles
        auto extended_nodes = getExtendedNodes();
        int v0 = num_original_nodes;
        int v1 = num_original_nodes + 1;
        int v2 = num_original_nodes + 2;
        int v3 = num_original_nodes + 3;
        
        triangles.emplace_back(v0, v1, v2, extended_nodes);
        triangles.emplace_back(v0, v2, v3, extended_nodes);

        // Incrementally add each point
        for (int i = 0; i < num_original_nodes; ++i) {
            bowyerWatsonStep(i);
        }

        // Remove triangles connected to super triangle vertices
        removeSuperTriangleVertices();
    }

    size_t size() const {
        return triangles.size();
    }

    const Triangle& operator[](size_t i) const {
        return triangles[i];
    }
    
    void getTriangleIndices(int* results) const {
        for (size_t i = 0; i < triangles.size(); ++i) {
            results[i * 3] = triangles[i].ind[0];
            results[i * 3 + 1] = triangles[i].ind[1];
            results[i * 3 + 2] = triangles[i].ind[2];
        }
    }
};

// Thread-local storage for triangulation instance
thread_local std::unique_ptr<DelaunayTriangulation> g_triangulation;

void triangular_mesh(int* num_mesh, double* nodes, int num_nodes) {
    std::span<const double> nodes_span(nodes, num_nodes * 2);
    
    if (!g_triangulation || g_triangulation->size() == 0) {
        g_triangulation = std::make_unique<DelaunayTriangulation>(nodes_span);
    }
    
    g_triangulation->triangulate();
    num_mesh[0] = static_cast<int>(g_triangulation->size());
}

void get_triangular_mesh(int* results) {
    if (!g_triangulation) {
        return;
    }
    
    g_triangulation->getTriangleIndices(results);
}