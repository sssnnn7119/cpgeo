#include "triangulation.h"
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>

namespace cpgeo {

// Custom hash for edge pairs
struct EdgeHash {
    std::size_t operator()(const std::pair<int, int>& edge) const noexcept {
        return std::hash<int>{}(edge.first) ^ (std::hash<int>{}(edge.second) << 1);
    }
};

DelaunayTriangulation::DelaunayTriangulation(std::span<const double> nodes_span) 
    : nodes(nodes_span), num_original_nodes(static_cast<int>(nodes_span.size() / 2)) {
    triangles.reserve(num_original_nodes * 5);
}

inline std::pair<double, double> DelaunayTriangulation::getNode(int idx) const {
    if (idx < num_original_nodes) {
        return {nodes[idx * 2], nodes[idx * 2 + 1]};
    }
    int super_idx = idx - num_original_nodes;
    return {super_triangle_nodes[super_idx * 2], super_triangle_nodes[super_idx * 2 + 1]};
}

// Helper function to ensure triangle vertices are in counter-clockwise order
inline DelaunayTriangulation::Triangle DelaunayTriangulation::makeCounterClockwise(int v0, int v1, int v2) const {
    auto [x0, y0] = getNode(v0);
    auto [x1, y1] = getNode(v1);
    auto [x2, y2] = getNode(v2);
    
    // Calculate signed area (cross product)
    double signed_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    
    // If signed area is negative, vertices are clockwise, so swap v1 and v2
    if (signed_area < 0) {
        return {v0, v2, v1};
    }
    return {v0, v1, v2};
}

bool DelaunayTriangulation::isInCircumcircle(const Triangle& tri, int point_idx) const {
    auto [x, y] = getNode(point_idx);
    auto [x0, y0] = getNode(tri[0]);
    auto [x1, y1] = getNode(tri[1]);
    auto [x2, y2] = getNode(tri[2]);
    
    // Calculate circumcenter
    double d = 2.0 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1));
    if (std::abs(d) < 1e-10) return false;  // Degenerate triangle
    
    double cx = ((x0*x0 + y0*y0) * (y1 - y2) + (x1*x1 + y1*y1) * (y2 - y0) + (x2*x2 + y2*y2) * (y0 - y1)) / d;
    double cy = ((x0*x0 + y0*y0) * (x2 - x1) + (x1*x1 + y1*y1) * (x0 - x2) + (x2*x2 + y2*y2) * (x1 - x0)) / d;
    
    // Calculate radius squared
    double dx = x0 - cx;
    double dy = y0 - cy;
    double r_squared = dx*dx + dy*dy;
    
    // Check if point is inside
    dx = x - cx;
    dy = y - cy;
    return (dx*dx + dy*dy) < r_squared;
}

void DelaunayTriangulation::bowyerWatsonStep(int node_idx) {
    // Find triangles whose circumcircle contains the node
    std::vector<size_t> bad_triangles;
    bad_triangles.reserve(32);  // 预估大小，避免多次扩容
    
    const size_t num_triangles = triangles.size();
    
    // 当三角形数量较多时使用并行优化
    if (num_triangles > 100) {
        const int num_threads = 8;
        std::vector<std::vector<size_t>> thread_bad_triangles(num_threads);
        
        // 预分配每个线程的空间
        for (int t = 0; t < num_threads; ++t) {
            thread_bad_triangles[t].reserve(num_triangles / num_threads + 16);
        }
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for schedule(static)
            for (int i = 0; i < static_cast<int>(num_triangles); ++i) {
                if (isInCircumcircle(triangles[i], node_idx)) {
                    thread_bad_triangles[tid].push_back(static_cast<size_t>(i));
                }
            }
        }
        
        // 合并各线程结果（无需加锁）
        size_t total_size = 0;
        for (const auto& vec : thread_bad_triangles) {
            total_size += vec.size();
        }
        bad_triangles.reserve(total_size);
        
        for (auto& vec : thread_bad_triangles) {
            bad_triangles.insert(bad_triangles.end(), 
                               std::make_move_iterator(vec.begin()),
                               std::make_move_iterator(vec.end()));
        }
    } else {
        for (size_t i = 0; i < num_triangles; ++i) {
            if (isInCircumcircle(triangles[i], node_idx)) {
                bad_triangles.push_back(i);
            }
        }
    }
    
    if (bad_triangles.empty()) return;

    // Find the boundary edges (edges that appear only once)
    std::unordered_map<std::pair<int, int>, int, EdgeHash> edge_count;
    edge_count.reserve(bad_triangles.size() * 3);
    
    for (size_t tri_idx : bad_triangles) {
        const auto& tri = triangles[tri_idx];
        for (int j = 0; j < 3; ++j) {
            int a = tri[j];
            int b = tri[(j + 1) % 3];
            auto edge = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
            ++edge_count[edge];
        }
    }

    // 收集边界边
    std::vector<std::pair<int, int>> boundary_edges;
    boundary_edges.reserve(bad_triangles.size() * 2);
    for (const auto& [edge, count] : edge_count) {
        if (count == 1) {
            boundary_edges.emplace_back(edge);
        }
    }

    // Remove bad triangles (in reverse order to maintain indices)
    std::sort(bad_triangles.begin(), bad_triangles.end(), std::greater<size_t>());
    for (size_t idx : bad_triangles) {
        triangles.erase(triangles.begin() + idx);
    }

    // Add new triangles for each boundary edge
    triangles.reserve(triangles.size() + boundary_edges.size());
    for (const auto& edge : boundary_edges) {
        triangles.emplace_back(makeCounterClockwise(edge.first, edge.second, node_idx));
    }
}

void DelaunayTriangulation::removeSuperTriangleVertices() {
    // Remove all triangles that include any super triangle vertex
    triangles.erase(
        std::remove_if(triangles.begin(), triangles.end(),
            [this](const Triangle& tri) {
                return tri[0] >= num_original_nodes ||
                       tri[1] >= num_original_nodes ||
                       tri[2] >= num_original_nodes;
            }),
        triangles.end()
    );
}

void DelaunayTriangulation::triangulate() {
    triangles.clear();
    triangles.reserve(num_original_nodes * 2 + 10);  // 预估最终三角形数量
    
    if (num_original_nodes < 3) return;

    // Calculate bounding box
    double minX = nodes[0];
    double minY = nodes[1];
    double maxX = minX;
    double maxY = minY;
    
    // 当点数较多时使用并行归约
    if (num_original_nodes > 1000) {
        const int num_threads = 8;
        std::vector<double> thread_minX(num_threads, nodes[0]);
        std::vector<double> thread_minY(num_threads, nodes[1]);
        std::vector<double> thread_maxX(num_threads, nodes[0]);
        std::vector<double> thread_maxY(num_threads, nodes[1]);
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for schedule(static)
            for (int i = 1; i < num_original_nodes; ++i) {
                double x = nodes[i * 2];
                double y = nodes[i * 2 + 1];
                thread_minX[tid] = (x < thread_minX[tid]) ? x : thread_minX[tid];
                thread_minY[tid] = (y < thread_minY[tid]) ? y : thread_minY[tid];
                thread_maxX[tid] = (x > thread_maxX[tid]) ? x : thread_maxX[tid];
                thread_maxY[tid] = (y > thread_maxY[tid]) ? y : thread_maxY[tid];
            }
        }
        
        // 合并各线程结果
        for (int t = 0; t < num_threads; ++t) {
            minX = (thread_minX[t] < minX) ? thread_minX[t] : minX;
            minY = (thread_minY[t] < minY) ? thread_minY[t] : minY;
            maxX = (thread_maxX[t] > maxX) ? thread_maxX[t] : maxX;
            maxY = (thread_maxY[t] > maxY) ? thread_maxY[t] : maxY;
        }
    } else {
        for (int i = 1; i < num_original_nodes; ++i) {
            double x = nodes[i * 2];
            double y = nodes[i * 2 + 1];
            minX = (x < minX) ? x : minX;
            minY = (y < minY) ? y : minY;
            maxX = (x > maxX) ? x : maxX;
            maxY = (y > maxY) ? y : maxY;
        }
    }

    // Expand bounding box
    constexpr double margin = 1.0;
    minX -= margin;
    minY -= margin;
    maxX += margin;
    maxY += margin;

    // Create super triangle vertices
    super_triangle_nodes = {
        maxX, maxY,  // vertex 0
        minX, maxY,  // vertex 1
        minX, minY,  // vertex 2
        maxX, minY   // vertex 3
    };

    // Create initial super triangles
    int v0 = num_original_nodes;
    int v1 = num_original_nodes + 1;
    int v2 = num_original_nodes + 2;
    int v3 = num_original_nodes + 3;
    
    triangles.emplace_back(makeCounterClockwise(v0, v1, v2));
    triangles.emplace_back(makeCounterClockwise(v0, v2, v3));

    // Incrementally add each point
    for (int i = 0; i < num_original_nodes; ++i) {
        bowyerWatsonStep(i);
    }

    // Remove triangles connected to super triangle vertices
    removeSuperTriangleVertices();
}

size_t DelaunayTriangulation::size() const {
    return triangles.size();
}

void DelaunayTriangulation::getTriangleIndices(int* results) const {
    for (size_t i = 0; i < triangles.size(); ++i) {
        results[i * 3] = triangles[i][0];
        results[i * 3 + 1] = triangles[i][1];
        results[i * 3 + 2] = triangles[i][2];
    }
}

void DelaunayTriangulation::exportToObj(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file " << filename << std::endl;
        return;
    }
    
    file << "# Generated by CPGEO Delaunay Triangulation\n";
    file << "# Vertices: " << num_original_nodes << "\n";
    file << "# Triangles: " << triangles.size() << "\n\n";
    
    // Write vertices (v x y z)
    for (int i = 0; i < num_original_nodes; ++i) {
        file << "v " << nodes[i*2] << " " << nodes[i*2+1] << " 0.0\n";
    }
    
    file << "\n";
    
    // Write faces (f v1 v2 v3), OBJ indices start from 1
    for (size_t i = 0; i < triangles.size(); ++i) {
        file << "f " << (triangles[i][0] + 1) << " " 
             << (triangles[i][1] + 1) << " " 
             << (triangles[i][2] + 1) << "\n";
    }
    
    file.close();
    std::cout << "Exported " << triangles.size() << " triangles to " << filename << std::endl;
}

}  // namespace spgeo