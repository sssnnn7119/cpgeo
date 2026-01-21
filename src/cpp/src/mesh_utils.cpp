#include "mesh_utils.h"
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <string>

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