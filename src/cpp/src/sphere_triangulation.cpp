#include "sphere_triangulation.h"
#include "triangulation.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace cpgeo {

struct EdgeHash {
    std::size_t operator()(const std::pair<int, int>& edge) const noexcept {
        return std::hash<int>{}(edge.first) ^ (std::hash<int>{}(edge.second) << 1);
    }
};

SphereTriangulation::SphereTriangulation(std::span<const double> sphere_points_span)
    : sphere_points(sphere_points_span), num_points(static_cast<int>(sphere_points_span.size() / 3)) {
    triangles.reserve(num_points * 4);
}

std::pair<double, double> SphereTriangulation::stereographicProjection(int point_idx, bool north) const {
    double x = sphere_points[point_idx * 3];
    double y = sphere_points[point_idx * 3 + 1];
    double z = sphere_points[point_idx * 3 + 2];
    
    if (north) {
        double denom = 1.0 + z;
        if (std::abs(denom) < 1e-10) {
            return {x * 1e6, y * 1e6};
        }
        return {x / denom, y / denom};
    } else {
        double denom = 1.0 - z;
        if (std::abs(denom) < 1e-10) {
            return {x * 1e6, y * 1e6};
        }
        return {x / denom, y / denom};
    }
}

double SphereTriangulation::calculateTriangleQuality(const Triangle& tri) const {
    double x0 = sphere_points[tri[0] * 3];
    double y0 = sphere_points[tri[0] * 3 + 1];
    double z0 = sphere_points[tri[0] * 3 + 2];
    
    double x1 = sphere_points[tri[1] * 3];
    double y1 = sphere_points[tri[1] * 3 + 1];
    double z1 = sphere_points[tri[1] * 3 + 2];
    
    double x2 = sphere_points[tri[2] * 3];
    double y2 = sphere_points[tri[2] * 3 + 1];
    double z2 = sphere_points[tri[2] * 3 + 2];
    
    double dx01 = x1 - x0, dy01 = y1 - y0, dz01 = z1 - z0;
    double dx12 = x2 - x1, dy12 = y2 - y1, dz12 = z2 - z1;
    double dx20 = x0 - x2, dy20 = y0 - y2, dz20 = z0 - z2;
    
    double edge0 = std::sqrt(dx01*dx01 + dy01*dy01 + dz01*dz01);
    double edge1 = std::sqrt(dx12*dx12 + dy12*dy12 + dz12*dz12);
    double edge2 = std::sqrt(dx20*dx20 + dy20*dy20 + dz20*dz20);
    
    double cx = dy01 * dz12 - dz01 * dy12;
    double cy = dz01 * dx12 - dx01 * dz12;
    double cz = dx01 * dy12 - dy01 * dx12;
    double area = 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
    
    if (area < 1e-10) return 0.0;
    
    double edge_sum_sq = edge0*edge0 + edge1*edge1 + edge2*edge2;
    return (4.0 * std::sqrt(3.0) * area) / edge_sum_sq;
}

std::vector<std::pair<int, int>> SphereTriangulation::extractBoundaryEdges() const {
    std::unordered_map<std::pair<int, int>, int, EdgeHash> edge_count;
    
    for (const auto& tri : triangles) {
        for (int j = 0; j < 3; ++j) {
            int a = tri[j];
            int b = tri[(j + 1) % 3];
            auto edge = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
            ++edge_count[edge];
        }
    }
    
    std::vector<std::pair<int, int>> boundary_edges;
    for (const auto& [edge, count] : edge_count) {
        if (count == 1) {
            boundary_edges.push_back(edge);
        }
    }
    
    return boundary_edges;
}

void SphereTriangulation::filterLowQualityTriangles(double quality_threshold) {
    triangles.erase(
        std::remove_if(triangles.begin(), triangles.end(),
            [this, quality_threshold](const Triangle& tri) {
                return calculateTriangleQuality(tri) < quality_threshold;
            }),
        triangles.end()
    );
}

void SphereTriangulation::improveQualityByEdgeFlipping(int excluded_point_idx, double region_radius, int max_iterations) {
    // std::cout << "Improving mesh quality by edge flipping (near excluded point)..." << std::endl;
    
    // Get excluded point coordinates
    double ex = sphere_points[excluded_point_idx * 3];
    double ey = sphere_points[excluded_point_idx * 3 + 1];
    double ez = sphere_points[excluded_point_idx * 3 + 2];
    
    // Mark triangles in the region to optimize (near excluded point)
    std::vector<bool> in_region(triangles.size(), false);
    int region_tri_count = 0;
    
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        
        // Check if any vertex is close to the excluded point
        for (int v : tri) {
            double vx = sphere_points[v * 3];
            double vy = sphere_points[v * 3 + 1];
            double vz = sphere_points[v * 3 + 2];
            
            // Calculate dot product (cosine of angle between points on unit sphere)
            double dot = ex * vx + ey * vy + ez * vz;
            
            // If dot > cos(region_radius), the point is within the region
            if (dot > std::cos(region_radius)) {
                in_region[i] = true;
                region_tri_count++;
                break;
            }
        }
    }
    
    // std::cout << "  Region contains " << region_tri_count << " triangles ("
    //           << (100.0 * region_tri_count / triangles.size()) << "% of mesh)" << std::endl;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        int flips = 0;
        
        // Build edge-to-triangles map (only for region triangles and their neighbors)
        std::unordered_map<std::pair<int, int>, std::vector<size_t>, EdgeHash> edge_to_tris;
        
        for (size_t i = 0; i < triangles.size(); ++i) {
            const auto& tri = triangles[i];
            for (int j = 0; j < 3; ++j) {
                int a = tri[j];
                int b = tri[(j + 1) % 3];
                auto edge = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
                edge_to_tris[edge].push_back(i);
            }
        }
        
        // Try to flip each internal edge (only if at least one adjacent triangle is in region)
        std::vector<bool> processed(triangles.size(), false);
        
        for (const auto& [edge, tri_indices] : edge_to_tris) {
            if (tri_indices.size() != 2) continue;  // Only flip internal edges
            
            size_t tri1_idx = tri_indices[0];
            size_t tri2_idx = tri_indices[1];
            
            // Skip if neither triangle is in the optimization region
            if (!in_region[tri1_idx] && !in_region[tri2_idx]) continue;
            
            if (processed[tri1_idx] || processed[tri2_idx]) continue;
            
            const auto& tri1 = triangles[tri1_idx];
            const auto& tri2 = triangles[tri2_idx];
            
            // Find the four vertices of the quadrilateral
            int shared_a = edge.first;
            int shared_b = edge.second;
            int opposite1 = -1, opposite2 = -1;
            
            for (int v : tri1) {
                if (v != shared_a && v != shared_b) {
                    opposite1 = v;
                    break;
                }
            }
            
            for (int v : tri2) {
                if (v != shared_a && v != shared_b) {
                    opposite2 = v;
                    break;
                }
            }
            
            if (opposite1 == -1 || opposite2 == -1) continue;
            
            // Calculate quality before flip
            double q1_before = calculateTriangleQuality(tri1);
            double q2_before = calculateTriangleQuality(tri2);
            double min_quality_before = std::min(q1_before, q2_before);
            
            // Create new triangles after flip
            Triangle new_tri1 = {opposite1, opposite2, shared_a};
            Triangle new_tri2 = {opposite1, opposite2, shared_b};
            
            // Check if the new edge already exists (would create non-manifold edge)
            auto new_edge = (opposite1 < opposite2) ? std::make_pair(opposite1, opposite2) : std::make_pair(opposite2, opposite1);
            if (edge_to_tris.count(new_edge) > 0) {
                continue;  // Skip this flip to avoid non-manifold edge
            }
            
            // Calculate quality after flip
            double q1_after = calculateTriangleQuality(new_tri1);
            double q2_after = calculateTriangleQuality(new_tri2);
            double min_quality_after = std::min(q1_after, q2_after);
            
            // Flip if quality improves
            if (min_quality_after > min_quality_before * 1.05) {  // 5% improvement threshold
                // Ensure correct orientation for new triangles
                double x0 = sphere_points[new_tri1[0] * 3];
                double y0 = sphere_points[new_tri1[0] * 3 + 1];
                double z0 = sphere_points[new_tri1[0] * 3 + 2];
                
                double x1 = sphere_points[new_tri1[1] * 3];
                double y1 = sphere_points[new_tri1[1] * 3 + 1];
                double z1 = sphere_points[new_tri1[1] * 3 + 2];
                
                double x2 = sphere_points[new_tri1[2] * 3];
                double y2 = sphere_points[new_tri1[2] * 3 + 1];
                double z2 = sphere_points[new_tri1[2] * 3 + 2];
                
                double cx = (x0 + x1 + x2) / 3.0;
                double cy = (y0 + y1 + y2) / 3.0;
                double cz = (z0 + z1 + z2) / 3.0;
                
                double nx = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0);
                double ny = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0);
                double nz = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                
                if (nx * cx + ny * cy + nz * cz < 0) {
                    std::swap(new_tri1[1], new_tri1[2]);
                }
                
                x0 = sphere_points[new_tri2[0] * 3];
                y0 = sphere_points[new_tri2[0] * 3 + 1];
                z0 = sphere_points[new_tri2[0] * 3 + 2];
                
                x1 = sphere_points[new_tri2[1] * 3];
                y1 = sphere_points[new_tri2[1] * 3 + 1];
                z1 = sphere_points[new_tri2[1] * 3 + 2];
                
                x2 = sphere_points[new_tri2[2] * 3];
                y2 = sphere_points[new_tri2[2] * 3 + 1];
                z2 = sphere_points[new_tri2[2] * 3 + 2];
                
                cx = (x0 + x1 + x2) / 3.0;
                cy = (y0 + y1 + y2) / 3.0;
                cz = (z0 + z1 + z2) / 3.0;
                
                nx = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0);
                ny = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0);
                nz = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                
                if (nx * cx + ny * cy + nz * cz < 0) {
                    std::swap(new_tri2[1], new_tri2[2]);
                }
                
                // Apply the flip
                triangles[tri1_idx] = new_tri1;
                triangles[tri2_idx] = new_tri2;
                processed[tri1_idx] = true;
                processed[tri2_idx] = true;
                flips++;
            }
        }
        
        // std::cout << "  Iteration " << (iter + 1) << ": " << flips << " edges flipped" << std::endl;
        
        if (flips == 0) {
        // std::cout << "  Converged after " << (iter + 1) << " iterations" << std::endl;
            break;
        }
    }
}

void SphereTriangulation::triangulateHemisphere(bool north, const std::vector<std::pair<int, int>>& boundary_edges) {
    // Not used in new implementation
}

void SphereTriangulation::triangulate() {
    triangles.clear();
    
    if (num_points < 4) {
        std::cerr << "Error: Need at least 4 points" << std::endl;
        return;
    }
    
    // std::cout << "Triangulating entire sphere using stereographic projection..." << std::endl;
    
    // Find point closest to south pole to use as projection center
    int south_pole_idx = 0;
    double min_z = sphere_points[2];
    for (int i = 1; i < num_points; ++i) {
        double z = sphere_points[i * 3 + 2];
        if (z < min_z) {
            min_z = z;
            south_pole_idx = i;
        }
    }
    
    // std::cout << "  Excluding point " << south_pole_idx << " (z=" << min_z << ") from triangulation" << std::endl;
    
    // Project all points except the south pole
    std::vector<int> point_indices;
    std::vector<double> projected_points;
    
    for (int i = 0; i < num_points; ++i) {
        if (i == south_pole_idx) continue;  // Skip south pole
        
        point_indices.push_back(i);
        auto [px, py] = stereographicProjection(i, true);
        projected_points.push_back(px);
        projected_points.push_back(py);
    }
    
    // Perform Delaunay triangulation
    DelaunayTriangulation delaunay(projected_points);
    delaunay.triangulate();
    size_t tri_count = delaunay.size();
    std::vector<int> tri_indices(tri_count * 3);
    delaunay.getTriangleIndices(tri_indices.data());
    
    // std::cout << "  Generated " << tri_count << " triangles" << std::endl;
    
    // Convert to global indices and ensure correct orientation
    for (size_t i = 0; i < tri_count; ++i) {
        Triangle tri;
        tri[0] = point_indices[tri_indices[i * 3]];
        tri[1] = point_indices[tri_indices[i * 3 + 1]];
        tri[2] = point_indices[tri_indices[i * 3 + 2]];
        
        // Get triangle vertices
        double x0 = sphere_points[tri[0] * 3];
        double y0 = sphere_points[tri[0] * 3 + 1];
        double z0 = sphere_points[tri[0] * 3 + 2];
        
        double x1 = sphere_points[tri[1] * 3];
        double y1 = sphere_points[tri[1] * 3 + 1];
        double z1 = sphere_points[tri[1] * 3 + 2];
        
        double x2 = sphere_points[tri[2] * 3];
        double y2 = sphere_points[tri[2] * 3 + 1];
        double z2 = sphere_points[tri[2] * 3 + 2];
        
        // Compute triangle center
        double cx = (x0 + x1 + x2) / 3.0;
        double cy = (y0 + y1 + y2) / 3.0;
        double cz = (z0 + z1 + z2) / 3.0;
        
        // Compute normal
        double nx = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0);
        double ny = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0);
        double nz = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        
        // Ensure normal points outward
        if (nx * cx + ny * cy + nz * cz < 0) {
            std::swap(tri[1], tri[2]);
        }
        
        triangles.push_back(tri);
    }
    
    // std::cout << "Total triangles: " << triangles.size() << std::endl;
    
    // Fill the hole around the excluded point
    // std::cout << "Filling hole around excluded point..." << std::endl;
    std::unordered_map<std::pair<int, int>, int, EdgeHash> edge_count;
    for (const auto& tri : triangles) {
        for (int j = 0; j < 3; ++j) {
            int a = tri[j];
            int b = tri[(j + 1) % 3];
            auto edge = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
            edge_count[edge]++;
        }
    }
    
    // Find boundary edges
    std::vector<int> boundary_loop;
    std::unordered_map<int, std::vector<int>> adjacency;
    
    for (const auto& [edge, count] : edge_count) {
        if (count == 1) {
            adjacency[edge.first].push_back(edge.second);
            adjacency[edge.second].push_back(edge.first);
        }
    }
    
    // Build ordered boundary loop
    if (!adjacency.empty()) {
        int start = adjacency.begin()->first;
        boundary_loop.push_back(start);
        int current = adjacency[start][0];
        
        while (current != start && boundary_loop.size() < num_points) {
            boundary_loop.push_back(current);
            for (int next : adjacency[current]) {
                if (boundary_loop.size() < 2 || next != boundary_loop[boundary_loop.size() - 2]) {
                    current = next;
                    break;
                }
            }
        }
    }
    
    // std::cout << "  Found boundary loop with " << boundary_loop.size() << " vertices" << std::endl;
    
    // Create fan triangulation from excluded point to boundary
    size_t filled = 0;
    for (size_t i = 0; i < boundary_loop.size(); ++i) {
        Triangle tri;
        tri[0] = south_pole_idx;
        tri[1] = boundary_loop[i];
        tri[2] = boundary_loop[(i + 1) % boundary_loop.size()];
        
        // Check orientation
        double x0 = sphere_points[tri[0] * 3];
        double y0 = sphere_points[tri[0] * 3 + 1];
        double z0 = sphere_points[tri[0] * 3 + 2];
        
        double x1 = sphere_points[tri[1] * 3];
        double y1 = sphere_points[tri[1] * 3 + 1];
        double z1 = sphere_points[tri[1] * 3 + 2];
        
        double x2 = sphere_points[tri[2] * 3];
        double y2 = sphere_points[tri[2] * 3 + 1];
        double z2 = sphere_points[tri[2] * 3 + 2];
        
        double cx = (x0 + x1 + x2) / 3.0;
        double cy = (y0 + y1 + y2) / 3.0;
        double cz = (z0 + z1 + z2) / 3.0;
        
        double nx = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0);
        double ny = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0);
        double nz = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        
        if (nx * cx + ny * cy + nz * cz < 0) {
            std::swap(tri[1], tri[2]);
        }
        
        triangles.push_back(tri);
        filled++;
    }
    
    // std::cout << "  Filled hole with " << filled << " triangles" << std::endl;
    // std::cout << "Final total: " << triangles.size() << " triangles" << std::endl;
    
    // Post-process: improve quality by edge flipping (only near south pole)
    // Optimize within 30 degrees (pi/6 radians) of the excluded point
    const double pi = 3.14159265358979323846;
    improveQualityByEdgeFlipping(south_pole_idx, pi / 12.0, 10);
    
    // Calculate and report mesh quality statistics
    // std::cout << "Computing mesh quality statistics..." << std::endl;
    double min_quality = 1.0;
    double max_quality = 0.0;
    double avg_quality = 0.0;
    int low_quality_count = 0;
    
    for (const auto& tri : triangles) {
        double q = calculateTriangleQuality(tri);
        min_quality = std::min(min_quality, q);
        max_quality = std::max(max_quality, q);
        avg_quality += q;
        if (q < 0.3) low_quality_count++;
    }
    
    avg_quality /= triangles.size();
    
    // std::cout << "  Min quality: " << min_quality << std::endl;
    // std::cout << "  Max quality: " << max_quality << std::endl;
    // std::cout << "  Avg quality: " << avg_quality << std::endl;
    // std::cout << "  Low quality triangles (q<0.3): " << low_quality_count
    //           << " (" << (100.0 * low_quality_count / triangles.size()) << "%)" << std::endl;
}

size_t SphereTriangulation::size() const {
    return triangles.size();
}

void SphereTriangulation::getTriangleIndices(int* results) const {
    for (size_t i = 0; i < triangles.size(); ++i) {
        results[i * 3] = triangles[i][0];
        results[i * 3 + 1] = triangles[i][1];
        results[i * 3 + 2] = triangles[i][2];
    }
}

void SphereTriangulation::exportToObj(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file " << filename << std::endl;
        return;
    }
    
    file << "# Generated by CPGEO Sphere Triangulation\n";
    file << "# Vertices: " << num_points << "\n";
    file << "# Triangles: " << triangles.size() << "\n\n";
    
    for (int i = 0; i < num_points; ++i) {
        file << "v " << sphere_points[i*3] << " " 
             << sphere_points[i*3+1] << " " 
             << sphere_points[i*3+2] << "\n";
    }
    
    file << "\n";
    
    for (size_t i = 0; i < triangles.size(); ++i) {
        file << "f " << (triangles[i][0] + 1) << " " 
             << (triangles[i][1] + 1) << " " 
             << (triangles[i][2] + 1) << "\n";
    }
    
    file.close();
    // std::cout << "Exported " << triangles.size() << " triangles to " << filename << std::endl;
}

}  // namespace cpgeo
