#include "sphere_triangulation.h"
#include "triangulation.h"
#include "mesh_edge_flip.h"
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

void SphereTriangulation::set_points(std::span<const double> sphere_points_span) {
    sphere_points = sphere_points_span;
    num_points = static_cast<int>(sphere_points_span.size() / 3);
    triangles.clear();
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
    // Convert triangles to flat array for mesh_optimize_by_edge_flipping
    std::vector<int> faces_flat(triangles.size() * 3);
    for (size_t i = 0; i < triangles.size(); ++i) {
        faces_flat[i * 3] = triangles[i][0];
        faces_flat[i * 3 + 1] = triangles[i][1];
        faces_flat[i * 3 + 2] = triangles[i][2];
    }
    
    // Convert sphere_points span to vector for passing to mesh_optimize_by_edge_flipping
    std::vector<double> vertices(sphere_points.begin(), sphere_points.end());
    
    // Call the tested mesh edge flipping function
    auto optimized_faces = mesh_optimize_by_edge_flipping(
        std::span<const double>(vertices.data(), vertices.size()),
        3,  // vertices_dim = 3 for sphere points
        std::span<const int>(faces_flat.data(), faces_flat.size()),
        max_iterations
    );
    
    // Convert back to Triangle array
    triangles.clear();
    triangles.reserve(optimized_faces.size() / 3);
    for (size_t i = 0; i < optimized_faces.size() / 3; ++i) {
        Triangle tri;
        tri[0] = optimized_faces[i * 3];
        tri[1] = optimized_faces[i * 3 + 1];
        tri[2] = optimized_faces[i * 3 + 2];
        triangles.push_back(tri);
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

void SphereTriangulation::getTriangleIndices(std::span<int> results) const {
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
