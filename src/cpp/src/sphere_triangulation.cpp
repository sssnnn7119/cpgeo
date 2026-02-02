#include "sphere_triangulation.h"


namespace cpgeo {

SphereTriangulation::SphereTriangulation(std::span<const double> sphere_points_span)
    : sphere_points(sphere_points_span), num_points(static_cast<int>(sphere_points_span.size() / 3)) {
}

void SphereTriangulation::stereographicProjection(std::span<double, 2> uv_point, int point_idx, bool north) const {
    double x = sphere_points[point_idx * 3 + 0];
    double y = sphere_points[point_idx * 3 + 1];
    double z = sphere_points[point_idx * 3 + 2];

    double denom = north ? (1.0 - z) : (1.0 + z);
    if (denom == 0.0) {
        uv_point[0] = 0.0;
        uv_point[1] = 0.0;
    } else {
        uv_point[0] = x / denom;
        uv_point[1] = y / denom;
    }
}

std::vector<int> SphereTriangulation::triangulateGivenPoints(std::span<const double> uv_points) const {

    DelaunayTriangulation triangulator(uv_points);

    triangulator.triangulate();
    std::vector<int> triangle_indices(triangulator.size() * 3);
    triangulator.getTriangleIndices(triangle_indices.data());

    return triangle_indices;
}



void SphereTriangulation::triangulate() {
    
    std::vector<double> uv_north(num_points * 2);
    std::vector<double> uv_south(num_points * 2);

    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        stereographicProjection(std::span<double, 2>{&uv_north[i * 2], 2}, i, true);
        stereographicProjection(std::span<double, 2>{&uv_south[i * 2], 2}, i, false);
    }

    // Collect indices of points in the northern hemisphere
    std::vector<int> idx_north_part, idx_south_part;
    idx_north_part.reserve(num_points);  // reserve space
    for(int i = 0; i < num_points; ++i) {
        double z = sphere_points[i * 3 + 2];
        if (z <= 0) {
            idx_north_part.push_back(i);
        } else {
            idx_south_part.push_back(i);
        }
    }

    // Create UV points for northern hemisphere
    std::vector<double> uv_north_part(idx_north_part.size() * 2);
    std::unordered_map<int, int> north_idx_map;  // original idx to north part idx
    for (size_t i = 0; i < idx_north_part.size(); ++i) {
        uv_north_part[i * 2 + 0] = uv_north[idx_north_part[i] * 2 + 0];
        uv_north_part[i * 2 + 1] = uv_north[idx_north_part[i] * 2 + 1];
        north_idx_map[i] = idx_north_part[i];
    }

    std::vector<int> triangles_north = triangulateGivenPoints(uv_north_part);
    // Map back to original indices

    #pragma omp parallel for
    for (int i = 0; i < triangles_north.size(); ++i) {
        triangles_north[i] = idx_north_part[triangles_north[i]];
    }

    // Get the boundary edges from northern hemisphere triangulation
    auto boundary_edges = extractBoundaryLoops(std::span<const int>(triangles_north.data(), triangles_north.size()))[0];

    // Create UV points for southern hemisphere including boundary points
    std::vector<double> uv_south_part;
    uv_south_part.reserve((num_points + boundary_edges.size()) * 2);
    std::unordered_map<int, int> south_idx_map;  // original idx to south part idx
    for (size_t i = 0; i < idx_south_part.size(); ++i) {
        uv_south_part.push_back(uv_south[idx_south_part[i] * 2 + 0]);
        uv_south_part.push_back(uv_south[idx_south_part[i] * 2 + 1]);
        south_idx_map[i] = idx_south_part[i];
    }

    // include boundary points
    for (int idx : boundary_edges) {
        double u_boundary = uv_north[idx * 2 + 0];
        double v_boundary = uv_north[idx * 2 + 1];

        // Project to the unit circle in southern hemisphere
        // Transform the stereographic coordinate from one pole to the other
        // This corresponds to circle inversion: u_s = u_n / (u_n^2 + v_n^2)
        double denom = u_boundary * u_boundary + v_boundary * v_boundary;
        double u_south, v_south;
        if (denom > 1e-12) {
            u_south = u_boundary / denom * 3.0;
            v_south = v_boundary / denom * 3.0;
        } else {
            // Should not happen for boundary points near equator
             u_south = 0.0;
             v_south = 0.0;
        }

        uv_south_part.push_back(u_south);
        uv_south_part.push_back(v_south);
        south_idx_map[uv_south_part.size() / 2 - 1] = idx;  // map to original index
    }

    std::vector<int> triangles_south = triangulateGivenPoints(std::span<const double>(uv_south_part.data(), uv_south_part.size()));

    // Map back to original indices
    #pragma omp parallel for
    for (int i = 0; i < triangles_south.size(); ++i) {
        triangles_south[i] = south_idx_map[triangles_south[i]];
    }

    // Combine triangles from both hemispheres
    triangles.clear();
    triangles.reserve(triangles_north.size() + triangles_south.size());
    triangles.insert(triangles.end(), triangles_north.begin(), triangles_north.end());
    for(int idx=0; idx<triangles_south.size()/3; ++idx) {
        int v0 = triangles_south[idx * 3 + 0];
        int v1 = triangles_south[idx * 3 + 1];
        int v2 = triangles_south[idx * 3 + 2];

        triangles.push_back(v0);
        triangles.push_back(v2);
        triangles.push_back(v1);
        
    }

    triangles = mesh_optimize_by_edge_flipping(
        std::span<const double>(sphere_points.data(), sphere_points.size()),
        3,
        std::span<const int>(triangles.data(), triangles.size()),
        100);
    
}

void SphereTriangulation::getTriangleIndices(std::span<int> results) const {
    if (results.size() != triangles.size()) {
        throw std::runtime_error("Output span size does not match the number of triangle indices.");
    }
    std::copy(triangles.begin(), triangles.end(), results.begin());
}


size_t SphereTriangulation::size() const {
    return triangles.size() / 3;
}

void SphereTriangulation::exportToObj(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Write vertices
    for (int i = 0; i < num_points; ++i) {
        ofs << "v " << sphere_points[i * 3 + 0] << " "
                    << sphere_points[i * 3 + 1] << " "
                    << sphere_points[i * 3 + 2] << "\n";
    }

    // Write faces (triangles)
    for (size_t i = 0; i < triangles.size(); i += 3) {
        ofs << "f " << (triangles[i + 0] + 1) << " "
                    << (triangles[i + 1] + 1) << " "
                    << (triangles[i + 2] + 1) << "\n";
    }

    ofs.close();
}

} // namespace cpgeo