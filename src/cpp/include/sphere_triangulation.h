#pragma once

#include <span>
#include <vector>
#include <array>
#include <string>
#include <unordered_set>

namespace cpgeo {

class SphereTriangulation {
private:
    using Triangle = std::array<int, 3>;
    
    std::vector<Triangle> triangles;
    std::span<const double> sphere_points;  // x, y, z coordinates
    int num_points;
    
    // Helper functions
    std::pair<double, double> stereographicProjection(int point_idx, bool north) const;
    double calculateTriangleQuality(const Triangle& tri) const;
    std::vector<std::pair<int, int>> extractBoundaryEdges() const;
    void triangulateHemisphere(bool north, const std::vector<std::pair<int, int>>& boundary_edges = {});
    void filterLowQualityTriangles(double quality_threshold);
    void improveQualityByEdgeFlipping(int excluded_point_idx, double region_radius, int max_iterations = 10);

public:
    SphereTriangulation(std::span<const double> sphere_points_span);
    
    void triangulate();
    size_t size() const;
    void getTriangleIndices(std::span<int> results) const;
    void exportToObj(const std::string& filename) const;
};

}  // namespace cpgeo
