#pragma once

#include <span>
#include <vector>
#include <array>
#include <string>
namespace cpgeo {
class DelaunayTriangulation {
private:
    using Triangle = std::array<int, 3>;
    
    std::vector<Triangle> triangles;
    std::span<const double> nodes;
    std::array<double, 8> super_triangle_nodes;
    int num_original_nodes;
    
    inline std::pair<double, double> getNode(int idx) const;
    inline Triangle makeCounterClockwise(int v0, int v1, int v2) const;
    bool isInCircumcircle(const Triangle& tri, int point_idx) const;
    void bowyerWatsonStep(int node_idx);
    void removeSuperTriangleVertices();

public:
    DelaunayTriangulation(std::span<const double> nodes_span);
    
    void triangulate();
    size_t size() const;
    void getTriangleIndices(int* results) const;
    void exportToObj(const std::string& filename) const;
};

}