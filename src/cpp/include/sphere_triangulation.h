#pragma once

#include <span>
#include <vector>
#include <array>
#include <string>
#include <unordered_set>
#include "triangulation.h"
#include "mesh_utils.h"
#include "mesh_edge_flip.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace cpgeo {

class SphereTriangulation {
private:

    std::span<const double> sphere_points;  // x, y, z coordinates
    int num_points;
    std::vector<int> triangles;  // resulting triangle indices
    
    // Helper functions
    void stereographicProjection(std::span<double, 2> uv_point, int point_idx, bool north) const;
    std::vector<int> triangulateGivenPoints(std::span<const double> uv_points) const;

public:

    /*
    * Constructor
    ** @param sphere_points_span: span of input points on the sphere (x, y, z)
    */
    SphereTriangulation(std::span<const double> sphere_points_span);
    
    /* Perform triangulation on the sphere */
    void triangulate();

    size_t size() const;

    /* Get the resulting triangle indices 
    ** @param results: span to store the resulting triangle indices
    */
    void getTriangleIndices(std::span<int> results) const;

    /* Export the triangulated mesh to an OBJ file 
    ** @param filename: output OBJ file name
    */
    void exportToObj(const std::string& filename) const;
};

}  // namespace cpgeo
