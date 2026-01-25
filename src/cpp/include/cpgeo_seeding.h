#pragma once

#include <tuple>
#include <vector>
#include <span>

#include "space_tree.h"
#include "cpgeo_mapping.h"
#include "mesh_edge_flip.h"
#include "mesh_utils.h"
#include "sphere_triangulation.h"


namespace cpgeo {

/** 
 * @brief Smooth the vertices of a spherical mesh to improve uniformity
 * 
 * @param vertices_sphere Sphere vertices (flat array of x,y,z coordinates)
 * @param faces Face indices (triplets of vertex indices)
 * @param control_points Control points influencing the mesh (flat array of x,y,z coordinates)
 * @param tree SpaceTree for efficient spatial queries
 */
void vertice_smoothing(
    std::vector<double>& vertices_sphere,
    std::span<int> faces,
    std::span<const double> control_points,
    SpaceTree& tree
);

/** 
 * @brief Generate a uniformly meshed sphere surface based on initial vertices and control points
 * 
 * @param init_vertices_sphere Initial sphere vertices (flat array of x,y,z coordinates)
 * @param control_points Control points influencing the mesh (flat array of x,y,z coordinates)
 * @param tree SpaceTree for efficient spatial queries
 * @param seed_size Desired size for seeding points
 * @param max_iterations Maximum number of iterations for mesh uniforming
 * @return std::tuple<std::vector<double>, std::vector<int>> Tuple containing:
 *         - Vector of sphere vertices (flat array of x,y,z coordinates)
 *         - Vector of face indices (triplets of vertex indices)
 */
std::tuple<std::vector<double>, std::vector<int>> uniformlyMesh(
    std::span<double> init_vertices_sphere, 
    std::span<const double> control_points,
    SpaceTree& tree,
    double seed_size,
    int max_iterations
);
} // namespace cpgeo