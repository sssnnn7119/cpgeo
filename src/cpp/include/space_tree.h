#pragma once

#include <vector>
#include <memory>
#include <array>
#include <span>
#include <iostream>
#include "cpgeo.h"

namespace cpgeo {

class CPGEO_API SpaceTree {
public:
    struct Node {
        double xmin, xmax, ymin, ymax, zmin, zmax;
        std::vector<int> point_indices;
        std::array<std::unique_ptr<Node>, 8> children;
        bool is_leaf = false;
    };

    /**
     * @brief Construct a new Space_tree
     * 
     * @param knots Flat array of coordinates (x, y, z) size N*3
     * @param thresholds Array of influence radii size N
     */
    SpaceTree(std::span<const double> knots, std::span<const double> thresholds);

    /**
     * @brief Query points against the tree
     * 
     * @param query_points Flat array of coordinates (x1, y1, z1, x2, y2, z2, ...) size M*3
     * @return std::vector<std::vector<int>> List of indices for each query point
     */
    void compute_indices(const std::span<const double> query_points);
    const std::vector<std::vector<int>>& get_query_results() const {
        return query_results;
    }

    void print_tree_structure() const;
    void print_tree_stats() const;

private:
    std::vector<double> knots_storage;
    std::vector<double> threshold_storage;
    std::unique_ptr<Node> root;
    std::vector<std::vector<int>> query_results;

    static constexpr int MAX_DEPTH = 10;
    static constexpr size_t MAX_POINTS = 50;

    static int getOctant(double x, double y, double z, double x_mid, double y_mid, double z_mid);
    static bool sphere_box_intersect(double x, double y, double z, double r, const Node& node);
    
    void subdivide(Node* node, int depth);
    const Node* find_leaf_node(double x, double y, double z) const;
    void query_point(std::vector<int>& results, double x, double y, double z) const;
    void print_tree_structure(const Node* node, int depth, int& node_index) const;
};

} // namespace cpgeo
