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
    std::vector<std::vector<int>> query_point_batch(const std::span<const double> query_points);
    
    /**
     * @brief Query a single point against the tree
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @return std::vector<int> List of indices of knots influencing the query point
     */
    std::vector<int> query_point(double x, double y, double z) const;

    /**
     * @brief Get stored thresholds
     * 
     * @return std::vector<double> Copy of threshold storage
     */
    const std::vector<double>& get_thresholds() const {
        return threshold_storage;
    }

    /**
     * @brief Get stored knots
     * 
     * @return std::vector<double> Copy of knots storage
     */
    const std::vector<double>& get_knots() const {
        return knots_storage;
    }

    void print_tree_structure() const;
    void print_tree_stats() const;

private:
    std::vector<double> knots_storage;
    std::vector<double> threshold_storage;
    std::unique_ptr<Node> root;

    static constexpr int MAX_DEPTH = 5;
    static constexpr size_t MAX_POINTS = 100;

    static int getOctant(double x, double y, double z, double x_mid, double y_mid, double z_mid);
    static bool sphere_box_intersect(double x, double y, double z, double r, const Node& node);
    
    void subdivide(Node* node, int depth);
    const Node* find_leaf_node(double x, double y, double z) const;
    void print_tree_structure(const Node* node, int depth, int& node_index) const;
};

} // namespace cpgeo
