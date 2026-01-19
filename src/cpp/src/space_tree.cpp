#include "space_tree.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <limits>
#include <iomanip>
#include <stdexcept>

namespace cpgeo {

SpaceTree::SpaceTree(std::span<const double> knots, std::span<const double> thresholds) {
    if (knots.size() != thresholds.size() * 3) {
        throw std::invalid_argument("Knots size must be 3 times the size of thresholds");
    }

    size_t num_knots = thresholds.size();
    
    // Initialize storage
    knots_storage.assign(knots.begin(), knots.end());
    threshold_storage.assign(thresholds.begin(), thresholds.end());

    root = std::make_unique<Node>();

    // Calculate the bounding box for all spheres (point + influence radius)
    double xmin = std::numeric_limits<double>::max(), xmax = std::numeric_limits<double>::lowest();
    double ymin = std::numeric_limits<double>::max(), ymax = std::numeric_limits<double>::lowest();
    double zmin = std::numeric_limits<double>::max(), zmax = std::numeric_limits<double>::lowest();

    // Initialize root point indices
    root->point_indices.reserve(num_knots);

    for (size_t i = 0; i < num_knots; i++) {
        double x = knots_storage[i * 3];
        double y = knots_storage[i * 3 + 1];
        double z = knots_storage[i * 3 + 2];
        double r = threshold_storage[i];

        xmin = std::min(xmin, x - r);
        xmax = std::max(xmax, x + r);
        ymin = std::min(ymin, y - r);
        ymax = std::max(ymax, y + r);
        zmin = std::min(zmin, z - r);
        zmax = std::max(zmax, z + r);

        root->point_indices.push_back(static_cast<int>(i));
    }

    // Set bounding box with a small margin to strict inequalities issues
    constexpr double epsilon = 1e-6;
    root->xmin = xmin - epsilon; root->xmax = xmax + epsilon;
    root->ymin = ymin - epsilon; root->ymax = ymax + epsilon;
    root->zmin = zmin - epsilon; root->zmax = zmax + epsilon;

    // Build the tree
    subdivide(root.get(), 0);
}

int SpaceTree::getOctant(double x, double y, double z, double x_mid, double y_mid, double z_mid) {
    int octant = 0;
    if (x >= x_mid) octant |= 1;
    if (y >= y_mid) octant |= 2;
    if (z >= z_mid) octant |= 4;
    return octant;
}

bool SpaceTree::sphere_box_intersect(double x, double y, double z, double r, const Node& node) {
    // Find closest point on AABB to sphere center
    double closest_x = std::clamp(x, node.xmin, node.xmax);
    double closest_y = std::clamp(y, node.ymin, node.ymax);
    double closest_z = std::clamp(z, node.zmin, node.zmax);

    double dx = x - closest_x;
    double dy = y - closest_y;
    double dz = z - closest_z;
    
    double dist_squared = dx * dx + dy * dy + dz * dz;
    return dist_squared <= r * r;
}

void SpaceTree::subdivide(Node* node, int depth) {
    if (depth >= MAX_DEPTH || node->point_indices.size() <= MAX_POINTS) {
        node->is_leaf = true;
        return;
    }

    double x_mid = (node->xmin + node->xmax) * 0.5;
    double y_mid = (node->ymin + node->ymax) * 0.5;
    double z_mid = (node->zmin + node->zmax) * 0.5;

    // Initialize children
    for (int i = 0; i < 8; i++) {
        node->children[i] = std::make_unique<Node>();
        auto& child = node->children[i];
        
        child->xmin = (i & 1) ? x_mid : node->xmin;
        child->xmax = (i & 1) ? node->xmax : x_mid;
        child->ymin = (i & 2) ? y_mid : node->ymin;
        child->ymax = (i & 2) ? node->ymax : y_mid;
        child->zmin = (i & 4) ? z_mid : node->zmin;
        child->zmax = (i & 4) ? node->zmax : z_mid;
    }

    // Distribute points to children
    for (int idx : node->point_indices) {
        double x = knots_storage[idx * 3];
        double y = knots_storage[idx * 3 + 1];
        double z = knots_storage[idx * 3 + 2];
        double r = threshold_storage[idx];

        for (int i = 0; i < 8; i++) {
            if (sphere_box_intersect(x, y, z, r, *node->children[i])) {
                node->children[i]->point_indices.push_back(idx);
            }
        }
    }

    bool all_same = true;
    size_t parent_count = node->point_indices.size();
    
    // Check if children have same points as parent (infinite loop prevention)
    for (const auto& child : node->children) {
        if (child->point_indices.size() != parent_count) {
            all_same = false;
            break;
        }
    }

    if (all_same) {
        for (auto& child : node->children) {
            child.reset();
        }
        node->is_leaf = true;
        return;
    }

    node->is_leaf = false;
    // Clear parent indices to save memory
    std::vector<int>().swap(node->point_indices); 

    for (auto& child : node->children) {
        if (!child->point_indices.empty()) {
            subdivide(child.get(), depth + 1);
        } else {
            child->is_leaf = true;
        }
    }
}

const SpaceTree::Node* SpaceTree::find_leaf_node(double x, double y, double z) const {
    const Node* current = root.get();

    while (!current->is_leaf) {
        double x_mid = (current->xmin + current->xmax) * 0.5;
        double y_mid = (current->ymin + current->ymax) * 0.5;
        double z_mid = (current->zmin + current->zmax) * 0.5;

        int octant = getOctant(x, y, z, x_mid, y_mid, z_mid);
        auto& child = current->children[octant];
        
        if (child) {
            current = child.get();
        } else {
            break;
        }
    }
    return current;
}

void SpaceTree::query_point(std::vector<int>& results, double x, double y, double z) const {
    const Node* leaf = find_leaf_node(x, y, z);
    
    for (int idx : leaf->point_indices) {
        double kx = knots_storage[idx * 3];
        double ky = knots_storage[idx * 3 + 1];
        double kz = knots_storage[idx * 3 + 2];
        double r = threshold_storage[idx];

        double dx = x - kx;
        double dy = y - ky;
        double dz = z - kz;
        
        if (dx*dx + dy*dy + dz*dz <= r*r) {
            results.push_back(idx);
        }
    }
}

void SpaceTree::compute_indices(const std::span<const double> query_points) {
    size_t num_pts = query_points.size() / 3;
    
    // We must ensure the outer vector has the correct size
    if (this->query_results.size() != num_pts) {
        this->query_results.resize(num_pts);
    }

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(num_pts); i++) {
        this->query_results[i].clear();
        this->query_results[i].reserve(50); 

        double x = query_points[i * 3];
        double y = query_points[i * 3 + 1];
        double z = query_points[i * 3 + 2];
        query_point(this->query_results[i], x, y, z);
    }
}

void SpaceTree::print_tree_structure() const {
    int node_idx = 0;
    print_tree_structure(root.get(), 0, node_idx);
}

void SpaceTree::print_tree_structure(const Node* node, int depth, int& node_index) const {
    if (!node) return;

    std::cout << std::string(depth * 2, ' ') << "Node " << node_index++ << " at depth " << depth
              << (node->is_leaf ? " (LEAF)" : "") << ":\n";
    std::cout << std::string(depth * 2, ' ') << "Bounding Box: [("
              << node->xmin << ", " << node->ymin << ", " << node->zmin << "), ("
              << node->xmax << ", " << node->ymax << ", " << node->zmax << ")]\n";
    
    if (node->is_leaf) {
        std::cout << std::string(depth * 2, ' ') << "Points: " << node->point_indices.size() << "\n";
        if (node->point_indices.size() < 10 && !node->point_indices.empty()) {
            std::cout << std::string(depth * 2, ' ') << "Indices: ";
            for (size_t i = 0; i < std::min(size_t(5), node->point_indices.size()); i++) {
                std::cout << node->point_indices[i] << " ";
            }
            if (node->point_indices.size() > 5) std::cout << "...";
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            if (child) {
                print_tree_structure(child.get(), depth + 1, node_index);
            }
        }
    }
}

void SpaceTree::print_tree_stats() const {
    int total_nodes = 0;
    int leaf_nodes = 0;
    int max_depth_found = 0;
    int min_points = std::numeric_limits<int>::max();
    int max_points = 0;
    size_t total_points = 0;

    auto traverse = [&](auto&& self, const Node* node, int depth) -> void {
        if (!node) return;
        
        total_nodes++;
        max_depth_found = std::max(max_depth_found, depth);

        if (node->is_leaf) {
            leaf_nodes++;
            int count = static_cast<int>(node->point_indices.size());
            min_points = std::min(min_points, count);
            max_points = std::max(max_points, count);
            total_points += count;
        } else {
            for (const auto& child : node->children) {
                self(self, child.get(), depth + 1);
            }
        }
    };

    traverse(traverse, root.get(), 0);

    double avg_points = leaf_nodes > 0 ? static_cast<double>(total_points) / leaf_nodes : 0.0;
    if (leaf_nodes == 0) min_points = 0;

    std::cout << "Tree Statistics:\n"
              << "  Total nodes: " << total_nodes << "\n"
              << "  Leaf nodes:  " << leaf_nodes << "\n"
              << "  Max depth:   " << max_depth_found << "\n"
              << "  Leaf points (min/max/avg): " << min_points << " / " << max_points << " / " 
              << std::fixed << std::setprecision(1) << avg_points << "\n";
}

} // namespace cpgeo
