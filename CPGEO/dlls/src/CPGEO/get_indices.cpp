#include "pch.h"
#include "vector"
#include <omp.h>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>

#include <fstream>

#include <cmath>
#include <string>
#include <iomanip>

using namespace std;

class Space_tree {
private:
    static const int MAX_DEPTH = 6;  // Maximum depth
    static const int MAX_POINTS = 30;  // Maximum points in a leaf node

    struct Node {
        double xmin, xmax, ymin, ymax, zmin, zmax;  // Bounding box
        vector<int> point_indices;                  // Points in the node
        Node* children[8] = { nullptr };            // Eight child nodes
        bool is_leaf = true;                        // Is it a leaf node

        ~Node() {
            for (int i = 0; i < 8; i++) {
                if (children[i]) delete children[i];
            }
        }
    };

    Node* root;
    vector<double> knots_ptr;       // Pointer to knots data
    vector<double> threshold_ptr;   // Pointer to threshold data
    int knots_count;         // Number of knots
    double max_threshold;    // Maximum threshold

    // Get the octant where the point is located
    int getOctant(double x, double y, double z, double x_mid, double y_mid, double z_mid) {
        int octant = 0;
        if (x >= x_mid) octant |= 1;
        if (y >= y_mid) octant |= 2;
        if (z >= z_mid) octant |= 4;
        return octant;
    }

    // Check if the sphere intersects with the node's bounding box
    bool sphere_box_intersect(double x, double y, double z, double r, Node* node) {
        // Calculate the distance from the sphere center to the nearest point on the bounding box
        double closest_x = max(node->xmin, min(x, node->xmax));
        double closest_y = max(node->ymin, min(y, node->ymax));
        double closest_z = max(node->zmin, min(z, node->zmax));

        // Calculate the squared distance
        double dx = closest_x - x;
        double dy = closest_y - y;
        double dz = closest_z - z;
        double dist_squared = dx * dx + dy * dy + dz * dz;

        // If the distance is less than or equal to the radius, they intersect
        return dist_squared <= r * r;
    }

    // Subdivide the node, improved version
    void subdivide(Node* node, int depth) {
        // Termination condition: reach maximum depth or point count below threshold
        if (depth >= MAX_DEPTH || node->point_indices.size() <= MAX_POINTS) {
            node->is_leaf = true;
            return;
        }

        // Calculate the center point of the node
        double x_mid = (node->xmin + node->xmax) / 2;
        double y_mid = (node->ymin + node->ymax) / 2;
        double z_mid = (node->zmin + node->zmax) / 2;

        // Create 8 child nodes
        for (int i = 0; i < 8; i++) {
            node->children[i] = new Node();
            Node* child = node->children[i];

            // Set the bounding box of the child node
            child->xmin = (i & 1) ? x_mid : node->xmin;
            child->xmax = (i & 1) ? node->xmax : x_mid;
            child->ymin = (i & 2) ? y_mid : node->ymin;
            child->ymax = (i & 2) ? node->ymax : y_mid;
            child->zmin = (i & 4) ? z_mid : node->zmin;
            child->zmax = (i & 4) ? node->zmax : z_mid;
        }

        // Assign points to all possible child nodes
        for (int idx : node->point_indices) {
            double x = knots_ptr[idx * 3];
            double y = knots_ptr[idx * 3 + 1];
            double z = knots_ptr[idx * 3 + 2];
            double r = threshold_ptr[idx];

            // Check if the point's influence range intersects with the child node
            for (int i = 0; i < 8; i++) {
                if (sphere_box_intersect(x, y, z, r, node->children[i])) {
                    node->children[i]->point_indices.push_back(idx);
                }
            }
        }

        // Check if all child nodes contain the same points as the parent node
        bool all_same = true;
        for (int i = 0; i < 8; i++) {
            if (node->children[i]->point_indices.size() != node->point_indices.size()) {
                all_same = false;
                break;
            }
        }

        // If all child nodes contain the same point set, stop subdividing
        if (all_same) {
            for (int i = 0; i < 8; i++) {
                delete node->children[i];
                node->children[i] = nullptr;
            }
            node->is_leaf = true;
            return;
        }

        // Mark as non-leaf node
        node->is_leaf = false;

        // Recursively subdivide child nodes
        for (int i = 0; i < 8; i++) {
            if (!node->children[i]->point_indices.empty()) {
                subdivide(node->children[i], depth + 1);
            }
        }
    }

    // Find the leaf node containing the point
    Node* find_leaf_node(double x, double y, double z) {
        Node* current = root;

        while (!current->is_leaf) {
            double x_mid = (current->xmin + current->xmax) / 2;
            double y_mid = (current->ymin + current->ymax) / 2;
            double z_mid = (current->zmin + current->zmax) / 2;

            int octant = getOctant(x, y, z, x_mid, y_mid, z_mid);

            if (current->children[octant]) {
                current = current->children[octant];
            }
            else {
                // If there is no corresponding child node, the current node is the best leaf node
                break;
            }
        }

        return current;
    }

    // Find all knots that affect a single query point
    void query_point(vector<int>& results, double x, double y, double z) {
        // 1. Find the leaf node containing the query point
        Node* leaf = find_leaf_node(x, y, z);

        // 2. Check if all knots in the leaf node affect the query point
        for (int idx : leaf->point_indices) {
            double kx = knots_ptr[idx * 3];
            double ky = knots_ptr[idx * 3 + 1];
            double kz = knots_ptr[idx * 3 + 2];
            double r = threshold_ptr[idx];

            // Calculate the squared distance
            double dx = x - kx;
            double dy = y - ky;
            double dz = z - kz;
            double dist_squared = dx * dx + dy * dy + dz * dz;

            // If the distance is within the influence radius, add to the results
            if (dist_squared <= r * r) {
                results.push_back(idx);
            }
        }
    }

    // Recursive function to print the tree structure
    void print_tree_structure(Node* node, int depth, int& node_index) {
        if (!node) return;

        // Print node information
        std::cout << std::string(depth * 2, ' ') << "Node " << node_index++ << " at depth " << depth
            << (node->is_leaf ? " (LEAF)" : "") << ":\n";
        std::cout << std::string(depth * 2, ' ') << "Bounding Box: [("
            << node->xmin << ", " << node->ymin << ", " << node->zmin << "), ("
            << node->xmax << ", " << node->ymax << ", " << node->zmax << ")]\n";
        std::cout << std::string(depth * 2, ' ') << "Points: " << node->point_indices.size() << "\n";

        if (node->is_leaf && node->point_indices.size() < 10) {
            // Print the indices of the first few points
            std::cout << std::string(depth * 2, ' ') << "Sample indices: ";
            for (size_t i = 0; i < min(size_t(5), node->point_indices.size()); i++) {
                std::cout << node->point_indices[i] << " ";
            }
            if (node->point_indices.size() > 5) std::cout << "...";
            std::cout << "\n";
        }

        std::cout << std::endl;

        // Recursively print child nodes
        for (int i = 0; i < 8; i++) {
            if (node->children[i]) {
                print_tree_structure(node->children[i], depth + 1, node_index);
            }
        }
    }

public:
    Space_tree(double* _knots, double* _threshold, int _num_knots){

		// Store the pointers to knots and threshold data
		this->knots_count = _num_knots;
		this->knots_ptr.reserve(_num_knots * 3);
		this->threshold_ptr.reserve(_num_knots);
		for (int i = 0; i < _num_knots; i++) {
			this->knots_ptr.push_back(_knots[i * 3]);
            this->knots_ptr.push_back(_knots[i * 3 + 1]);
			this->knots_ptr.push_back(_knots[i * 3 + 2]);
			this->threshold_ptr.push_back(_threshold[i]);
		}

        // Calculate the maximum threshold
        max_threshold = 0.0;
        for (int i = 0; i < knots_count; i++) {
            if (threshold_ptr[i] > max_threshold) {
                max_threshold = threshold_ptr[i];
            }
        }

        root = new Node();

        // Calculate the bounding box
        double xmin = 1e10, xmax = -1e10;
        double ymin = 1e10, ymax = -1e10;
        double zmin = 1e10, zmax = -1e10;

        for (int i = 0; i < knots_count; i++) {
            double x = knots_ptr[i * 3];
            double y = knots_ptr[i * 3 + 1];
            double z = knots_ptr[i * 3 + 2];
            double r = threshold_ptr[i];  // Consider the influence radius

            xmin = min(xmin, x - r);
            xmax = max(xmax, x + r);
            ymin = min(ymin, y - r);
            ymax = max(ymax, y + r);
            zmin = min(zmin, z - r);
            zmax = max(zmax, z + r);

            root->point_indices.push_back(i);
        }

        // Set the bounding box of the root node
        root->xmin = xmin;
        root->xmax = xmax;
        root->ymin = ymin;
        root->ymax = ymax;
        root->zmin = zmin;
        root->zmax = zmax;

        // Build the tree
        subdivide(root, 0);
    }

    ~Space_tree() {
		if (root) {
			delete root;
		}
    }

    // Find the knots that affect multiple query points
    void get_indices(vector<vector<int>>& results, double* points, int num_pts) {
        for (int i = 0; i < num_pts; i++) {
            results.push_back(vector<int>());
			results[i].reserve(100);
        }

#pragma omp parallel for num_threads(16) schedule(dynamic)
        for (int i = 0; i < num_pts; i++) {
            double x = points[i * 3];
            double y = points[i * 3 + 1];
            double z = points[i * 3 + 2];
            query_point(results[i], x, y, z);
        }

    }

    // Print the tree structure
    void print_tree_structure() {
        int node_index = 0;
        print_tree_structure(root, 0, node_index);
    }

    // Print tree statistics
    void print_tree_stats() {
        int total_nodes = 0;
        int leaf_nodes = 0;
        int max_depth_found = 0;
        int min_points = knots_count;
        int max_points = 0;
        int total_points = 0;

        // Recursively traverse the tree and collect statistics
        std::function<void(Node*, int)> count_nodes = [&](Node* node, int depth) {
            if (!node) return;

            total_nodes++;
            max_depth_found = max(max_depth_found, depth);

            if (node->is_leaf) {
                leaf_nodes++;
                min_points = min(min_points, (int)node->point_indices.size());
                max_points = max(max_points, (int)node->point_indices.size());
                total_points += node->point_indices.size();
            }

            for (int i = 0; i < 8; i++) {
                if (node->children[i]) {
                    count_nodes(node->children[i], depth + 1);
                }
            }
            };

        count_nodes(root, 0);

        double avg_points = leaf_nodes > 0 ? (double)total_points / leaf_nodes : 0;

        cout << "\nTree statistics:" << endl;
        cout << "Total nodes: " << total_nodes << endl;
        cout << "Leaf nodes: " << leaf_nodes << endl;
        cout << "Maximum depth: " << max_depth_found << endl;
        cout << "Minimum points in leaf nodes: " << min_points << endl;
        cout << "Maximum points in leaf nodes: " << max_points << endl;
        cout << "Average points in leaf nodes: " << avg_points << endl;
    }
};

void* build_trees(double* knots, double* threshold, int num_knots)
{
    // Create a new tree
    Space_tree* tree = new Space_tree(knots, threshold, num_knots);

    return (void*)tree;
}

void delete_trees(void* base_tree)
{
    delete (Space_tree*)base_tree;
}



vector<int> results_temp;
void cal_indices(void* base_tree, double* points, int num_pts, int* sizes) {
    
    Space_tree* tree = (Space_tree*)base_tree;

    vector<vector<int>> indices_temp;
    indices_temp.reserve(num_pts);
    tree->get_indices(indices_temp, points, num_pts);

	int total_size = 0;
    for (int i = 0; i < num_pts; ++i) {
        sizes[i] = indices_temp[i].size();
		total_size += sizes[i];
    }
    
	results_temp.clear();
	results_temp.reserve(total_size * 2);

    for (int i = 0; i < indices_temp.size(); i++) {
        for (int j = 0; j < indices_temp[i].size(); j++) {

            results_temp.push_back(indices_temp[i][j]);
        }
    }

    for (int i = 0; i < indices_temp.size(); i++) {
        for (int j = 0; j < indices_temp[i].size(); j++) {

            results_temp.push_back(i);
        }
    }


    
}

void get_indices(int* results) {
	int total_size = results_temp.size();
	for (int i = 0; i < total_size; i++) {
		results[i] = (int)results_temp[i];
	}
}
