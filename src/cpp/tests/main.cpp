#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <array>
#include <chrono>
#include <algorithm>
#include <random>
#include "cpgeo.h"
#include <math.h>
#include <time.h>

#include "cpgeo_seeding.h"
#include "space_tree.h"
#include <fstream>
#include <sstream>
#include <string>

const double M_PI = 3.14159265358979323846;

namespace TestWeight{
    bool test_weight_function(){
        int num_knots_dim = 10;
        double knot_limit = 2.0;

        std::vector<double> knots(num_knots_dim * num_knots_dim * num_knots_dim * 3, 0.0);
        std::vector<double> thresholds(num_knots_dim * num_knots_dim * num_knots_dim, 0.6);

        for (int x = 0; x < num_knots_dim; ++x) {
            for (int y = 0; y < num_knots_dim; ++y) {
                for (int z = 0; z < num_knots_dim; ++z) {
                    int idx = (x * num_knots_dim * num_knots_dim + y * num_knots_dim + z);
                    knots[idx * 3 + 0] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * x + 0.01;
                    knots[idx * 3 + 1] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * y;
                    knots[idx * 3 + 2] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * z;
                }
            }
        }

        std::array<double, 6> query_point = {1.0, 0.0, 0.0};
        
        auto tree = space_tree_create(
            knots.data(),
            static_cast<int>(knots.size() / 3),
            thresholds.data()
        );
        
        int num_indices = 0;
        space_tree_query_compute(
            tree,
            query_point.data(),
            2,
            &num_indices
        );

        std::vector<int> indices_cps(num_indices);
        std::vector<int> indices_pts(2 + 1);
        space_tree_query_get(
            tree,
            num_indices,
            indices_cps.data(),
            indices_pts.data()
        );

        space_tree_destroy(tree);

        std::vector<double> weights(num_indices, 0.0);
        std::vector<double> wdu(num_indices * 2);
		std::vector<double> wdu2(num_indices * 4);

        std::array<double, 4> query_point_2d = {2.0, 0.0};
        cpgeo_get_weights_derivative2(
            indices_cps.data(),
            indices_pts.data(),
            num_indices,
            knots.data(),
            static_cast<int>(knots.size() / 3),
            thresholds.data(),
            query_point_2d.data(),
            2,
            weights.data(),
            wdu.data(),
            wdu2.data()
        );

        // Verify weights
        for(int i=0;i<num_indices;i++){
            if (weights[i] < 1e-10) {
                continue;
            }
            std::cout << "Knot Index: (" << indices_cps[i] << ")"
                << " Weight: " << weights[i] << "\t\tWdu: (" << wdu[i * 2 + 0] << ", " << wdu[i * 2 + 1] << ")" 
                "\t\tWdu2: (" << wdu2[i * 4 + 0 * 2 + 0] << ")"
                << std::endl;

        }

        std::cout << std::endl;


        std::vector<double> _weights(num_indices, 0.0);
        std::vector<double> _wdu(num_indices * 2);
        std::vector<double> _wdu2(num_indices * 4);
        query_point_2d[0] += 1e-8;
        cpgeo_get_weights_derivative2(
            indices_cps.data(),
            indices_pts.data(),
            num_indices,
            knots.data(),
            static_cast<int>(knots.size() / 3),
            thresholds.data(),
            query_point_2d.data(),
            2,
            _weights.data(),
            _wdu.data(),
            _wdu2.data()
        );

        // Verify weights
        for (int i = 0; i < num_indices; i++) {
            if (weights[i] < 1e-10) {
                continue;
            }
            std::cout << "Knot Index: (" << indices_cps[i] << ")"
                << " Weight: " << (weights[i] - _weights[i])/1e-8 << "\t\tWdu: (" << (wdu[i * 2 + 0]- _wdu[i * 2 + 0])/1e-8 << ", " << wdu[i * 2 + 1] << ")" << std::endl;

        }

        return true;
    
    }
}

namespace TestSpaceTree {

    void test_performance() {
        std::cout << "=== Test SpaceTree Performance ===" << std::endl;

        const int num_knots_dim = 20;
        const int num_knots = num_knots_dim* num_knots_dim* num_knots_dim;
        double knot_limit = 2.;
        const int num_queries = 2000;
        const double space_size = 100.0;
        const double avg_radius = 5.0;


        std::vector<double> thresholds(num_knots, 1.);

        std::vector<double> knots(num_knots * 3);
        for (int x = 0; x < num_knots_dim; ++x) {
            for (int y = 0; y < num_knots_dim; ++y) {
                for (int z = 0; z < num_knots_dim; ++z) {
                    int idx = (x * num_knots_dim * num_knots_dim + y * num_knots_dim + z);
                    knots[idx * 3 + 0] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * x;
                    knots[idx * 3 + 1] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * y;
                    knots[idx * 3 + 2] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * z;

                }
            }
        }
        std::vector<double> queries(num_queries * 3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> coord_dist(-space_size, space_size);
        std::uniform_real_distribution<> radius_dist(1.0, avg_radius * 2);

        // Generate queries
        for(int i=0; i<num_queries; ++i) {
            queries[i*3] = coord_dist(gen);
            queries[i*3+1] = coord_dist(gen);
            queries[i*3+2] = coord_dist(gen);
        }

        // Build Tree
        std::cout << "Building Tree with " << num_knots << " knots..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        cpgeo_handle_t tree_handle = space_tree_create(knots.data(), num_knots, thresholds.data());
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Build time: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

        if (!tree_handle) {
            std::cout << "ERROR: Failed to create SpaceTree" << std::endl;
            return;
        }

        // Print tree stats (we can't easily do this with C API, so skip for now)
        std::cout << "Tree created successfully" << std::endl;

        // 1. Correctness Check (Brute Force vs Tree)
        std::cout << "Verifying correctness..." << std::endl;
        // Check first 100 queries
        int check_count = 100;
        bool all_correct = true;

        // Prepare result arrays for C API
        int total_results;
        int ret = space_tree_query_compute(tree_handle, queries.data(), check_count, &total_results);
        if (ret != 0) {
            std::cout << "ERROR: Space tree query compute failed" << std::endl;
            space_tree_destroy(tree_handle);
            return;
        }

        std::vector<int> indices_cps(total_results);
        std::vector<int> indices_pts(check_count + 1);
        ret = space_tree_query_get(tree_handle, total_results, indices_cps.data(), indices_pts.data());
        if (ret != 0) {
            std::cout << "ERROR: Space tree query get failed" << std::endl;
            space_tree_destroy(tree_handle);
            return;
        }

        // Parse results into per-query vectors
        std::vector<std::vector<int>> tree_results(check_count);
        for (int i = 0; i < total_results; ++i) {
            int query_idx = indices_pts[i];
            int knot_idx = indices_cps[i];
            if (query_idx < 0 || query_idx >= check_count) {
                 std::cout << "CRITICAL ERROR: query_idx out of bounds: " << query_idx << " (max " << check_count << ")" << std::endl;
                 continue;
            }
            tree_results[query_idx].push_back(knot_idx);
        }

        for(int i = 0; i < check_count; ++i) {
            std::vector<int> bf_indices;
            double qx = queries[i*3+0];
            double qy = queries[i*3+1];
            double qz = queries[i*3+2];

            for(int j = 0; j < num_knots; ++j) {
                double kx = knots[j*3+0];
                double ky = knots[j*3+1];
                double kz = knots[j*3+2];
                double r = thresholds[j];
                double dx = qx - kx;
                double dy = qy - ky;
                double dz = qz - kz;
                if (dx*dx + dy*dy + dz*dz <= r*r) {
                    bf_indices.push_back(j);
                }
            }
            
            std::sort(bf_indices.begin(), bf_indices.end());
            
            // Extract tree results for this query
            std::vector<int> tree_indices = tree_results[i];
            std::sort(tree_indices.begin(), tree_indices.end());

            if (bf_indices != tree_indices) {
                std::cout << "Mismatch at query " << i << ". BF: " << bf_indices.size() << ", Tree: " << tree_indices.size() << std::endl;
                
                // 打印详细的点对信息
                std::cout << "  Query point " << i << ": (" << qx << ", " << qy << ", " << qz << ")" << std::endl;
                std::cout << "  BF indices: ";
                for (int idx : bf_indices) std::cout << idx << " ";
                std::cout << std::endl;
                std::cout << "  Tree indices: ";
                for (int idx : tree_indices) std::cout << idx << " ";
                std::cout << std::endl;
                
                // 检查缺失的点对
                std::vector<int> missing_in_tree;
                std::set_difference(bf_indices.begin(), bf_indices.end(), 
                                  tree_indices.begin(), tree_indices.end(), 
                                  std::back_inserter(missing_in_tree));
                if (!missing_in_tree.empty()) {
                    std::cout << "  Missing in tree: ";
                    for (int idx : missing_in_tree) std::cout << idx << " ";
                    std::cout << std::endl;
                }
                
                std::vector<int> extra_in_tree;
                std::set_difference(tree_indices.begin(), tree_indices.end(), 
                                  bf_indices.begin(), bf_indices.end(), 
                                  std::back_inserter(extra_in_tree));
                if (!extra_in_tree.empty()) {
                    std::cout << "  Extra in tree: ";
                    for (int idx : extra_in_tree) std::cout << idx << " ";
                    std::cout << std::endl;
                }
                
                all_correct = false;
                break;
            }
        }

        if (all_correct) {
            std::cout << "Correctness Verified!" << std::endl;
        } else {
            std::cout << "Correctness Check FAILED!" << std::endl;
        }

        // 2. Performance Benchmark
        std::cout << "Running Benchmark (" << num_queries << " queries)..." << std::endl;

        // Tree
        start = std::chrono::high_resolution_clock::now();
        int bench_total_results;
        int bench_ret = space_tree_query_compute(tree_handle, queries.data(), num_queries, &bench_total_results);
        if (bench_ret != 0) {
            std::cout << "ERROR: Bench query compute failed" << std::endl;
            space_tree_destroy(tree_handle);
            return;
        }
        std::vector<int> bench_results(bench_total_results);
        std::vector<int> bench_indices_pts(num_queries + 1);
        bench_ret = space_tree_query_get(tree_handle, bench_total_results, bench_results.data(), bench_indices_pts.data());
        end = std::chrono::high_resolution_clock::now();
        double tree_time = std::chrono::duration<double>(end - start).count();
        std::cout << "Tree Query Time: " << tree_time * 1000 << " ms" << std::endl;

        // Brute Force (single threaded for fair baseline comparison implies basic loop)
        // Note: Real world BF might be vectorized, but O(N*M) is the point.
        start = std::chrono::high_resolution_clock::now();
        volatile int dummy = 0;
        #pragma omp parallel for reduction(+:dummy)
        for(int i=0; i<num_queries; ++i) {
            double qx = queries[i*3+0];
            double qy = queries[i*3+1];
            double qz = queries[i*3+2];
            for(int j=0; j<num_knots; ++j) {
                double kx = knots[j*3+0];
                double ky = knots[j*3+1];
                double kz = knots[j*3+2];
                double r = thresholds[j];
                // simple check
                if ((qx-kx)*(qx-kx) + (qy-ky)*(qy-ky) + (qz-kz)*(qz-kz) <= r*r) {
                    dummy++;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double bf_time = std::chrono::duration<double>(end - start).count();
        std::cout << "Brute Force (OpenMP) Time: " << bf_time * 1000 << " ms" << std::endl;

        std::cout << "Speedup: " << bf_time / tree_time << "x" << std::endl;

        // Clean up
        space_tree_destroy(tree_handle);
    }
}

namespace TestTriangulationPlain {
    // Test case 1: Simple square with 4 points
    bool test_square() {
        std::cout << "=== Test 1: Square (4 points) ===" << std::endl;
        
        std::vector<double> nodes = {
            0.0, 0.0,  // point 0
            1.0, 0.0,  // point 1
            1.0, 1.0,  // point 2
            0.0, 1.0   // point 3
        };
        
        int num_nodes = 4;
        int num_triangles = 0;
        
        cpgeo_handle_t handle = triangulation_compute(nodes.data(), num_nodes, &num_triangles);
        if (!handle) {
            std::cout << "ERROR: Triangulation compute failed" << std::endl;
            return false;
        }
        
        std::vector<int> triangles(num_triangles * 3);
        
        int result = triangulation_get_data(handle, triangles.data());
        if (result != 0) {
            std::cout << "ERROR: Triangulation get data failed with code " << result << std::endl;
            return false;
        }
        
        // std::cout << "Number of triangles: " << num_triangles << std::endl;
        // std::cout << "Expected: 2 triangles" << std::endl;
        
        // for (int i = 0; i < num_triangles; ++i) {
        //     std::cout << "Triangle " << i << ": [" 
        //             << triangles[i*3] << ", " 
        //             << triangles[i*3+1] << ", " 
        //             << triangles[i*3+2] << "]" << std::endl;
        // }
        
        bool passed = (num_triangles == 2);
        std::cout << (passed ? "PASSED" : "FAILED") << std::endl << std::endl;
        
        if (passed) {
            // DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            // triangulation.triangulate();
            // triangulation.exportToObj("test1_square.obj");
        }
        
        return passed;
    }

    // Test case 2: Circle points
    bool test_circle() {
        std::cout << "=== Test 2: Circle (8 points) ===" << std::endl;
        
        std::vector<double> nodes;
        int num_nodes = 8;
        
        // Generate points on a circle
        for (int i = 0; i < num_nodes; ++i) {
            double angle = 2.0 * M_PI * i / num_nodes;
            nodes.push_back(std::cos(angle));
            nodes.push_back(std::sin(angle));
        }
        
        int num_triangles = 0;
        
        cpgeo_handle_t handle = triangulation_compute(nodes.data(), num_nodes, &num_triangles);
        if (!handle) {
            std::cout << "ERROR: Triangulation compute failed" << std::endl;
            return false;
        }
        
        std::vector<int> triangles(num_triangles * 3);
        
        int result = triangulation_get_data(handle, triangles.data());
        if (result != 0) {
            std::cout << "ERROR: Triangulation get data failed with code " << result << std::endl;
            return false;
        }
        
        // std::cout << "Number of triangles: " << num_triangles << std::endl;
        // std::cout << "Expected: approximately " << (num_nodes - 2) << " triangles" << std::endl;
        
        // Verify all vertex indices are valid
        bool valid = true;
        for (int i = 0; i < num_triangles * 3; ++i) {
            if (triangles[i] < 0 || triangles[i] >= num_nodes) {
                std::cout << "ERROR: Invalid vertex index " << triangles[i] << std::endl;
                valid = false;
            }
        }
        
        // Verify no degenerate triangles (all three vertices different)
        for (int i = 0; i < num_triangles; ++i) {
            int v0 = triangles[i*3];
            int v1 = triangles[i*3+1];
            int v2 = triangles[i*3+2];
            
            if (v0 == v1 || v1 == v2 || v0 == v2) {
                std::cout << "ERROR: Degenerate triangle [" << v0 << ", " << v1 << ", " << v2 << "]" << std::endl;
                valid = false;
            }
        }
        
        bool passed = (result == 0 && valid && num_triangles > 0);
        std::cout << (passed ? "PASSED" : "FAILED") << std::endl << std::endl;
        
        if (passed) {
            // DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            // triangulation.triangulate();
            // triangulation.exportToObj("test2_circle.obj");
        }
        
        return passed;
    }

    // Test case 3: Random points
    bool test_random_points() {
        std::cout << "=== Test 3: Random Points (20 points) ===" << std::endl;
        
        std::vector<double> nodes = {
            0.5, 0.5,   1.2, 0.3,   2.1, 0.8,   0.7, 1.5,
            1.8, 1.2,   2.5, 0.5,   0.2, 2.0,   1.5, 2.3,
            2.8, 1.5,   0.9, 2.8,   1.7, 3.0,   2.4, 2.5,
            3.0, 0.8,   3.2, 2.0,   0.3, 0.2,   2.9, 3.1,
            1.1, 0.9,   2.2, 1.8,   0.6, 3.2,   3.3, 1.2
        };
        
        int num_nodes = 20;
        int num_triangles = 0;
        
        cpgeo_handle_t handle = triangulation_compute(nodes.data(), num_nodes, &num_triangles);
        if (!handle) {
            std::cout << "ERROR: Triangulation compute failed" << std::endl;
            return false;
        }
        
        std::vector<int> triangles(num_triangles * 3);
        
        int result = triangulation_get_data(handle, triangles.data());
        if (result != 0) {
            std::cout << "ERROR: Triangulation get data failed with code " << result << std::endl;
            return false;
        }
        
        std::cout << "Number of triangles: " << num_triangles << std::endl;
        
        // Calculate total area of all triangles
        double total_area = 0.0;
        for (int i = 0; i < num_triangles; ++i) {
            int v0 = triangles[i*3];
            int v1 = triangles[i*3+1];
            int v2 = triangles[i*3+2];
            
            double x0 = nodes[v0*2], y0 = nodes[v0*2+1];
            double x1 = nodes[v1*2], y1 = nodes[v1*2+1];
            double x2 = nodes[v2*2], y2 = nodes[v2*2+1];
            
            // Calculate triangle area using cross product
            double area = std::abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)) / 2.0;
            total_area += area;
        }
        
        // std::cout << "Total mesh area: " << total_area << std::endl;
        
        bool passed = (result == 0 && num_triangles > 0 && total_area > 0);
        std::cout << (passed ? "PASSED" : "FAILED") << std::endl << std::endl;
        
        if (passed) {
            // DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            // triangulation.triangulate();
            // triangulation.exportToObj("test3_random.obj");
        }
        
        return passed;
    }

    // Test case 4: Edge case - minimum points
    bool test_minimum_points() {
        std::cout << "=== Test 4: Minimum Points (3 points) ===" << std::endl;
        
        std::vector<double> nodes = {
            0.0, 0.0,
            1.0, 0.0,
            0.5, 1.0
        };
        
        int num_nodes = 3;
        int num_triangles = 0;
        
        cpgeo_handle_t handle = triangulation_compute(nodes.data(), num_nodes, &num_triangles);
        if (!handle) {
            std::cout << "ERROR: Triangulation compute failed" << std::endl;
            return false;
        }
        
        std::vector<int> triangles(num_triangles * 3);
        
        int result = triangulation_get_data(handle, triangles.data());
        if (result != 0) {
            std::cout << "ERROR: Triangulation get data failed with code " << result << std::endl;
            return false;
        }
        
        std::cout << "Number of triangles: " << num_triangles << std::endl;
        // std::cout << "Expected: 2 triangles" << std::endl;
        
        // if (num_triangles == 1) {
        //     std::cout << "Triangle: [" 
        //             << triangles[0] << ", " 
        //             << triangles[1] << ", " 
        //             << triangles[2] << "]" << std::endl;
        // }
        
        bool passed = (num_triangles == 1);
        std::cout << (passed ? "PASSED" : "FAILED") << std::endl << std::endl;
        
        if (passed) {
            // DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            // triangulation.triangulate();
            // triangulation.exportToObj("test4_minimum.obj");
        }
        
        return passed;
    } 
    // 检查三角形法向量是否朝外（对于球面，法向量应该与顶点位置向量同向）
    bool checkNormalOrientation(const std::vector<std::array<int, 3>>& triangles, 
                                 const double* vertices, int num_vertices) {
        int correct_normals = 0;
        int incorrect_normals = 0;
        
        for (size_t i = 0; i < triangles.size(); ++i) {
            const auto& tri = triangles[i];
            int v0_idx = tri[0];
            int v1_idx = tri[1];
            int v2_idx = tri[2];
            
            // 获取三个顶点坐标
            double v0[3] = {vertices[v0_idx * 3], vertices[v0_idx * 3 + 1], vertices[v0_idx * 3 + 2]};
            double v1[3] = {vertices[v1_idx * 3], vertices[v1_idx * 3 + 1], vertices[v1_idx * 3 + 2]};
            double v2[3] = {vertices[v2_idx * 3], vertices[v2_idx * 3 + 1], vertices[v2_idx * 3 + 2]};
            
            // 计算两条边向量
            double edge1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
            double edge2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
            
            // 计算叉积得到法向量 (edge1 × edge2)
            double normal[3] = {
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            };
            
            // 计算三角形中心点
            double center[3] = {
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0
            };
            
            // 对于球面，中心点就是从球心指向外的参考方向
            // 检查法向量与中心点的点积
            double dot = normal[0] * center[0] + normal[1] * center[1] + normal[2] * center[2];
            
            if (dot > 0) {
                correct_normals++;
            } else {
                incorrect_normals++;
                if (incorrect_normals <= 10) {  // 只打印前10个
                    std::cout << "  Triangle " << i << " [" << v0_idx << ", " << v1_idx << ", " << v2_idx 
                              << "] has INWARD normal (dot=" << dot << ")" << std::endl;
                }
            }
        }
        
        std::cout << "Normal orientation check:" << std::endl;
        std::cout << "  Correct (outward) normals: " << correct_normals << std::endl;
        std::cout << "  Incorrect (inward) normals: " << incorrect_normals << std::endl;
        
        bool all_correct = (incorrect_normals == 0);
        if (all_correct) {
            std::cout << "  ✓ All normals point OUTWARD (counter-clockwise from outside)" << std::endl;
        } else {
            std::cout << "  ✗ Some normals point inward - inconsistent winding order" << std::endl;
        }
        
        return all_correct;
    }
    
    // 检查网格的边拓扑：每条边应该恰好出现2次（闭合流形）
    bool checkEdgeTopology(const std::vector<std::array<int, 3>>& triangles, int num_vertices) {
        // 使用 map 存储每条边的出现次数，边用有序对表示
        std::unordered_map<uint64_t, int> edge_count;
        
        auto make_edge_key = [](int v1, int v2) -> uint64_t {
            // 确保 v1 < v2，这样 (v1, v2) 和 (v2, v1) 是同一条边
            if (v1 > v2) std::swap(v1, v2);
            return (static_cast<uint64_t>(v1) << 32) | static_cast<uint64_t>(v2);
        };
        
        // 统计每条边出现的次数
        for (const auto& tri : triangles) {
            edge_count[make_edge_key(tri[0], tri[1])]++;
            edge_count[make_edge_key(tri[1], tri[2])]++;
            edge_count[make_edge_key(tri[2], tri[0])]++;
        }
        
        // 检查每条边是否恰好出现2次
        int bad_edges = 0;
        int boundary_edges = 0;  // 出现1次的边（边界边）
        int interior_edges = 0;  // 出现2次的边（内部边）
        int invalid_edges = 0;   // 出现>2次的边（非流形）
        
        for (const auto& [edge, count] : edge_count) {
            if (count == 1) {
                boundary_edges++;
                int v1 = edge >> 32;
                int v2 = edge & 0xFFFFFFFF;
                if (bad_edges < 10) {  // 只打印前10条
                    std::cout << "  Boundary edge: " << v1 << " - " << v2 << std::endl;
                }
                bad_edges++;
            } else if (count == 2) {
                interior_edges++;
            } else {
                invalid_edges++;
                int v1 = edge >> 32;
                int v2 = edge & 0xFFFFFFFF;
                std::cout << "  INVALID edge (count=" << count << "): " << v1 << " - " << v2 << std::endl;
                bad_edges++;
            }
        }
        
        std::cout << "Edge topology check:" << std::endl;
        std::cout << "  Total edges: " << edge_count.size() << std::endl;
        std::cout << "  Interior edges (count=2): " << interior_edges << std::endl;
        std::cout << "  Boundary edges (count=1): " << boundary_edges << std::endl;
        std::cout << "  Invalid edges (count>2): " << invalid_edges << std::endl;
        
        bool is_closed_manifold = (boundary_edges == 0 && invalid_edges == 0);
        if (is_closed_manifold) {
            std::cout << "  ✓ Mesh is a CLOSED MANIFOLD (no boundary, no non-manifold edges)" << std::endl;
        } else {
            std::cout << "  ✗ Mesh has topological issues" << std::endl;
        }
        
        return is_closed_manifold;
    }
    
    bool test_sphere_simple() {
        
        
        const int num_points = 5000;

        std::cout << "=== Test 5: Sphere Triangulation (" << num_points << " points) ===" << std::endl;
        auto sphere_points = std::make_unique<double[]>(num_points * 3);
        const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
        const double golden_angle = 2.0 * M_PI / phi;

        // Original code commented out
        /*
        for (int i = 0; i < num_points; ++i) {
            double y = 1.0 - (2.0 * i) / (num_points - 1.0);
            double r = std::sqrt(1.0 - y * y);
            double theta = i * golden_angle;
            double x = r * std::cos(theta);
            double z = r * std::sin(theta);
            sphere_points[i * 3] = x;
            sphere_points[i * 3 + 1] = y;
            sphere_points[i * 3 + 2] = z;
        }
        */
        
        // Generate random points on the sphere
        srand(time(NULL));
        for (int i = 0; i < num_points; ++i) {
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            double theta = 2.0 * M_PI * u1;
            double phi = acos(2.0 * u2 - 1.0);
            double x = sin(phi) * cos(theta);
            double y = sin(phi) * sin(theta);
            double z = cos(phi);
            sphere_points[i * 3] = x;
            sphere_points[i * 3 + 1] = y;
            sphere_points[i * 3 + 2] = z;
        }
        // std::cout << "Number of sphere points: " << num_points << std::endl;
        
        int num_triangles = 0;
        cpgeo_handle_t handle = sphere_triangulation_compute(sphere_points.get(), num_points, &num_triangles);
        if (!handle) {
            std::cout << "ERROR: Sphere triangulation compute failed" << std::endl;
            return false;
        }
        
        std::vector<int> triangle_indices(num_triangles * 3);
        int result = sphere_triangulation_get_data(handle, triangle_indices.data());
        if (result != 0) {
            std::cout << "ERROR: Sphere triangulation get data failed with code " << result << std::endl;
            return false;
        }
        
        // std::cout << "Number of triangles: " << num_triangles << std::endl;
        
        // 转换为 vector<array<int,3>> 格式
        std::vector<std::array<int, 3>> triangles_vec;
        triangles_vec.reserve(num_triangles);
        for (int i = 0; i < num_triangles; ++i) {
            triangles_vec.push_back({
                triangle_indices[i * 3],
                triangle_indices[i * 3 + 1],
                triangle_indices[i * 3 + 2]
            });
        }
        
        // 检查法向量方向
        bool normals_ok = checkNormalOrientation(triangles_vec, sphere_points.get(), num_points);
        
        // 检查边拓扑
        bool topology_ok = checkEdgeTopology(triangles_vec, num_points);
        
        bool passed = (num_triangles > 0) && topology_ok && normals_ok;
        std::cout << (passed ? "PASSED" : "FAILED") << std::endl << std::endl;
        
        if (num_triangles > 0) {
            // sphere_tri.exportToObj("test5_sphere.obj");
        }
        
        return passed;
    }
    void test_mesh(){
    std::cout << "========================================" << std::endl;
    std::cout << "CPGEO Delaunay Triangulation Test Suite" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    int passed = 0;
    int total = 0;

    // Run all tests
    auto t0 = clock();
    if (test_minimum_points()) passed++;
    total++;
    auto t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    t0 = clock();
    if (test_square()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;
    
    t0 = clock();
    if (test_circle()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    t0 = clock();
    if (test_random_points()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    t0 = clock();
    if (test_sphere_simple()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    

    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "========================================" << std::endl;
}
}

namespace TestEdgeRefinement{

    void test_refinement() {
        std::cout << "=== Test Edge Refinement ===" << std::endl;
        std::vector<double> vertices = {0.5,0.0,1.5,0.0,1.5,1.5,0.5,1.5};
        std::vector<int> edges = {0,1,1,2,2,3,3,0, 0,2};
        int num_indices;
        mesh_closure_edge_length_derivative2_compute(vertices.data(), 4, 2, edges.data(), 5, 2, &num_indices);

        double loss = 0.0;
        std::vector<double> Ldr(vertices.size(), 0.0);
        std::vector<int> Ldr2_indices(num_indices * 4);
        std::vector<double> Ldr2_values(num_indices);

        int result = mesh_closure_edge_length_derivative2_get(&loss, Ldr.data(), Ldr2_indices.data(), Ldr2_values.data());

        // Output ldr2 in COO format
        std::cout << "ldr2 in COO format:" << std::endl;
        for (size_t i = 0; i < Ldr2_values.size(); ++i) {
            std::cout << Ldr2_indices[i * 4] << " " << Ldr2_indices[i * 4 + 1] << " " << Ldr2_indices[i * 4 + 2] << " " << Ldr2_indices[i * 4 + 3] << " " << Ldr2_values[i] << std::endl;
        }
    }
}

namespace TestRefineMesh{

    static bool read_points_from_file(const std::string &path, std::vector<double> &out_points) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) return false;
        out_points.clear();
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            double x, y, z;
            char comma;
            // Expect format: x,y,z
            if (!(ss >> x)) continue;
            if (!(ss >> comma)) continue;
            if (!(ss >> y)) continue;
            if (!(ss >> comma)) continue;
            if (!(ss >> z)) continue;
            out_points.push_back(x);
            out_points.push_back(y);
            out_points.push_back(z);
        }
        return !out_points.empty();
    }

    bool test_remesh_build_spacetree() {
        std::cout << "=== Test Remesh: Load knots & controlpoints and build SpaceTree ===" << std::endl;



        std::vector<double> knots;
        std::vector<double> controlpoints;

        knots.reserve(10000);
        controlpoints.reserve(10000);

        // Prefer explicit filenames
        bool ok1 = read_points_from_file("A:/MineData/Learning/Code/Projects/Modules/CPGEO/src/cpp/tests/knots.txt", knots);
        bool ok2 = read_points_from_file("A:/MineData/Learning/Code/Projects/Modules/CPGEO/src/cpp/tests/control_points.txt", controlpoints);

        int num_knots = static_cast<int>(knots.size() / 3);
        std::cout << "Loaded " << num_knots << " knots" << std::endl;
        if (num_knots <= 0) return false;

        // Compute thresholds using k-nearest (k=4)
        std::vector<double> thresholds(num_knots, 1.0);
        int k = 20;
        cpgeo_compute_thresholds(knots.data(), num_knots, k, thresholds.data());

        // Build space tree
        cpgeo_handle_t tree = space_tree_create(knots.data(), num_knots, thresholds.data());
        if (!tree) {
            std::cout << "space_tree_create failed" << std::endl;
            return false;
        }

        std::cout << "SpaceTree created successfully (" << num_knots << " knots)" << std::endl;


		cpgeo::uniformlyMesh(knots, controlpoints, *(cpgeo::SpaceTree*)tree, 1.0, 10);

        // cleanup
        space_tree_destroy(tree);
        return true;
    }

    bool simpletest(){
        int num_knots_dim = 10;
        double knot_limit = 2.0;

        std::vector<double> knots(num_knots_dim * num_knots_dim * num_knots_dim * 3, 0.0);
        std::vector<double> thresholds(num_knots_dim * num_knots_dim * num_knots_dim, 0.6);

        for (int x = 0; x < num_knots_dim; ++x) {
            for (int y = 0; y < num_knots_dim; ++y) {
                for (int z = 0; z < num_knots_dim; ++z) {
                    int idx = (x * num_knots_dim * num_knots_dim + y * num_knots_dim + z);
                    knots[idx * 3 + 0] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * x + 0.01;
                    knots[idx * 3 + 1] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * y;
                    knots[idx * 3 + 2] = -knot_limit + (2.0 * knot_limit / (num_knots_dim - 1)) * z;
                }
            }
        }

        auto tree = space_tree_create(
            knots.data(),
            static_cast<int>(knots.size() / 3),
            thresholds.data()
        );

        std::vector<double> nodes(3*3);

        auto point_sphere = cpgeo::stereographicProjection2_3(std::array<double, 2>({-2., 0.}));
		nodes[0] = point_sphere[0];
		nodes[1] = point_sphere[1];
		nodes[2] = point_sphere[2];
		point_sphere = cpgeo::stereographicProjection2_3(std::array<double, 2>({ 2., 0. }));
        nodes[3] = point_sphere[0];
        nodes[4] = point_sphere[1];
        nodes[5] = point_sphere[2];
        point_sphere = cpgeo::stereographicProjection2_3(std::array<double, 2>({ 0., 1. }));
        nodes[6] = point_sphere[0];
        nodes[7] = point_sphere[1];
        nodes[8] = point_sphere[2];
        

        std::vector<int> faces = {0, 1, 2};

        cpgeo::vertice_smoothing(nodes, faces, knots, *(cpgeo::SpaceTree*)tree);

        return true;
    }
}

int main() {

    // TestSpaceTree::test_performance();
    // TestTriangulationPlain::test_mesh();
     //TestWeight::test_weight_function();   
     //TestEdgeRefinement::test_refinement(); 
    TestRefineMesh::test_remesh_build_spacetree();


    return 0;
}