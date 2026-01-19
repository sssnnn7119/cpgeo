#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <array>
#include "cpgeo.h"
#include "triangulation.h"
#include "sphere_triangulation.h"
#include <math.h>
#include <time.h>

const double M_PI = 3.14159265358979323846;


using namespace cpgeo;

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
        std::vector<int> triangles(num_nodes * 2 * 3);  // Allocate enough space
        int num_triangles = 0;
        
        int result = cpgeo_triangulate(nodes.data(), num_nodes, triangles.data(), &num_triangles);
        
        if (result != 0) {
            std::cout << "ERROR: Triangulation failed with code " << result << std::endl;
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
            DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            triangulation.triangulate();
            triangulation.exportToObj("test1_square.obj");
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
        
        std::vector<int> triangles(num_nodes * 3 * 3);
        int num_triangles = 0;
        
        int result = cpgeo_triangulate(nodes.data(), num_nodes, triangles.data(), &num_triangles);
        
        if (result != 0) {
            std::cout << "ERROR: Triangulation failed with code " << result << std::endl;
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
            DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            triangulation.triangulate();
            triangulation.exportToObj("test2_circle.obj");
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
        std::vector<int> triangles(num_nodes * 3 * 3);
        int num_triangles = 0;
        
        int result = cpgeo_triangulate(nodes.data(), num_nodes, triangles.data(), &num_triangles);
        
        if (result != 0) {
            std::cout << "ERROR: Triangulation failed with code " << result << std::endl;
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
            DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            triangulation.triangulate();
            triangulation.exportToObj("test3_random.obj");
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
        std::vector<int> triangles(10 * 3);
        int num_triangles = 0;
        
        int result = cpgeo_triangulate(nodes.data(), num_nodes, triangles.data(), &num_triangles);
        
        if (result != 0) {
            std::cout << "ERROR: Triangulation failed with code " << result << std::endl;
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
            DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
            triangulation.triangulate();
            triangulation.exportToObj("test4_minimum.obj");
        }
        
        return passed;
    }

}


namespace TestTriangulationSphere {
    
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
        
        SphereTriangulation sphere_tri(std::span<const double>(sphere_points.get(), num_points * 3));
        sphere_tri.triangulate();
        
        size_t num_triangles = sphere_tri.size();
        // std::cout << "Number of triangles: " << num_triangles << std::endl;
        
        // 获取三角形数据
        std::vector<int> triangle_indices(num_triangles * 3);
        sphere_tri.getTriangleIndices(triangle_indices.data());
        
        // 转换为 vector<array<int,3>> 格式
        std::vector<std::array<int, 3>> triangles_vec;
        triangles_vec.reserve(num_triangles);
        for (size_t i = 0; i < num_triangles; ++i) {
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
            sphere_tri.exportToObj("test5_sphere.obj");
        }
        
        return passed;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CPGEO Delaunay Triangulation Test Suite" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    int passed = 0;
    int total = 0;

    // Run all tests
    auto t0 = clock();
    if (TestTriangulationPlain::test_minimum_points()) passed++;
    total++;
    auto t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    t0 = clock();
    if (TestTriangulationPlain::test_square()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;
    
    t0 = clock();
    if (TestTriangulationPlain::test_circle()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    t0 = clock();
    if (TestTriangulationPlain::test_random_points()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    t0 = clock();
    if (TestTriangulationSphere::test_sphere_simple()) passed++;
    total++;
    t1 = clock();
    std::cout << "Test "<< total << " time: " << double(t1 - t0) / CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;
    
    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (passed == total) ? 0 : 1;
}