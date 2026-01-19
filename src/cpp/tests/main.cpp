#include <iostream>
#include <vector>
#include <cmath>
#include "cpgeo.h"
#include "../include/triangular_mesh.h"
#include <math.h>
const double M_PI = 3.14159265358979323846;


using namespace cpgeo;


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
    
    std::cout << "Number of triangles: " << num_triangles << std::endl;
    std::cout << "Expected: 2 triangles" << std::endl;
    
    for (int i = 0; i < num_triangles; ++i) {
        std::cout << "Triangle " << i << ": [" 
                  << triangles[i*3] << ", " 
                  << triangles[i*3+1] << ", " 
                  << triangles[i*3+2] << "]" << std::endl;
    }
    
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
    
    std::cout << "Number of triangles: " << num_triangles << std::endl;
    std::cout << "Expected: approximately " << (num_nodes - 2) << " triangles" << std::endl;
    
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
    
    std::cout << "Total mesh area: " << total_area << std::endl;
    
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
    std::cout << "Expected: 1 triangle" << std::endl;
    
    if (num_triangles == 1) {
        std::cout << "Triangle: [" 
                  << triangles[0] << ", " 
                  << triangles[1] << ", " 
                  << triangles[2] << "]" << std::endl;
    }
    
    bool passed = (num_triangles == 1);
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl << std::endl;
    
    if (passed) {
        DelaunayTriangulation triangulation(std::span<const double>(nodes.data(), num_nodes * 2));
        triangulation.triangulate();
        triangulation.exportToObj("test4_minimum.obj");
    }
    
    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CPGEO Delaunay Triangulation Test Suite" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    int passed = 0;
    int total = 0;
    
    // Run all tests
    if (test_minimum_points()) passed++;
    total++;
    
    if (test_square()) passed++;
    total++;
    
    if (test_circle()) passed++;
    total++;
    
    if (test_random_points()) passed++;
    total++;
    
    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (passed == total) ? 0 : 1;
}