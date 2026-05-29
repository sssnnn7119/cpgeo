#pragma once

#include <vector>
#include <memory>
#include <array>
#include <span>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

namespace cpgeo {

/**
 * @brief Result of enforce_rotational_symmetry_z
 */
struct RotationalSymmetryResult {
    std::vector<double> vertices;
    std::vector<int64_t> faces;
    std::vector<int64_t> match;  // shape (periods, base_vertex_count), row-major
};

/**
 * @brief Enforce Cn rotational symmetry around the z-axis by strict open-sector clipping + replication.
 *
 * @param vertices Flat vertex coordinates [x0,y0,z0, x1,y1,z1, ...] (shape N*3)
 * @param faces Flat face indices [t0v0,t0v1,t0v2, t1v0,...] (shape F*3)
 * @param periods Number of periods in Cn symmetry (>= 2)
 * @param threshold Threshold for determining the sector boundary (negative = auto)
 * @param tol Numerical tolerance
 * @param return_match Whether to also compute and return the rotational correspondence
 * @return RotationalSymmetryResult with new vertices, faces, and optionally match
 */
RotationalSymmetryResult enforce_rotational_symmetry_z(
    std::span<const double> vertices,
    std::span<const int64_t> faces,
    int periods,
    double threshold = -1.0,
    double tol = 1e-8,
    bool return_match = false);

// ---- Internal helpers ----

/**
 * Rotate 2D points around Z axis by angle.
 * points: (N, 3)
 */
std::vector<double> rot_z(std::span<const double> points, double angle);

/**
 * Zipper-stitch two seam chains into triangle strips using DFS branch-and-bound.
 * Returns triangles (M, 3) as flat int64 vector.
 */
std::vector<int64_t> zipper_stitch(
    std::span<const int64_t> right_ids,
    std::span<const double> right_pts,
    std::span<const int64_t> left_ids,
    std::span<const double> left_pts,
    double dihedral_angle_threshold = 60.0);

/**
 * Decide how many vertices to trim from pole ends of seam chains.
 */
int decide_pole_trim_count(
    std::span<const int64_t> left_side,
    std::span<const int64_t> right_side,
    std::span<const double> sector_vertices,
    double mean_edge);

/**
 * Order seam chains south->north for monotonic stitching.
 */
void order_seam_chains(
    std::span<const int64_t> right_ids_in,
    std::span<const double> right_pts_in,
    std::span<const int64_t> left_ids_in,
    std::span<const double> left_pts_in,
    std::vector<int64_t>& out_rid, std::vector<double>& out_rpt,
    std::vector<int64_t>& out_lid, std::vector<double>& out_lpt);

/**
 * Find connected components of faces by edge adjacency.
 */
std::vector<std::vector<int64_t>> face_components_by_edges(std::span<const int64_t> faces_flat, int64_t num_faces);

/**
 * Sum of triangle areas.
 */
double tri_area_sum(std::span<const double> vertices, std::span<const int64_t> faces_flat, int64_t num_faces);

/**
 * Mean edge length of mesh.
 */
double mean_edge_length(std::span<const double> vertices, std::span<const int64_t> faces_flat, int64_t num_faces);

/**
 * Check if z-axis intersects a triangle, and find intersection point.
 */
std::pair<bool, std::array<double, 3>> axis_triangle_intersection(
    std::span<const double> vertices, std::span<const int64_t> tri, double eps = 1e-12);

/**
 * Find south and north pole triangles (those intersected by z-axis).
 */
struct PoleTriangles {
    std::vector<int64_t> south_tri;
    std::array<double, 3> south_hit;
    std::vector<int64_t> north_tri;
    std::array<double, 3> north_hit;
    bool has_south = false;
    bool has_north = false;
};
PoleTriangles find_axis_pole_triangles(std::span<const double> vertices, std::span<const int64_t> faces_flat, int64_t num_faces);

/**
 * Create a virtual anchor point near a pole triangle.
 */
std::array<double, 3> virtual_anchor_from_triangle(
    std::span<const double> tri_points,
    const std::array<double, 3>& axis_hit,
    double phase, double alpha, double target_r);

/**
 * Recover faces removed from interior holes by expanding from boundary vertices.
 */
std::vector<int64_t> recover_removed_faces_for_interior_holes(
    std::span<const int64_t> f_sector_flat, int64_t n_sector,
    std::span<const int64_t> f_comp_flat, int64_t n_comp,
    const std::unordered_set<int64_t>& interior_boundary_vertices);

/**
 * Seal boundary loops with ear-clipping triangulation.
 */
std::vector<int64_t> seal_loops_constrained(
    std::span<const int64_t> faces_flat, int64_t num_faces,
    std::span<const int64_t> loop_vertices,
    std::span<const int64_t> loop_starts,
    int64_t num_loops,
    int64_t max_loop_size = 256);

/**
 * Split a boundary loop into two seam paths (south->north).
 */
std::pair<std::vector<int64_t>, std::vector<int64_t>> split_seam_paths(
    std::span<const int64_t> loop,
    std::span<const double> vertices,
    int64_t preferred_south_idx = -1,
    int64_t preferred_north_idx = -1,
    const double* preferred_south_point = nullptr,
    const double* preferred_north_point = nullptr);

/**
 * Perimeter of a boundary loop.
 */
double loop_perimeter(std::span<const double> vertices, std::span<const int64_t> loop);

/**
 * Rotate a cyclic array so that start_idx becomes the first element.
 */
std::vector<int64_t> rotate_cycle(std::span<const int64_t> ids, int64_t start_idx);

/**
 * Zipper-stitch two CLOSED loops (cyclic).
 */
std::vector<int64_t> zipper_stitch_closed(
    std::span<const int64_t> loop_a,
    std::span<const int64_t> loop_b,
    std::span<const double> vertices);

/**
 * Sample `periods` evenly-spaced indices from a loop.
 */
std::vector<int64_t> sample_loop_ids(std::span<const int64_t> loop, int periods);

/**
 * Build a polar candidate ring, optionally pulling far points toward the axis.
 * Returns (candidate_points_flat, pulled).
 */
std::pair<std::vector<double>, bool> build_polar_candidate_ring(
    std::span<const double> vertices,
    std::span<const int64_t> loop,
    int periods,
    double mean_edge,
    double far_factor = 2.0);

/**
 * Insert a quality polar ring and bridge/cap it.
 * Returns (new_vertices_flat, new_faces_flat).
 */
std::pair<std::vector<double>, std::vector<int64_t>> insert_polar_quality_ring(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    std::span<const int64_t> loop,
    int periods,
    double mean_edge,
    bool upward);

/**
 * Mesh polar holes (both south and north).
 * Returns (new_vertices_flat, new_faces_flat).
 */
std::pair<std::vector<double>, std::vector<int64_t>> mesh_polar_holes(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    int periods,
    double mean_edge);

/**
 * Extract a clean sector from the mesh. This is the core sector-finding logic.
 * Returns (sector_vertices_flat, sector_faces_flat, left_side, right_side).
 * left_side and right_side are vertex indices in the local sector.
 */
struct SectorResult {
    std::vector<double> vertices;
    std::vector<int64_t> faces;
    std::vector<int64_t> left_side;
    std::vector<int64_t> right_side;
    bool valid = false;
};
SectorResult extract_sector(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    double alpha, double threshold, double tol);

/**
 * Wrapper: call extractBoundaryLoops from mesh_utils and return as flat + starts arrays.
 */
std::pair<std::vector<int64_t>, std::vector<int64_t>> extract_boundary_loops_flat(std::span<const int64_t> faces_flat, int64_t num_faces);

/**
 * Wrapper: call extractEdgesWithNumber and return as flat (v0,v1,count)*3.
 */
std::vector<int64_t> get_mesh_edges_flat(std::span<const int64_t> faces_flat, int64_t num_faces);

/**
 * Wrapper: call mesh_optimize_by_edge_flipping.
 */
std::vector<int64_t> optimize_mesh_by_edge_flipping_wrapper(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    int max_iterations = 100);

} // namespace cpgeo
