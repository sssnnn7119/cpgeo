#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <memory>
#include <span>
#include "tensor.h"
#include "space_tree.h"





namespace cpgeo {

    inline std::array<double, 2> stereographicProjection3_2(const std::span<const double, 3> cooSphere);
    inline std::array<double, 3> stereographicProjection2_3(const std::span<const double, 2> cooPlane);
    inline std::tuple<std::array<double, 3>, Tensor2D> stereographicProjection2_3Derivative1(const std::span<const double> cooPlane);
    inline std::tuple<std::array<double, 3>, Tensor2D, Tensor3D> stereographicProjection2_3Derivative2(const std::span<const double> cooPlane);


    /** calculate the k-th nearest neighbor distances for each point in knots
        @param knots array of size (numPoints * 3) representing 3D points
        @param k which nearest neighbor to consider
        @return vector of size numPoints with the k-th nearest neighbor distances
    */
    std::vector<double> compute_thresholds(const std::span<const double> knots, int k);

    /** compute weights for query points based on influence of knots within thresholds
        @param indices flat array of size (numIndices * 2) representing (query_idx, knot_idx) pairs
        @param knots array of size (numCps * 3) representing 3D control points
        @param thresholds array of size (numCps) representing influence radii for each control point
        @param query_points array of size (2) representing query points in 2D plane coordinates
        @return vector of size numQueries with computed weights
    */
    std::vector<double> get_weights(
        const std::span<const int> indices,
        const std::span<const double> knots, 
        const std::span<const double> thresholds, 
        const std::span<const double, 2> query_points_plane);

    std::vector<double> get_weights(
        const std::span<const int> indices,
        const std::span<const double> knots,
        const std::span<const double> thresholds,
        const std::span<const double, 3> query_points);

    /** compute weights and first derivatives for query points
        @param indices flat array of size (numIndices * 2) representing (query_idx, knot_idx) pairs
        @param knots array of size (numCps * 3) representing 3D control points
        @param thresholds array of size (numCps) representing influence radii for each control point
        @param query_points array of size (2) representing query points in 2D plane coordinates
        @return tuple of (weights vector of size numIndices, Tensor2D of size (numIndices, 2) for first derivatives)
    */
    std::array<std::vector<double>, 2> get_weights_derivative1(
        const std::span<const int> indices,
        const std::span<const double> knots,
        const std::span<const double> thresholds,
        const std::span<const double, 2> query_points_plane);

    /** compute weights and first and second derivatives for query points
        @param indices flat array of size (numIndices * 2) representing (query_idx, knot_idx) pairs
        @param knots array of size (numCps * 3) representing 3D control points
        @param thresholds array of size (numCps) representing influence radii for each control point
        @param query_points array of size (2) representing query points in 2D plane coordinates
        @return tuple of (weights vector of size numIndices, Tensor2D of size (numIndices, 2) for first derivatives, Tensor3D of size (numIndices, 2, 2) for second derivatives)
    */
    std::array<std::vector<double>, 3> get_weights_derivative2(
        const std::span<const int> indices,
        const std::span<const double> knots,
        const std::span<const double> thresholds,
        const std::span<const double, 2> query_points_plane);


    /** compute the single mapped point from control points using sparse weights
        @param indices_cps Array of control point indices for each weight (size: num_indices)
        @param weights Array of weight values (size: num_indices)
        @param controlpoints Flat array of 3D control point coordinates [x0,y0,z0,...]
        @return array of mapped points (size: (3)
    */
    std::array<double, 3> get_mapped_points(
        const std::span<const int> indices_cps,
        const std::span<const double> weights,
        const std::span<const double> controlpoints);

    /** compute mapped points from control points using sparse weights
        @param indices_cps Array of control point indices for each weight (size: num_indices)
        @param indices_pts Array of starting indices for each query point (size: num_queries + 1)
        @param weights Array of weight values (size: num_indices)
        @param controlpoints Flat array of 3D control point coordinates [x0,y0,z0,...]
        @return vector of mapped points (size: (indices_pts.max() - 1) * 3)
    */
    std::vector<double> get_mapped_points_batch(
        const std::span<const int> indices_cps,
        const std::span<const int> indices_pts,
        const std::span<const double> weights,
        const std::span<const double> controlpoints);

    /** perform stereographic projection from 3D sphere to 2D plane
        @param query_point array of size (3) representing 3D coordinates on the sphere
        @param tree SpaceTree instance for querying knot influences
        @param controlpoints Flat array of 3D control point coordinates [x0,y0
        @return array of size (3) representing mapped 3D coordinates
    */
    std::array<double, 3> map_points(
        std::span<const double, 3> query_point,
        SpaceTree& tree,
        const std::span<const double> controlpoints
    );

    /** perform stereographic projection from 3D sphere to 2D plane for batch of points
        @param query_point array of size (num_queries * 3) representing 3D coordinates on the sphere
        @param tree SpaceTree instance for querying knot influences
        @param controlpoints Flat array of 3D control point coordinates [x0,y0
        @return vector of size (num_queries * 3) representing mapped 3D coordinates
    */
    std::vector<double> map_points_batch(
        std::span<const double> query_point,
        SpaceTree& tree,
        const std::span<const double> controlpoints
    );
}