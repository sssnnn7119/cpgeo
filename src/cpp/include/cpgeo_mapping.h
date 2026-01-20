#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <memory>
#include <span>
#include "tensor.h"






namespace cpgeo {

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
        const std::span<const double, 2> query_points);

    /** compute weights and first derivatives for query points
        @param indices flat array of size (numIndices * 2) representing (query_idx, knot_idx) pairs
        @param knots array of size (numCps * 3) representing 3D control points
        @param thresholds array of size (numCps) representing influence radii for each control point
        @param query_points array of size (2) representing query points in 2D plane coordinates
        @return tuple of (weights vector of size numIndices, Tensor2D of size (numIndices, 2) for first derivatives)
    */
    std::tuple<std::vector<double>, Tensor2D> get_weights_derivative1(
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
    std::tuple<std::vector<double>, Tensor2D, Tensor3D> get_weights_derivative2(
        const std::span<const int> indices,
        const std::span<const double> knots,
        const std::span<const double> thresholds,
        const std::span<const double, 2> query_points_plane);

}