#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <memory>
#include <span>








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
        @param query_points array of size (numQueries * 3) representing 3D query points
        @return vector of size numQueries with computed weights
    */
    std::vector<double> get_weights(
        const std::span<const double> indices,
        const std::span<const double> knots, 
        const std::span<const double> thresholds, 
        const std::span<const double> query_points);
}