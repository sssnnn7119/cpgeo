
#include "cpgeo_mapping.h"
#include "space_tree.h"
#include "cpgeo.h"

namespace cpgeo {

std::vector<double> compute_thresholds(const std::span<const double> knots, int k) {
    int numCps = knots.size() / 3;
    
    auto results = std::vector<double>(numCps);

#pragma omp parallel for
    for (int i = 0; i < numCps; i++) {
        std::vector<double> distances;
        distances.reserve(numCps - 1);

        double x1 = knots[i * 3], y1 = knots[i * 3 + 1], z1 = knots[i * 3 + 2];

        for (int j = 0; j < numCps; j++) {
            if (i != j) {
                double x2 = knots[j * 3], y2 = knots[j * 3 + 1], z2 = knots[j * 3 + 2];
                double dist_squared = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
                distances.push_back(dist_squared);
            }
        }

        // only need the k-th smallest element
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

        // get the distance of the k-th nearest neighbor
		results[i] = std::sqrt(distances[k - 1]);
    }

    return results;  // std::unique_ptr will be moved automatically
}


double weightFunction(const double dist, const double threshold) {
    if (dist >= threshold) {
        return 0.0;
    }
    double ratio = dist / threshold;
    
    double weight = 20 * pow(ratio, 7) - 70 * pow(ratio, 6) + 84 * pow(ratio, 5) - 35 * pow(ratio,4) + 1;

    return weight;
}

double weightFunctionDerivative(const double dist, const double threshold) {
    if (dist >= threshold) {
        return 0.0;
    }
    double ratio = dist / threshold;
    
    double derivative = (140 * pow(ratio, 6) - 420 * pow(ratio, 5) + 420 * pow(ratio, 4) - 140 * pow(ratio,3)) / threshold;

    return derivative;
}

double weightFunctionSecondDerivative(const double dist, const double threshold) {
    if (dist >= threshold) {
        return 0.0;
    }
    double ratio = dist / threshold;
    
    double second_derivative = (840 * pow(ratio, 5) - 2100 * pow(ratio, 4) + 1680 * pow(ratio, 3) - 420 * pow(ratio,2)) / (threshold * threshold);

    return second_derivative;
}


std::vector<double> get_weights(
    const std::span<const double> indices,
    const std::span<const double> knots, 
    const std::span<const double> thresholds, 
    const std::span<const double> query_points) {

    const int numCps = knots.size() / 3;
    const int numQueries = query_points.size() / 3;
    const int numIndices = indices.size() / 2;

    std::vector<double> weights(numQueries, 0.0);
    std::vector<double> weight_sums(numQueries, 0.0);

    // compute the initial weights
    for(int idx = 0; idx < numIndices; ++idx) {
        int query_idx = static_cast<int>(indices[idx * 2]);
        int knot_idx = static_cast<int>(indices[idx * 2 + 1]);

        double qx = query_points[query_idx * 3];
        double qy = query_points[query_idx * 3 + 1];
        double qz = query_points[query_idx * 3 + 2];

        double kx = knots[knot_idx * 3];
        double ky = knots[knot_idx * 3 + 1];
        double kz = knots[knot_idx * 3 + 2];

        double dx = qx - kx;
        double dy = qy - ky;
        double dz = qz - kz;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        double threshold = thresholds[knot_idx];
        double w = weightFunction(dist, threshold);

        weights[query_idx] += w;
        weight_sums[query_idx] += w;
    }

    // normalize weights
    for(int i = 0; i < numQueries; ++i) {
        if (weight_sums[i] > 0) {
            weights[i] /= weight_sums[i];
        }
    }

    return weights;
}


}