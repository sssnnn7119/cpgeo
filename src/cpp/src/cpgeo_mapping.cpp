#include "cpgeo_mapping.h"
#include "space_tree.h"
#include "cpgeo.h"
#include "tensor.h"
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


inline static double weightFunction(const std::span<const double, 3> dp, const double threshold) {
    double dist = std::sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
    if (dist >= threshold) {
        return 0.0;
    }
    double ratio = dist / threshold;
    
    double weight = 20 * pow(ratio, 7) - 70 * pow(ratio, 6) + 84 * pow(ratio, 5) - 35 * pow(ratio,4) + 1;

    return weight;
}

inline static std::tuple<double, std::array<double, 3>> weightFunctionDerivative1(const std::span<const double> dp, const double threshold) {
    double dist = std::sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
    if (dist >= threshold) {
        return std::make_tuple(0.0, std::array<double, 3>{0.0, 0.0, 0.0});
    }
    double ratio = dist / threshold;

    double weight = 20 * pow(ratio, 7) - 70 * pow(ratio, 6) + 84 * pow(ratio, 5) - 35 * pow(ratio,4) + 1;
    std::array<double, 3> derivative{0.0, 0.0, 0.0};

    double temp = (140 * pow(ratio, 5) - 420 * pow(ratio, 4) + 420 * pow(ratio, 3) - 140 * pow(ratio,2)) / threshold / threshold;
    for (int dim = 0; dim < 3; dim++)
        derivative[dim] = temp * dp[dim];

    return std::make_tuple(weight, derivative);
}

inline static std::tuple<double, std::array<double, 3>, Tensor2D> weightFunctionDerivative2(const std::span<const double> dp, const double threshold) {
    double dist = std::sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
    if (dist >= threshold) {
        return std::tuple<double, std::array<double, 3>, Tensor2D>(0.0, std::array<double, 3>{0.0, 0.0, 0.0}, Tensor2D(3,3));
    }
    double ratio = dist / threshold;
    
    double weight = 20 * pow(ratio, 7) - 70 * pow(ratio, 6) + 84 * pow(ratio, 5) - 35 * pow(ratio,4) + 1;

    std::array<double, 3> derivative{0.0, 0.0, 0.0};
    double temp = (140 * pow(ratio, 5) - 420 * pow(ratio, 4) + 420 * pow(ratio, 3) - 140 * pow(ratio,2)) / threshold / threshold;
    for (int dim = 0; dim < 3; dim++)
        derivative[dim] = temp * dp[dim];

    Tensor2D derivative2(3,3);
    temp = (700 * pow(ratio, 3) - 1680 * pow(ratio, 2) + 1260 * ratio - 280) / (threshold * threshold * threshold * threshold);
    for (int dim1 = 0; dim1 < 3; dim1++){
        for (int dim2 = 0; dim2 < 3; dim2++){
            derivative2.at(dim1, dim2) = temp * dp[dim1] * dp[dim2];
        }
    }

    temp = (140 * pow(ratio, 5) - 420 * pow(ratio, 4) + 420 * pow(ratio, 3) - 140 * pow(ratio,2)) / (threshold * threshold);
    for (int dim = 0; dim < 3; dim++){
        derivative2.at(dim, dim) += temp;
    }

    return std::tuple<double, std::array<double, 3>, Tensor2D>(weight, derivative, std::move(derivative2));
}

inline std::array<double, 2> stereographicProjection3_2(const std::span<const double, 3> cooSphere, bool north_pole) {
	std::array<double, 2> cooPlane;
	if (north_pole) {
		cooPlane[0] = 2 * cooSphere[0] / (1 - cooSphere[2]);
		cooPlane[1] = 2 * cooSphere[1] / (1 - cooSphere[2]);
	} else {
		cooPlane[0] = 2 * cooSphere[0] / (1 + cooSphere[2]);
		cooPlane[1] = 2 * cooSphere[1] / (1 + cooSphere[2]);
	}
    return cooPlane;
}

inline std::array<double, 3> stereographicProjection2_3(const std::span<const double, 2> cooPlane, bool north_pole) {
    std::array<double, 3> cooSphere;
    double t = 1 / (4 + cooPlane[0] * cooPlane[0] + cooPlane[1] * cooPlane[1]);
	cooSphere[0] = 4 * cooPlane[0] * t;
	cooSphere[1] = 4 * cooPlane[1] * t;
	if (north_pole) {
		cooSphere[2] = (1 - 8 * t);
	} else {
		cooSphere[2] = (8 * t - 1);
	}
    return cooSphere;
}

inline std::tuple<std::array<double, 3>, Tensor2D> stereographicProjection2_3Derivative1(const std::span<const double> cooPlane, bool north_pole) {
    
    // first get the mapping
    std::array<double, 3> cooSphere;
    double t = 1 / (4 + cooPlane[0] * cooPlane[0] + cooPlane[1] * cooPlane[1]);
    cooSphere[0] = 4 * cooPlane[0] * t;
    cooSphere[1] = 4 * cooPlane[1] * t;
	if (north_pole) {
		cooSphere[2] = (1 - 8 * t);
	} else {
		cooSphere[2] = (8 * t - 1);
	}

	// second get their first derivatives
    Tensor2D xdu(3, 2);
	std::array<double, 2> tdu;

	double t2 = t * t;
	tdu[0] = -2 * cooPlane[0] * t2;
	tdu[1] = -2 * cooPlane[1] * t2;

	xdu[0 * 2 + 0] = 4 * t + 4 * cooPlane[0] * tdu[0];
	xdu[0 * 2 + 1] = 4 * cooPlane[0] * tdu[1];
	xdu[1 * 2 + 0] = 4 * cooPlane[1] * tdu[0];
	xdu[1 * 2 + 1] = 4 * t + 4 * cooPlane[1] * tdu[1];
	if (north_pole) {
		xdu[2 * 2 + 0] = -8 * tdu[0];
		xdu[2 * 2 + 1] = -8 * tdu[1];
	} else {
		xdu[2 * 2 + 0] = 8 * tdu[0];
		xdu[2 * 2 + 1] = 8 * tdu[1];
	}

	return std::tuple<std::array<double, 3>, Tensor2D>(cooSphere, std::move(xdu));
}

inline std::tuple<std::array<double, 3>, Tensor2D, Tensor3D> stereographicProjection2_3Derivative2(const std::span<const double> cooPlane, bool north_pole) {

    // first get the mapping
    std::array<double, 3> cooSphere;
    double t = 1 / (4 + cooPlane[0] * cooPlane[0] + cooPlane[1] * cooPlane[1]);
    cooSphere[0] = 4 * cooPlane[0] * t;
    cooSphere[1] = 4 * cooPlane[1] * t;
	if (north_pole) {
		cooSphere[2] = (1 - 8 * t);
	} else {
		cooSphere[2] = (8 * t - 1);
	}

    // second get their first derivatives
    Tensor2D xdu(3, 2);

    std::array<double, 2> tdu;

    

    double t2 = t * t;
    tdu[0] = -2 * cooPlane[0] * t2;
    tdu[1] = -2 * cooPlane[1] * t2;

    xdu[0 * 2 + 0] = 4 * t + 4 * cooPlane[0] * tdu[0];
    xdu[0 * 2 + 1] = 4 * cooPlane[0] * tdu[1];
    xdu[1 * 2 + 0] = 4 * cooPlane[1] * tdu[0];
    xdu[1 * 2 + 1] = 4 * t + 4 * cooPlane[1] * tdu[1];
	if (north_pole) {
		xdu[2 * 2 + 0] = -8 * tdu[0];
		xdu[2 * 2 + 1] = -8 * tdu[1];
	} else {
		xdu[2 * 2 + 0] = 8 * tdu[0];
		xdu[2 * 2 + 1] = 8 * tdu[1];
	}

    // third get their second derivatives
    Tensor3D xdu2(3, 2, 2);
    Tensor2D tdu2(2, 2);

    tdu2.at(0,0) = -2 * t2 - 4 * cooPlane[0] * t * tdu[0];
    tdu2.at(0,1) = -4 * cooPlane[0] * t * tdu[1];
    tdu2.at(1,0) = tdu2.at(0,1);
    tdu2.at(1,1) = -2 * t2 - 4 * cooPlane[1] * t * tdu[1];

    xdu2.at(0, 0, 0) = 4 * tdu[0] + 4 * cooPlane[0] * tdu2.at(0,0) + 4 * tdu[0];
    xdu2.at(0, 0, 1) = 4 * tdu[1] + 4 * cooPlane[0] * tdu2.at(0,1);
    xdu2.at(0, 1, 0) = xdu2.at(0, 0, 1);
    xdu2.at(0, 1, 1) = 4 * cooPlane[0] * tdu2.at(1,1);
    xdu2.at(1, 0, 0) = 4 * cooPlane[1] * tdu2.at(0,0);
    xdu2.at(1, 0, 1) = 4 * tdu[0] + 4 * cooPlane[1] * tdu2.at(0,1);
    xdu2.at(1, 1, 0) = xdu2.at(1, 0, 1);
    xdu2.at(1, 1, 1) = 4 * tdu[1] + 4 * cooPlane[1] * tdu2.at(1,1) + 4 * tdu[1];
	if (north_pole) {
		xdu2.at(2, 0, 0) = -8 * tdu2.at(0,0);
		xdu2.at(2, 0, 1) = -8 * tdu2.at(0,1);
		xdu2.at(2, 1, 0) = -8 * tdu2.at(1,0);
		xdu2.at(2, 1, 1) = -8 * tdu2.at(1,1);
	} else {
		xdu2.at(2, 0, 0) = 8 * tdu2.at(0,0);
		xdu2.at(2, 0, 1) = 8 * tdu2.at(0,1);
		xdu2.at(2, 1, 0) = 8 * tdu2.at(1,0);
		xdu2.at(2, 1, 1) = 8 * tdu2.at(1,1);
	}

    return std::tuple<std::array<double, 3>, Tensor2D, Tensor3D>(cooSphere, std::move(xdu), std::move(xdu2));
}


std::vector<double> get_weights(
    const std::span<const int> indices,
    const std::span<const double> knots, 
    const std::span<const double> thresholds, 
    const std::span<const double, 3> query_points) {

    const int numCps = knots.size() / 3;
    const int numIndices = indices.size();

    std::vector<double> weights(numIndices, 0.0);
    Tensor2D weight_dx(numIndices, 3);
    double weight_sums = 0.0;
    std::array<double, 3> weight_sum_dx = { 0.0, 0.0, 0.0 };


    double qx = query_points[0];
    double qy = query_points[1];
    double qz = query_points[2];

    // compute the initial weights
#pragma omp parallel for reduction(+:weight_sums)
    for (int idx = 0; idx < numIndices; ++idx) {
        int knot_idx = indices[idx];

        double kx = knots[knot_idx * 3];
        double ky = knots[knot_idx * 3 + 1];
        double kz = knots[knot_idx * 3 + 2];

        std::array<double, 3> dp = { qx - kx, qy - ky, qz - kz };

        double threshold = thresholds[knot_idx];
        auto result = weightFunctionDerivative1(dp, threshold);

        weights[idx] = std::get<0>(result);

        weight_sums += weights[idx];
    }
    if (weight_sums == 0.0) {
        return weights;
    }
    // normalize weights
    std::vector<double> rdot(numIndices, 0.0);
    Tensor2D rdudot(numIndices, 2);

    for (int idx = 0; idx < numIndices; ++idx) {
        int query_idx = indices[idx];
        if (weight_sums > 0) {
            rdot[idx] = weights[idx] / weight_sums;
        }
    }

    return rdot;
}

std::vector<double> get_weights(
    const std::span<const int> indices,
    const std::span<const double> knots,
    const std::span<const double> thresholds,
    const std::span<const double, 2> query_points_plane,
    bool north_pole) {

    const int numCps = knots.size() / 3;
    const int numIndices = indices.size();

    std::vector<double> weights(numIndices, 0.0);
    Tensor2D weight_dx(numIndices, 3);
    double weight_sums = 0.0;
    std::array<double, 3> weight_sum_dx = { 0.0, 0.0, 0.0 };


	auto query_points = stereographicProjection2_3(query_points_plane, north_pole);

    double qx = query_points[0];
    double qy = query_points[1];
    double qz = query_points[2];

    // compute the initial weights
//#pragma omp parallel for reduction(+:weight_sums)
    for (int idx = 0; idx < numIndices; ++idx) {
        int knot_idx = indices[idx];




        double kx = knots[knot_idx * 3];
        double ky = knots[knot_idx * 3 + 1];
        double kz = knots[knot_idx * 3 + 2];

        std::array<double, 3> dp = { qx - kx, qy - ky, qz - kz };

        double threshold = thresholds[knot_idx];
        auto result = weightFunctionDerivative1(dp, threshold);

        weights[idx] = std::get<0>(result);

        weight_sums += weights[idx];
    }
    if (weight_sums == 0.0) {
        return weights;
    }
    // normalize weights
    std::vector<double> rdot(numIndices, 0.0);
    Tensor2D rdudot(numIndices, 2);

    for (int idx = 0; idx < numIndices; ++idx) {
        int query_idx = indices[idx];
        if (weight_sums > 0) {
            rdot[idx] = weights[idx] / weight_sums;
        }
    }

    return rdot;
}

std::array<std::vector<double>, 2> get_weights_derivative1(
    const std::span<const int> indices,
    const std::span<const double> knots,
    const std::span<const double> thresholds,
    const std::span<const double, 2> query_points_plane,
    bool north_pole) {

    const int numCps = knots.size() / 3;
    const int numIndices = indices.size();

    std::vector<double> weights(numIndices, 0.0);
    std::vector<double> _weight_dx(numIndices * 3, 0.0);
    TensorView2D weight_dx(numIndices, 3, _weight_dx);
    double weight_sums = 0.0;
    std::array<double, 3> weight_sum_dx = { 0.0, 0.0, 0.0 };

    auto [query_points, xdu] = stereographicProjection2_3Derivative1(query_points_plane, north_pole);

    // compute the initial weights
//#pragma omp parallel for
    for (int idx = 0; idx < numIndices; ++idx) {
        int knot_idx = indices[idx];


        double qx = query_points[0];
        double qy = query_points[1];
        double qz = query_points[2];

        double kx = knots[knot_idx * 3];
        double ky = knots[knot_idx * 3 + 1];
        double kz = knots[knot_idx * 3 + 2];

        std::array<double, 3> dp = { qx - kx, qy - ky, qz - kz };

        double threshold = thresholds[knot_idx];
        auto result = weightFunctionDerivative1(dp, threshold);

        weights[idx] = std::get<0>(result);
        weight_dx.at(idx, 0) = std::get<1>(result)[0];
        weight_dx.at(idx, 1) = std::get<1>(result)[1];
        weight_dx.at(idx, 2) = std::get<1>(result)[2];

        weight_sums += weights[idx];
        weight_sum_dx[0] += weight_dx.at(idx, 0);
        weight_sum_dx[1] += weight_dx.at(idx, 1);
        weight_sum_dx[2] += weight_dx.at(idx, 2);
    }

    if (weight_sums == 0.0) {
        return { weights, _weight_dx };
    }

    // normalize weights
    std::vector<double> rdot(numIndices, 0.0);
	std::vector<double> _rdudot(numIndices * 2, 0.0);
	TensorView2D rdudot(2, numIndices, _rdudot);

    for (int idx = 0; idx < numIndices; ++idx) {
        int query_idx = indices[idx];
        if (weight_sums > 0) {
            rdot[idx] = weights[idx] / weight_sums;
            std::array<double, 3> rdxdot;
            rdxdot[0] = (weight_dx.at(idx, 0) / weight_sums) - (weights[idx] * weight_sum_dx[0]) / (weight_sums * weight_sums);
            rdxdot[1] = (weight_dx.at(idx, 1) / weight_sums) - (weights[idx] * weight_sum_dx[1]) / (weight_sums * weight_sums);
            rdxdot[2] = (weight_dx.at(idx, 2) / weight_sums) - (weights[idx] * weight_sum_dx[2]) / (weight_sums * weight_sums);

            // chain rule to get derivative respect to plane coordinates
            rdudot.at(0, idx) = rdxdot[0] * xdu.at(0, 0) + rdxdot[1] * xdu.at(1, 0) + rdxdot[2] * xdu.at(2, 0);
            rdudot.at(1, idx) = rdxdot[0] * xdu.at(0, 1) + rdxdot[1] * xdu.at(1, 1) + rdxdot[2] * xdu.at(2, 1);
        }


    }



    return { rdot, _rdudot };
}

std::array<std::vector<double>, 3> get_weights_derivative2(
    const std::span<const int> indices,
    const std::span<const double> knots,
    const std::span<const double> thresholds,
    const std::span<const double, 2> query_points_plane,
    bool north_pole) {

    const int numCps = knots.size() / 3;
    const int numIndices = indices.size();

    std::vector<double> weights(numIndices, 0.0);
    Tensor2D weight_dx(numIndices, 3);
    Tensor3D weight_dx2(numIndices, 3, 3);

    double weight_sums = 0.0;
    std::array<double, 3> weight_sum_dx = { 0.0, 0.0, 0.0 };
    Tensor2D weight_sum_dx2(3, 3);

    auto [query_points, xdu, xdu2] = stereographicProjection2_3Derivative2(query_points_plane, north_pole);

    // compute the initial weights
//#pragma omp parallel for
    for (int idx = 0; idx < numIndices; ++idx) {
        int knot_idx = indices[idx];


        double qx = query_points[0];
        double qy = query_points[1];
        double qz = query_points[2];

        double kx = knots[knot_idx * 3];
        double ky = knots[knot_idx * 3 + 1];
        double kz = knots[knot_idx * 3 + 2];

        std::array<double, 3> dp = { qx - kx, qy - ky, qz - kz };

        double threshold = thresholds[knot_idx];
        auto [w, wdx, wdx2] = weightFunctionDerivative2(dp, threshold);

        weights[idx] = w;
        weight_dx.at(idx, 0) = wdx[0];
        weight_dx.at(idx, 1) = wdx[1];
        weight_dx.at(idx, 2) = wdx[2];
		weight_dx2.at(idx, 0, 0) = wdx2.at(0, 0);
		weight_dx2.at(idx, 0, 1) = wdx2.at(0, 1);
		weight_dx2.at(idx, 0, 2) = wdx2.at(0, 2);
        weight_dx2.at(idx, 1, 0) = wdx2.at(1, 0);
		weight_dx2.at(idx, 1, 1) = wdx2.at(1, 1);
		weight_dx2.at(idx, 1, 2) = wdx2.at(1, 2);
		weight_dx2.at(idx, 2, 0) = wdx2.at(2, 0);
		weight_dx2.at(idx, 2, 1) = wdx2.at(2, 1);
		weight_dx2.at(idx, 2, 2) = wdx2.at(2, 2);

        weight_sums += weights[idx];
        weight_sum_dx[0] += weight_dx.at(idx, 0);
        weight_sum_dx[1] += weight_dx.at(idx, 1);
        weight_sum_dx[2] += weight_dx.at(idx, 2);
        weight_sum_dx2.at(0, 0) += weight_dx2.at(idx, 0, 0);
        weight_sum_dx2.at(0, 1) += weight_dx2.at(idx, 0, 1);
        weight_sum_dx2.at(0, 2) += weight_dx2.at(idx, 0, 2);
        weight_sum_dx2.at(1, 0) += weight_dx2.at(idx, 1, 0);
        weight_sum_dx2.at(1, 1) += weight_dx2.at(idx, 1, 1);
        weight_sum_dx2.at(1, 2) += weight_dx2.at(idx, 1, 2);
        weight_sum_dx2.at(2, 0) += weight_dx2.at(idx, 2, 0);
        weight_sum_dx2.at(2, 1) += weight_dx2.at(idx, 2, 1);
        weight_sum_dx2.at(2, 2) += weight_dx2.at(idx, 2, 2);
    }

    if (weight_sums == 0.0) {
        return { weights, std::vector<double>(numIndices * 2, 0.0), std::vector<double>(numIndices * 2 * 2, 0.0) };
    }

    // normalize weights
    std::vector<double> rdot(numIndices, 0.0);
	std::vector<double> _rdudot(2 * numIndices, 0.0);
	std::vector<double> _rdu2dot(2 * 2 * numIndices, 0.0);

	TensorView2D rdudot(2, numIndices, _rdudot);
	TensorView3D rdu2dot(2, 2, numIndices, _rdu2dot);

    for (int idx = 0; idx < numIndices; ++idx) {
        int query_idx = indices[idx];
        if (weight_sums > 0) {
            rdot[idx] = weights[idx] / weight_sums;
            std::array<double, 3> rdxdot;
            rdxdot[0] = (weight_dx.at(idx, 0) / weight_sums) - (weights[idx] * weight_sum_dx[0]) / (weight_sums * weight_sums);
            rdxdot[1] = (weight_dx.at(idx, 1) / weight_sums) - (weights[idx] * weight_sum_dx[1]) / (weight_sums * weight_sums);
            rdxdot[2] = (weight_dx.at(idx, 2) / weight_sums) - (weights[idx] * weight_sum_dx[2]) / (weight_sums * weight_sums);

            // chain rule to get derivative respect to plane coordinates
            rdudot.at(0, idx) = rdxdot[0] * xdu.at(0, 0) + rdxdot[1] * xdu.at(1, 0) + rdxdot[2] * xdu.at(2, 0);
            rdudot.at(1, idx) = rdxdot[0] * xdu.at(0, 1) + rdxdot[1] * xdu.at(1, 1) + rdxdot[2] * xdu.at(2, 1);

            Tensor2D rdxdot2(3, 3);
            for (int dim1 = 0; dim1 < 3; dim1++) {
                for (int dim2 = 0; dim2 < 3; dim2++) {
                    rdxdot2.at(dim1, dim2) = 
                        + (weight_dx2.at(idx, dim1, dim2) / weight_sums)
                        - (weight_dx.at(idx, dim1) * weight_sum_dx[dim2]) / (weight_sums * weight_sums)
                        - (weight_dx.at(idx, dim2) * weight_sum_dx[dim1]) / (weight_sums * weight_sums)
                        - (weights[idx] * weight_sum_dx2.at(dim1, dim2)) / (weight_sums * weight_sums)
                        + 2 * (weights[idx] * weight_sum_dx[dim1] * weight_sum_dx[dim2]) / (weight_sums * weight_sums * weight_sums);
                }
            }
            
            // chain rule to get second derivative respect to plane coordinates
            for (int dim1 = 0; dim1 < 2; dim1++)
                for (int dim2 = 0; dim2 < 2; dim2++){
                    rdu2dot.at(dim1, dim2, idx) = 0;

                    for(int i=0;i<3;++i){
                        for(int j=0;j<3;++j){
                            rdu2dot.at(dim1, dim2, idx) += rdxdot2.at(i, j) * xdu.at(i, dim1) * xdu.at(j, dim2);
                        }

                        rdu2dot.at(dim1, dim2, idx) += rdxdot[i] * xdu2.at(i, dim1, dim2);
                    }
                    
                }
            
        }


    }



    return { rdot, _rdudot, _rdu2dot};
}

// compute mapped points from control points using sparse weights

std::array<double, 3> get_mapped_points(
    const std::span<const int> indices_cps,
    const std::span<const double> weights,
    const std::span<const double> controlpoints) {

    std::array<double, 3> out_mapped = { 0.0, 0.0, 0.0 };
    int num_controlpoints = static_cast<int>(controlpoints.size() / 3);
    int num_indices = static_cast<int>(indices_cps.size());
    for (int j = 0; j < num_indices; ++j) {
        int cp_idx = indices_cps[j];
        double w = weights[j];
        if (cp_idx >= 0 && cp_idx < num_controlpoints) {
            for (int k = 0; k < 3; ++k) {
                out_mapped[k] += w * controlpoints[cp_idx * 3 + k];
            }
        }
    }

    return out_mapped;
}

void get_mapped_points_(
    const std::span<const int> indices_cps,
    const std::span<const double> weights,
    const std::span<const double> controlpoints,
    std::span<double, 3> result) {

    std::array<double, 3> out_mapped = { 0.0, 0.0, 0.0 };
    int num_controlpoints = static_cast<int>(controlpoints.size() / 3);
    int num_indices = static_cast<int>(indices_cps.size());
    for (int j = 0; j < num_indices; ++j) {
        int cp_idx = indices_cps[j];
        double w = weights[j];
        if (cp_idx >= 0 && cp_idx < num_controlpoints) {
            for (int k = 0; k < 3; ++k) {
                result[k] += w * controlpoints[cp_idx * 3 + k];
            }
        }
    }
}

std::vector<double> get_mapped_points_batch(
    const std::span<const int> indices_cps,
    const std::span<const int> indices_pts,
    const std::span<const double> weights,
    const std::span<const double> controlpoints) {

    int num_queries = 0;
    if (indices_pts.size() >= 1) {
        num_queries = static_cast<int>(indices_pts.size()) - 1;
    }
    int num_controlpoints = static_cast<int>(controlpoints.size() / 3);

    std::vector<double> out_mapped(num_queries * 3, 0.0);
    if (num_queries <= 0 || num_controlpoints <= 0) {
        return out_mapped;
    }

    // Each query writes to its own segment so this is safe to parallelize over queries
    #pragma omp parallel for
    for (int i = 0; i < num_queries; ++i) {
        int start = indices_pts[i];
        int end = indices_pts[i + 1];
        for (int j = start; j < end; ++j) {
            int cp_idx = indices_cps[j];
            double w = weights[j];
            if (cp_idx >= 0 && cp_idx < num_controlpoints) {
                for (int k = 0; k < 3; ++k) {
                    out_mapped[i * 3 + k] += w * controlpoints[cp_idx * 3 + k];
                }
            }
        }
    }

    return out_mapped;
} 

std::array<double, 3> map_points(
    std::span<const double, 3> query_point,
    SpaceTree& tree,
    const std::span<const double> controlpoints,
    bool north_pole
){
    auto indices_cps = tree.query_point(query_point[0], query_point[1], query_point[2]);
    const auto& thresholds_vec = tree.get_thresholds();
    const auto& knots_vec = tree.get_knots();
    auto weights = get_weights(indices_cps, knots_vec, thresholds_vec, stereographicProjection3_2(query_point, north_pole), north_pole);
    return get_mapped_points(indices_cps, weights, controlpoints);
}



std::vector<double> map_points_batch(
    std::span<const double> query_point,
    SpaceTree& tree,
    const std::span<const double> controlpoints,
    bool north_pole
){
    int num_queries = static_cast<int>(query_point.size() / 3);
    std::vector<double> out_mapped(num_queries * 3, 0.0);
    if (num_queries <= 0) {
        return out_mapped;
    }

    const auto& thresholds_vec = tree.get_thresholds();
    const auto& knots_vec = tree.get_knots();


    // for each query, compute weights and mapped points
    #pragma omp parallel for
    for (int i = 0; i < num_queries; ++i) {
        auto indices_cps = tree.query_point(query_point[3 * i], query_point[3 * i+1], query_point[3 * i+2]);

        std::array<double, 2> query_point_plane = stereographicProjection3_2(
            std::span<const double, 3>(query_point.data() + i * 3, 3),
            north_pole
        );

        auto weights = get_weights(indices_cps, knots_vec, thresholds_vec, query_point_plane, north_pole);
        get_mapped_points_(indices_cps, weights, controlpoints, std::span<double, 3>(out_mapped.data() + i * 3, 3));
    }

    return out_mapped;
}


std::array<std::vector<double>, 3> map_points_batch_derivative2(
    std::span<const double> query_point,
    SpaceTree& tree,
    const std::span<const double> controlpoints,
    bool north_pole
) {

    int num_queries = static_cast<int>(query_point.size() / 3);
    std::vector<double> _r(num_queries * 3, 0.0);
	std::vector<double> _rdu(num_queries * 3 * 2, 0.0);
	std::vector<double> _rdu2(num_queries * 3 * 2 * 2, 0.0);

    const auto& thresholds_vec = tree.get_thresholds();
    const auto& knots_vec = tree.get_knots();


    // for each query, compute weights and mapped points
#pragma omp parallel for
    for (int i = 0; i < num_queries; ++i) {
        auto indices_cps = tree.query_point(query_point[3 * i], query_point[3 * i + 1], query_point[3 * i + 2]);
		int num_indices = static_cast<int>(indices_cps.size());

        std::array<double, 2> query_point_plane = stereographicProjection3_2(
            std::span<const double, 3>(query_point.data() + i * 3, 3),
            north_pole
        );

        auto [rdot, rdudot, rdu2dot] = get_weights_derivative2(indices_cps, knots_vec, thresholds_vec, query_point_plane, north_pole);
        get_mapped_points_(indices_cps, rdot, controlpoints, std::span<double, 3>( _r.data() + i * 3, 3));

        // plane dimensions: 0..1
        for (int pd = 0; pd < 2; ++pd) {
            // fill rdu at (i, pd, :)
            get_mapped_points_(
                indices_cps,
                std::span<const double>(rdudot.data() + pd * num_indices, num_indices),
                controlpoints,
                std::span<double, 3>(_rdu.data() + i * 2 * 3 + pd * 3, 3)
            );

            // fill rdu2 at (i, pd, pd2, :)
            for (int pd2 = 0; pd2 < 2; ++pd2) {
                get_mapped_points_(
                    indices_cps,
                    std::span<const double>(rdu2dot.data() + (pd * 2 + pd2) * num_indices, num_indices),
                    controlpoints,
                    std::span<double, 3>(_rdu2.data() + i * 2 * 2 * 3 + (pd * 2 + pd2) * 3, 3)
                );
             }
         }
    }

    return { _r, _rdu, _rdu2};
}


}