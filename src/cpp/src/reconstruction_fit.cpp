#include "cpgeo.h"
#include "cpgeo_mapping.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

struct BasisRow {
    std::vector<int> indices;
    std::vector<double> weights;
};

// Damped CGLS for one coordinate of P = target + delta. The three
// coordinates share W, but each has its own residual and stopping criterion.
bool fit_coordinate(const std::vector<BasisRow>& basis, const double* targets,
                    int axis, double alpha, std::vector<double>& fitted) {
    const size_t n = basis.size();
    std::vector<double> delta(n, 0.0), residual(n, 0.0), gradient(n), direction(n), mapped(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& row = basis[i];
        // W is a partition of unity. Differences avoid cancellation far from the origin.
        for (size_t k = 0; k < row.indices.size(); ++k) {
            const size_t neighbor = row.indices[k];
            residual[i] += row.weights[k] * (targets[3 * i + axis] - targets[3 * neighbor + axis]);
        }
    }

    auto update_gradient = [&]() {
        for (size_t i = 0; i < n; ++i) gradient[i] = -alpha * delta[i];
        for (size_t i = 0; i < n; ++i) {
            const auto& row = basis[i];
            for (size_t k = 0; k < row.indices.size(); ++k)
                gradient[row.indices[k]] += row.weights[k] * residual[i];
        }
        return std::inner_product(gradient.begin(), gradient.end(), gradient.begin(), 0.0);
    };

    double norm2 = update_gradient();
    if (!std::isfinite(norm2)) return false;
    const double tolerance2 = 1e-20 * norm2;
    direction = gradient;
    // In exact arithmetic at most n steps are needed. Allow for loss of
    // conjugacy, especially in the unregularized fit.
    const size_t max_steps = std::min<size_t>(10000, 8 * n);
    for (size_t step = 0; norm2 > tolerance2; ++step) {
        if (step == max_steps) return false;
        double curvature = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const auto& row = basis[i];
            double value = 0.0;
            for (size_t k = 0; k < row.indices.size(); ++k)
                value += row.weights[k] * direction[row.indices[k]];
            mapped[i] = value;
            curvature += value * value + alpha * direction[i] * direction[i];
        }
        if (!(curvature > 0.0) || !std::isfinite(curvature)) return false;
        const double length = norm2 / curvature;
        for (size_t i = 0; i < n; ++i) {
            delta[i] += length * direction[i];
            residual[i] -= length * mapped[i];
        }
        const double next_norm2 = update_gradient();
        if (!std::isfinite(next_norm2)) return false;
        const double beta = next_norm2 / norm2;
        for (size_t i = 0; i < n; ++i) direction[i] = gradient[i] + beta * direction[i];
        norm2 = next_norm2;
    }
    for (size_t i = 0; i < n; ++i) {
        fitted[3 * i + axis] = targets[3 * i + axis] + delta[i];
        if (!std::isfinite(fitted[3 * i + axis])) return false;
    }
    return true;
}

} // namespace

extern "C" CPGEO_API int cpgeo_fit_reconstructed_control_points(
    cpgeo_handle_t handle, const double* targets, int num_points, double alpha, double* out_points
) {
    if (!handle || !targets || !out_points || num_points <= 0 || !std::isfinite(alpha) || alpha < 0.0)
        return -1;
    try {
        const auto& tree = *static_cast<const cpgeo::SpaceTree*>(handle);
        const auto& knots = tree.get_knots();
        if (knots.size() / 3 != static_cast<size_t>(num_points)) return -1;
        if (!std::all_of(targets, targets + knots.size(), [](double v) { return std::isfinite(v); }))
            return -1;

        std::vector<BasisRow> basis(num_points);
        for (size_t i = 0; i < basis.size(); ++i) {
            auto point = std::span<const double, 3>(knots.data() + 3 * i, 3);
            auto& row = basis[i];
            row.indices = tree.query_point(point[0], point[1], point[2]);
            row.weights = cpgeo::get_weights(row.indices, knots, tree.get_thresholds(), point);
            if (!(std::accumulate(row.weights.begin(), row.weights.end(), 0.0) > 0.0)) return -1;
        }

        std::vector<double> fitted(knots.size());
        for (int axis = 0; axis < 3; ++axis) {
            if (!fit_coordinate(basis, targets, axis, alpha, fitted)) return -2;
        }
        std::copy(fitted.begin(), fitted.end(), out_points);
        return 0;
    } catch (...) {
        return -1;
    }
}
