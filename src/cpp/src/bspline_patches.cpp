#include "cpgeo.h"
#include "cpgeo_mapping.h"
#include "csr_cg.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace {
using XYZ = std::array<double, 3>;
using GridPoint = std::array<int, 3>;

// Face order +X, -X, +Y, -Y, +Z, -Z. Increasing (u,v) winds outward.
template<class T>
std::array<T, 3> cube_point(int face, T extent, T u, T v) {
    std::array<T, 3> point{};
    const int axis = face / 2;
    const int sign = face % 2 == 0 ? 1 : -1;
    point[axis] = sign * extent;
    point[(axis + 1) % 3] = sign * u;
    point[(axis + 2) % 3] = v;
    return point;
}

struct Basis {
    int first;
    std::array<double, 4> value{}, derivative{};
};

Basis cubic(double t, int spans) {
    const int cell = std::min(static_cast<int>(t * spans), spans - 1);
    const int span = cell + 3;
    auto knot = [spans](int i) { return std::clamp(i - 3, 0, spans) / double(spans); };
    Basis basis{cell};
    basis.value[0] = 1;
    for (int degree = 1; degree <= 3; ++degree) {
        double carry = 0, derivative_carry = 0;
        for (int j = 0; j < degree; ++j) {
            const double start = knot(span + 1 + j - degree), end = knot(span + 1 + j);
            const double ratio = basis.value[j] / (end - start);
            basis.value[j] = carry + (end - t) * ratio;
            carry = (t - start) * ratio;
            if (degree == 3) {
                basis.derivative[j] = derivative_carry - 3 * ratio;
                derivative_carry = 3 * ratio;
            }
        }
        basis.value[degree] = carry;
        if (degree == 3) basis.derivative[degree] = derivative_carry;
    }
    return basis;
}

std::vector<double> sample_face(int face, const std::vector<double>& parameters,
                                cpgeo::SpaceTree& tree, std::span<const double> controls) {
    std::vector<double> points;
    points.reserve(3 * parameters.size() * parameters.size());
    for (double u : parameters) {
        for (double v : parameters) {
            auto p = cube_point(face, 1.0, 2*u - 1, 2*v - 1);
            double norm = std::hypot(p[0], p[1], p[2]);
            for (double x : p) points.push_back(x / norm);
        }
    }
    return cpgeo::map_points_batch(points, tree, controls);
}

std::vector<double> fit(cpgeo::SpaceTree& tree, std::span<const double> controls, int spans) {
    const int n = spans + 3;
    std::map<GridPoint, int> global;
    std::vector<int> indices;
    std::vector<bool> corner;
    // Identify identical control variables on edges, including reversed edges.
    for (int f = 0; f < 6; ++f) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                auto key = cube_point(f, n-1, 2*i-(n-1), 2*j-(n-1));
                auto [it, inserted] = global.try_emplace(key, static_cast<int>(global.size()));
                if (inserted) corner.push_back((i == 0 || i == n-1) && (j == 0 || j == n-1));
                indices.push_back(it->second);
            }
        }
    }
    const int samples = 2*spans + 3;
    std::vector<double> parameters;
    std::vector<Basis> basis;
    std::vector<std::array<double, 7>> gram(n);
    for (int i = 0; i < samples; ++i) {
        parameters.push_back(double(i) / (samples - 1));
        auto b = cubic(parameters.back(), spans);
        basis.push_back(b);
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                gram[b.first+j][k-j+3] += b.value[j] * b.value[k];
    }

    // The tensor-product normal matrix on one patch is the Kronecker product
    // of the two 1-D Gram matrices. Assemble all six into the shared variables.
    const int count = static_cast<int>(global.size());
    std::vector<std::map<int, double>> matrix(count);
    std::vector<XYZ> rhs(count), poles(count);
    for (int f = 0; f < 6; ++f) {
        const int* ids = indices.data() + f*n*n;
        auto target = sample_face(f, parameters, tree, controls);
        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                const double* p = target.data() + 3*(i*samples+j);
                for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) {
                        int id = ids[(basis[i].first+a)*n + basis[j].first+b];
                        double w = basis[i].value[a] * basis[j].value[b];
                        for (int d = 0; d < 3; ++d) rhs[id][d] += w*p[d];
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int row = ids[i*n+j];
                if (corner[row]) {
                    const int sample = (i == 0 ? 0 : samples-1)*samples + (j == 0 ? 0 : samples-1);
                    std::copy_n(target.data() + 3*sample, 3, poles[row].begin());
                    continue;
                }
                for (int a = std::max(i-3,0); a <= std::min(i+3,n-1); ++a)
                    for (int b = std::max(j-3,0); b <= std::min(j+3,n-1); ++b)
                        matrix[row][ids[a*n+b]] += gram[i][a-i+3] * gram[j][b-j+3];
            }
        }
    }
    std::vector<int> offsets{0}, columns;
    std::vector<double> values;
    for (int row = 0; row < count; ++row) {
        if (corner[row]) {
            columns.push_back(row);
            values.push_back(1);
            rhs[row] = poles[row];
        } else {
            for (auto [col, value] : matrix[row]) {
                if (corner[col]) {
                    for (int d = 0; d < 3; ++d) rhs[row][d] -= value * poles[col][d];
                } else {
                    columns.push_back(col);
                    values.push_back(value);
                }
            }
        }
        offsets.push_back(static_cast<int>(values.size()));
    }
    for (int d = 0; d < 3; ++d) {
        std::vector<double> b(count), x;
        for (int i = 0; i < count; ++i) b[i] = rhs[i][d];
        double norm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
        if (!cpgeo::conjugate_gradient(offsets, columns, values, b, x, 1500,
                                       1e-13*std::max(norm, 1e-100), 0.0))
            throw std::runtime_error("Singular spline fit");
        for (int i = 0; i < count; ++i)
            if (!corner[i]) poles[i][d] = x[i];
    }
    std::vector<double> result;
    result.reserve(3*indices.size());
    for (int id : indices) result.insert(result.end(), poles[id].begin(), poles[id].end());
    return result;
}

XYZ cross(const XYZ& a, const XYZ& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

// Eight subintervals per knot span, including knot lines and the patch boundary.
// Match seam samples by cube coordinates; compare all three normals at corners.
std::array<double, 3> validate(const std::vector<double>& poles, int spans,
                               cpgeo::SpaceTree& tree, std::span<const double> controls) {
    const int n = spans + 3, extent = 8*spans;
    std::vector<double> parameters;
    std::vector<Basis> basis;
    for (int t = 0; t <= extent; ++t) {
        parameters.push_back(double(t) / extent);
        basis.push_back(cubic(parameters.back(), spans));
    }
    const int samples = extent + 1;
    double max_error = 0, sum_squared = 0, max_angle = 0;
    std::map<GridPoint, std::vector<XYZ>> normals;
    for (int f = 0; f < 6; ++f) {
        auto targets = sample_face(f, parameters, tree, controls);
        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                XYZ p{}, du{}, dv{};
                bool boundary = i == 0 || j == 0 || i+1 == samples || j+1 == samples;
                for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) {
                        const double* pole = poles.data() + 3*((f*n+basis[i].first+a)*n+basis[j].first+b);
                        for (int d = 0; d < 3; ++d) {
                            p[d] += basis[i].value[a]*basis[j].value[b]*pole[d];
                            if (boundary) {
                                du[d] += basis[i].derivative[a]*basis[j].value[b]*pole[d];
                                dv[d] += basis[i].value[a]*basis[j].derivative[b]*pole[d];
                            }
                        }
                    }
                }
                for (int d = 0; d < 3; ++d) p[d] -= targets[3*(i*samples+j)+d];
                double error = std::hypot(p[0],p[1],p[2]);
                if (!std::isfinite(error)) throw std::runtime_error("Invalid sample");
                max_error = std::max(max_error, error);
                sum_squared += error*error;
                if (boundary) {
                    auto normal = cross(du, dv);
                    double norm = std::hypot(normal[0],normal[1],normal[2]);
                    if (!(norm > 0) || !std::isfinite(norm)) throw std::runtime_error("Singular seam");
                    for (double& x : normal) x /= norm;
                    auto& others = normals[cube_point(f, extent, 2*i-extent, 2*j-extent)];
                    for (const auto& other : others) {
                        auto sine = cross(normal, other);
                        double cosine = std::inner_product(normal.begin(), normal.end(), other.begin(), 0.0);
                        max_angle = std::max(max_angle, std::atan2(std::hypot(sine[0],sine[1],sine[2]), cosine));
                    }
                    others.push_back(normal);
                }
            }
        }
    }
    return {max_error, std::sqrt(sum_squared/(6*samples*samples)), max_angle*180/std::numbers::pi};
}
} // namespace

extern "C" CPGEO_API int cpgeo_fit_bspline_patches(cpgeo_handle_t handle, const double* controls,
    int count, int spans, double* out_poles, double* out_metrics) {
    if (!handle || !controls || !out_poles || !out_metrics || count <= 0 || spans < 1 || spans > 128)
        return -1;
    try {
        auto& tree = *static_cast<cpgeo::SpaceTree*>(handle);
        if (tree.get_knots().size() != 3*static_cast<size_t>(count)) return -1;
        std::vector<double> centered(tree.get_knots().size());
        for (size_t i = 0; i < centered.size(); ++i) {
            centered[i] = controls[i] - controls[i%3];
            if (!std::isfinite(centered[i])) return -1;
        }
        auto poles = fit(tree, centered, spans);
        for (size_t i = 0; i < poles.size(); ++i) poles[i] += controls[i%3];
        auto metrics = validate(poles, spans, tree, {controls, centered.size()});
        std::copy(poles.begin(), poles.end(), out_poles);
        std::copy(metrics.begin(), metrics.end(), out_metrics);
        return 0;
    } catch (...) {
        return -2;
    }
}
