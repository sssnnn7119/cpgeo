#include "cpgeo.h"
#include "cpgeo_mapping.h"
#include "csr_cg.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
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

// Rectangular lattice for the four sides and two caps. The same integer
// coordinates identify seam poles and validation samples in adjacent tiles.
struct PatchAtlas {
    int bands = 1;
    XYZ north{}, tangent{}, bitangent{};
    double inverse_pole = 0, north_radius = 0, south_radius = 0;

    int faces() const { return 4*bands + 2; }

    template<class T>
    std::array<T, 3> lattice(int patch, T extent, T u, T v) const {
        if (patch < 4*bands) {
            auto p = cube_point(patch % 4, extent, u, v);
            p[2] += (2*(patch/4) + 1 - bands)*extent;
            return p;
        }
        auto p = cube_point(4 + patch - 4*bands, extent, u, v);
        p[2] *= bands;
        return p;
    }

    XYZ sphere_point(int patch, double u, double v) const {
        auto p = lattice(patch, 1.0, u, v);
        if (north_radius == 0) {
            double norm = std::hypot(p[0], p[1], p[2]);
            for (double& x : p) x /= norm;
            return p;
        }
        // Homogeneous stereographic coordinates w/d. Linear square end caps
        // meet logarithmically spaced side bands along identical curves.
        std::complex<double> xy(p[0], p[1]), w, d = 1.0;
        if (patch == 4*bands) {
            w = north_radius * xy;
        } else if (patch == 4*bands + 1) {
            w = 1.0;
            d = south_radius * std::conj(xy);
        } else {
            double r = std::abs(xy), t = (1.0 + p[2]/bands)/2;
            double log_radius = t*std::log(north_radius*r) - (1-t)*std::log(south_radius*r);
            w = (std::exp(log_radius)/r)*xy;
        }
        // Möbius transform taking 0 and infinity to the two knot clusters.
        d += inverse_pole*w;
        double scale = std::norm(d) + std::norm(w);
        auto lateral = 2.0*w*std::conj(d);
        for (int i = 0; i < 3; ++i)
            p[i] = ((std::norm(d)-std::norm(w))*north[i] + lateral.real()*tangent[i]
                    + lateral.imag()*bitangent[i]) / scale;
        return p;
    }
};

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

std::vector<double> sample_face(const PatchAtlas& atlas, int face, const std::vector<double>& parameters,
                                cpgeo::SpaceTree& tree, std::span<const double> controls) {
    std::vector<double> points;
    points.reserve(3 * parameters.size() * parameters.size());
    for (double u : parameters) {
        for (double v : parameters) {
            auto p = atlas.sphere_point(face, 2*u - 1, 2*v - 1);
            points.insert(points.end(), p.begin(), p.end());
        }
    }
    return cpgeo::map_points_batch(points, tree, controls);
}

std::vector<double> fit(const PatchAtlas& atlas, cpgeo::SpaceTree& tree, std::span<const double> controls, int spans) {
    const int n = spans + 3;
    std::map<GridPoint, int> global;
    std::vector<int> indices;
    std::vector<bool> corner;
    // Identify identical control variables on edges, including reversed edges.
    for (int f = 0; f < atlas.faces(); ++f) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                auto key = atlas.lattice(f, n-1, 2*i-(n-1), 2*j-(n-1));
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
    // of the two 1-D Gram matrices. Assemble patches into the shared variables.
    const int count = static_cast<int>(global.size());
    std::vector<std::map<int, double>> matrix(count);
    std::vector<XYZ> rhs(count), poles(count);
    for (int f = 0; f < atlas.faces(); ++f) {
        const int* ids = indices.data() + f*n*n;
        auto target = sample_face(atlas, f, parameters, tree, controls);
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
std::array<double, 3> validate(const PatchAtlas& atlas, const std::vector<double>& poles, int spans,
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
    for (int f = 0; f < atlas.faces(); ++f) {
        auto targets = sample_face(atlas, f, parameters, tree, controls);
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
                    auto& others = normals[atlas.lattice(f, extent, 2*i-extent, 2*j-extent)];
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
    return {max_error, std::sqrt(sum_squared/(double(atlas.faces())*samples*samples)), max_angle*180/std::numbers::pi};
}

int fit_patches(const PatchAtlas& atlas, cpgeo_handle_t handle, const double* controls,
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
        auto poles = fit(atlas, tree, centered, spans);
        for (size_t i = 0; i < poles.size(); ++i) poles[i] += controls[i%3];
        auto metrics = validate(atlas, poles, spans, tree, {controls, centered.size()});
        std::copy(poles.begin(), poles.end(), out_poles);
        std::copy(metrics.begin(), metrics.end(), out_metrics);
        return 0;
    } catch (...) {
        return -2;
    }
}

} // namespace

extern "C" CPGEO_API int cpgeo_fit_bspline_patches(cpgeo_handle_t handle, const double* controls,
    int count, int spans, double* out_poles, double* out_metrics) {
    return fit_patches({}, handle, controls, count, spans, out_poles, out_metrics);
}

extern "C" CPGEO_API int cpgeo_fit_banded_bspline_patches(cpgeo_handle_t handle, const double* controls,
    int count, int spans, const double* chart, int bands, double* out_poles, double* out_metrics) {
    if (!chart || bands < 1 || bands > 32) return -1;
    PatchAtlas atlas;
    atlas.bands = bands;
    // The caller supplies an orthonormal (north,tangent,bitangent) frame.
    std::copy_n(chart, 3, atlas.north.begin());
    std::copy_n(chart + 3, 3, atlas.tangent.begin());
    std::copy_n(chart + 6, 3, atlas.bitangent.begin());
    atlas.inverse_pole = chart[9];
    atlas.north_radius = chart[10];
    atlas.south_radius = chart[11];
    if (!std::all_of(chart, chart+12, [](double x) { return std::isfinite(x); })
        || !(atlas.north_radius > 0) || !(atlas.south_radius > 0)) return -1;
    return fit_patches(atlas, handle, controls, count, spans, out_poles, out_metrics);
}

// Map increasing cube coordinates (u,v) in [0,1]^2 to the reference sphere.
// On negative faces the two free axes exchange places to keep outward winding.
extern "C" CPGEO_API int cpgeo_patch_reference(int face, const double* chart,
    const double* uv, int count, double* xyz) {
    if (face < 0 || face >= 6 || !uv || !xyz || count < 0) return -1;
    PatchAtlas atlas;
    if (chart) {
        std::copy_n(chart, 3, atlas.north.begin());
        std::copy_n(chart + 3, 3, atlas.tangent.begin());
        std::copy_n(chart + 6, 3, atlas.bitangent.begin());
        atlas.inverse_pole = chart[9];
        atlas.north_radius = chart[10];
        atlas.south_radius = chart[11];
    }
    for (int i = 0; i < count; ++i) {
        double u = 2*uv[2*i]-1, v = 2*uv[2*i+1]-1;
        auto q = face % 2 ? atlas.sphere_point(face,-v,u) : atlas.sphere_point(face,u,v);
        std::copy(q.begin(), q.end(), xyz+3*i);
    }
    return 0;
}

// Cubic clamped interpolation at all simple knots, with prescribed first
// derivatives at both ends. At an interior knot only three poles contribute.
// Factor that tridiagonal matrix once for all coordinate columns.
extern "C" CPGEO_API int cpgeo_interpolate_cubic(const double* breaks, int size,
    const double* data, int columns, double* poles) {
    if (!breaks || !data || !poles || size < 2 || columns < 1) return -1;
    for (int i = 0; i < size; ++i)
        if (!std::isfinite(breaks[i]) || (i && breaks[i] <= breaks[i-1])) return -1;
    try {
        auto knot = [&](int i) { return breaks[std::clamp(i,0,size-1)]; };
        auto row = [columns](int i) { return static_cast<size_t>(columns)*i; };
        // Data rows: left derivative, knot values, right derivative.
        for (int c = 0; c < columns; ++c) {
            poles[c] = data[row(1)+c];
            poles[row(1)+c] = poles[c]+(knot(1)-knot(0))*data[c]/3;
            poles[row(size+1)+c] = data[row(size)+c];
            poles[row(size)+c] = poles[row(size+1)+c] -
                (knot(size-1)-knot(size-2))*data[row(size+1)+c]/3;
        }
        std::vector<double> upper(size);
        for (int i = 1; i < size-1; ++i) {
            const double left = knot(i)-knot(i-1), right = knot(i+1)-knot(i);
            double a = right*right / ((left+right)*(knot(i+1)-knot(i-2)));
            double c = left*left / ((left+right)*(knot(i+2)-knot(i-1)));
            double diag = 1-a-c-a*upper[i-1];
            upper[i] = c/diag;
            for (int d = 0; d < columns; ++d)
                poles[row(i+1)+d] = (data[row(i+1)+d]-a*poles[row(i)+d])/diag;
        }
        for (int i = size-2; i >= 1; --i)
            for (int d = 0; d < columns; ++d)
                poles[row(i+1)+d] -= upper[i]*poles[row(i+2)+d];
        return 0;
    } catch (...) { return -2; }
}
