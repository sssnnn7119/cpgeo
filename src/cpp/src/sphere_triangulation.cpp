#include "sphere_triangulation.h"
#include "triangulation.h"
#include "mesh_edge_flip.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace cpgeo {

namespace {

struct EdgeHash {
    std::size_t operator()(const std::pair<int, int>& edge) const noexcept {
        return std::hash<int>{}(edge.first) ^ (std::hash<int>{}(edge.second) << 1);
    }
};

inline std::array<double, 3> get_point(std::span<const double> points, int idx) {
    return {
        points[idx * 3],
        points[idx * 3 + 1],
        points[idx * 3 + 2]
    };
}

inline double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline std::array<double, 3> cross3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline double length3(const std::array<double, 3>& v) {
    return std::sqrt(dot3(v, v));
}

inline bool is_triangle_degenerate(
    const std::array<double, 3>& v0,
    const std::array<double, 3>& v1,
    const std::array<double, 3>& v2,
    double eps = 1e-12)
{
    auto e0 = std::array<double, 3>{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    auto e1 = std::array<double, 3>{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    auto n = cross3(e0, e1);
    return length3(n) < eps;
}

inline void orient_outward(
    std::array<int, 3>& tri,
    std::span<const double> points)
{
    auto v0 = get_point(points, tri[0]);
    auto v1 = get_point(points, tri[1]);
    auto v2 = get_point(points, tri[2]);

    auto e0 = std::array<double, 3>{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    auto e1 = std::array<double, 3>{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    auto n = cross3(e0, e1);

    auto center = std::array<double, 3>{
        (v0[0] + v1[0] + v2[0]) / 3.0,
        (v0[1] + v1[1] + v2[1]) / 3.0,
        (v0[2] + v1[2] + v2[2]) / 3.0
    };

    if (dot3(n, center) < 0.0) {
        std::swap(tri[1], tri[2]);
    }
}

} // namespace

SphereTriangulation::SphereTriangulation(std::span<const double> sphere_points_span)
    : sphere_points(sphere_points_span),
      num_points(static_cast<int>(sphere_points_span.size() / 3)) {
    triangles.reserve(static_cast<size_t>(num_points) * 2);
}

void SphereTriangulation::set_points(std::span<const double> sphere_points_span) {
    sphere_points = sphere_points_span;
    num_points = static_cast<int>(sphere_points_span.size() / 3);
}

std::pair<double, double> SphereTriangulation::stereographicProjection(int point_idx, bool north) const {
    double x = sphere_points[point_idx * 3];
    double y = sphere_points[point_idx * 3 + 1];
    double z = sphere_points[point_idx * 3 + 2];

    const double eps = 1e-15;
    if (north) {
        double denom = (1.0 - z + eps);
        return {2.0 * x / denom, 2.0 * y / denom};
    }
    double denom = (1.0 + z + eps);
    return {2.0 * x / denom, 2.0 * y / denom};
}

double SphereTriangulation::calculateTriangleQuality(const Triangle& tri) const {
    auto v0 = get_point(sphere_points, tri[0]);
    auto v1 = get_point(sphere_points, tri[1]);
    auto v2 = get_point(sphere_points, tri[2]);

    auto e0 = std::array<double, 3>{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    auto e1 = std::array<double, 3>{v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    auto e2 = std::array<double, 3>{v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]};

    double l0 = length3(e0);
    double l1 = length3(e1);
    double l2 = length3(e2);

    if (l0 < 1e-12 || l1 < 1e-12 || l2 < 1e-12) return 0.0;

    auto clamp = [](double v) { return std::max(-1.0, std::min(1.0, v)); };

    double cos0 = clamp(dot3(e0, std::array<double, 3>{-e2[0], -e2[1], -e2[2]}) / (l0 * l2));
    double cos1 = clamp(dot3(e1, std::array<double, 3>{-e0[0], -e0[1], -e0[2]}) / (l1 * l0));
    double cos2 = clamp(dot3(e2, std::array<double, 3>{-e1[0], -e1[1], -e1[2]}) / (l2 * l1));

    double a0 = std::acos(cos0);
    double a1 = std::acos(cos1);
    double a2 = std::acos(cos2);

    return std::min({a0, a1, a2});
}

std::vector<std::pair<int, int>> SphereTriangulation::extractBoundaryEdges() const {
    std::unordered_map<std::pair<int, int>, int, EdgeHash> edge_count;
    edge_count.reserve(triangles.size() * 3);

    for (const auto& tri : triangles) {
        int a = tri[0];
        int b = tri[1];
        int c = tri[2];
        auto e0 = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
        auto e1 = (b < c) ? std::make_pair(b, c) : std::make_pair(c, b);
        auto e2 = (c < a) ? std::make_pair(c, a) : std::make_pair(a, c);
        ++edge_count[e0];
        ++edge_count[e1];
        ++edge_count[e2];
    }

    std::vector<std::pair<int, int>> boundary_edges;
    boundary_edges.reserve(edge_count.size());
    for (const auto& kv : edge_count) {
        if (kv.second == 1) {
            boundary_edges.push_back(kv.first);
        }
    }
    return boundary_edges;
}

void SphereTriangulation::triangulateHemisphere(
    bool north,
    const std::vector<std::pair<int, int>>& boundary_edges)
{
    const double split_eps = 1e-12;
    std::unordered_set<int> selected;
    selected.reserve(static_cast<size_t>(num_points));

    for (int i = 0; i < num_points; ++i) {
        double z = sphere_points[i * 3 + 2];
        if (north) {
            if (z <= split_eps) {
                selected.insert(i);
            }
        } else {
            if (z > split_eps) {
                selected.insert(i);
            }
        }
    }

    for (const auto& edge : boundary_edges) {
        selected.insert(edge.first);
        selected.insert(edge.second);
    }

    if (selected.size() < 3) return;

    std::vector<int> local_to_global;
    local_to_global.reserve(selected.size());

    std::vector<double> projected;
    projected.reserve(selected.size() * 2);

    for (int idx : selected) {
        auto uv = stereographicProjection(idx, north);
        local_to_global.push_back(idx);
        projected.push_back(uv.first);
        projected.push_back(uv.second);
    }

    DelaunayTriangulation dt(std::span<const double>(projected.data(), projected.size()));
    dt.triangulate();

    if (dt.size() == 0) return;

    std::vector<int> local_tris(dt.size() * 3);
    dt.getTriangleIndices(local_tris.data());

    std::vector<Triangle> hemi_tris;
    hemi_tris.reserve(dt.size());

    for (size_t i = 0; i < dt.size(); ++i) {
        int l0 = local_tris[i * 3];
        int l1 = local_tris[i * 3 + 1];
        int l2 = local_tris[i * 3 + 2];
        int g0 = local_to_global[l0];
        int g1 = local_to_global[l1];
        int g2 = local_to_global[l2];

        Triangle tri = {g0, g1, g2};

        auto v0 = get_point(sphere_points, tri[0]);
        auto v1 = get_point(sphere_points, tri[1]);
        auto v2 = get_point(sphere_points, tri[2]);

        if (is_triangle_degenerate(v0, v1, v2)) {
            continue;
        }

        if (!north) {
            if (v0[2] <= split_eps && v1[2] <= split_eps && v2[2] <= split_eps) {
                continue;
            }
        }

        orient_outward(tri, sphere_points);
        hemi_tris.push_back(tri);
    }

    if (north) {
        triangles = std::move(hemi_tris);
    } else {
        triangles.insert(triangles.end(), hemi_tris.begin(), hemi_tris.end());
    }
}

void SphereTriangulation::filterLowQualityTriangles(double quality_threshold) {
    triangles.erase(
        std::remove_if(triangles.begin(), triangles.end(),
            [this, quality_threshold](const Triangle& tri) {
                return calculateTriangleQuality(tri) < quality_threshold;
            }),
        triangles.end());
}

void SphereTriangulation::improveQualityByEdgeFlipping(
    int /*excluded_point_idx*/,
    double /*region_radius*/,
    int max_iterations)
{
    if (triangles.empty()) return;

    std::vector<int> faces;
    faces.reserve(triangles.size() * 3);
    for (const auto& tri : triangles) {
        faces.push_back(tri[0]);
        faces.push_back(tri[1]);
        faces.push_back(tri[2]);
    }

    auto optimized = mesh_optimize_by_edge_flipping(
        std::span<const double>(sphere_points.data(), sphere_points.size()),
        3,
        std::span<const int>(faces.data(), faces.size()),
        max_iterations
    );

    triangles.clear();
    triangles.reserve(optimized.size() / 3);
    for (size_t i = 0; i + 2 < optimized.size(); i += 3) {
        Triangle tri = {optimized[i], optimized[i + 1], optimized[i + 2]};
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0) continue;
        orient_outward(tri, sphere_points);
        triangles.push_back(tri);
    }
}

void SphereTriangulation::triangulate() {
    triangles.clear();
    if (num_points < 4) return;

    struct TriangleKeyHash {
        std::size_t operator()(const std::array<int, 3>& k) const noexcept {
            return std::hash<int>{}(k[0]) ^ (std::hash<int>{}(k[1]) << 1) ^ (std::hash<int>{}(k[2]) << 2);
        }
    };

    auto make_key = [](const Triangle& tri) {
        std::array<int, 3> key = {tri[0], tri[1], tri[2]};
        std::sort(key.begin(), key.end());
        return key;
    };

    struct CandidateTri {
        Triangle tri{};
        std::array<std::pair<int, int>, 3> edges{};
        double quality = 0.0;
    };

    auto build_projection_tris = [this](bool north) {
        std::vector<double> projected;
        projected.reserve(static_cast<size_t>(num_points) * 2);
        for (int i = 0; i < num_points; ++i) {
            auto uv = stereographicProjection(i, north);
            projected.push_back(uv.first);
            projected.push_back(uv.second);
        }

        DelaunayTriangulation dt(std::span<const double>(projected.data(), projected.size()));
        dt.triangulate();

        std::vector<int> local_tris(dt.size() * 3);
        dt.getTriangleIndices(local_tris.data());

        std::vector<Triangle> out;
        out.reserve(dt.size());
        for (size_t i = 0; i < dt.size(); ++i) {
            Triangle tri = {local_tris[i * 3], local_tris[i * 3 + 1], local_tris[i * 3 + 2]};
            auto v0 = get_point(sphere_points, tri[0]);
            auto v1 = get_point(sphere_points, tri[1]);
            auto v2 = get_point(sphere_points, tri[2]);
            if (is_triangle_degenerate(v0, v1, v2)) continue;
            orient_outward(tri, sphere_points);
            out.push_back(tri);
        }
        return out;
    };

    auto build_convex_hull = [this]() {
        std::vector<Triangle> faces;
        if (num_points < 4) return faces;

        auto point = [this](int idx) {
            return get_point(sphere_points, idx);
        };

        auto distance_to_line = [&](int idx, int a, int b) {
            auto pa = point(a);
            auto pb = point(b);
            auto p = point(idx);
            auto ab = std::array<double, 3>{pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
            auto ap = std::array<double, 3>{p[0] - pa[0], p[1] - pa[1], p[2] - pa[2]};
            auto cp = cross3(ab, ap);
            double denom = length3(ab);
            return (denom < 1e-12) ? 0.0 : (length3(cp) / denom);
        };

        auto distance_to_plane = [&](int idx, int a, int b, int c) {
            auto pa = point(a);
            auto pb = point(b);
            auto pc = point(c);
            auto p = point(idx);
            auto ab = std::array<double, 3>{pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
            auto ac = std::array<double, 3>{pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
            auto n = cross3(ab, ac);
            double denom = length3(n);
            if (denom < 1e-12) return 0.0;
            auto ap = std::array<double, 3>{p[0] - pa[0], p[1] - pa[1], p[2] - pa[2]};
            return std::abs(dot3(n, ap)) / denom;
        };

        int i0 = 0;
        int i1 = 0;
        for (int i = 1; i < num_points; ++i) {
            if (sphere_points[i * 3] < sphere_points[i0 * 3]) i0 = i;
            if (sphere_points[i * 3] > sphere_points[i1 * 3]) i1 = i;
        }
        if (i0 == i1) return faces;

        int i2 = -1;
        double max_d = -1.0;
        for (int i = 0; i < num_points; ++i) {
            if (i == i0 || i == i1) continue;
            double d = distance_to_line(i, i0, i1);
            if (d > max_d) {
                max_d = d;
                i2 = i;
            }
        }
        if (i2 < 0 || max_d < 1e-9) return faces;

        int i3 = -1;
        max_d = -1.0;
        for (int i = 0; i < num_points; ++i) {
            if (i == i0 || i == i1 || i == i2) continue;
            double d = distance_to_plane(i, i0, i1, i2);
            if (d > max_d) {
                max_d = d;
                i3 = i;
            }
        }
        if (i3 < 0 || max_d < 1e-9) return faces;

        auto p0 = point(i0);
        auto p1 = point(i1);
        auto p2 = point(i2);
        auto p3 = point(i3);
        std::array<double, 3> inside_point = {
            (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0,
            (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0,
            (p0[2] + p1[2] + p2[2] + p3[2]) / 4.0
        };

        auto orient_face = [&](Triangle& tri) {
            auto a = point(tri[0]);
            auto b = point(tri[1]);
            auto c = point(tri[2]);
            auto ab = std::array<double, 3>{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            auto ac = std::array<double, 3>{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            auto n = cross3(ab, ac);
            auto ai = std::array<double, 3>{inside_point[0] - a[0], inside_point[1] - a[1], inside_point[2] - a[2]};
            if (dot3(n, ai) > 0.0) {
                std::swap(tri[1], tri[2]);
            }
        };

        faces = {
            {i0, i1, i2},
            {i0, i2, i3},
            {i0, i3, i1},
            {i1, i3, i2}
        };
        for (auto& f : faces) {
            orient_face(f);
        }

        const double eps = -1e-12;
        for (int p = 0; p < num_points; ++p) {
            if (p == i0 || p == i1 || p == i2 || p == i3) continue;

            std::vector<size_t> visible;
            visible.reserve(faces.size());
            for (size_t fi = 0; fi < faces.size(); ++fi) {
                const auto& f = faces[fi];
                auto a = point(f[0]);
                auto b = point(f[1]);
                auto c = point(f[2]);
                auto ab = std::array<double, 3>{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
                auto ac = std::array<double, 3>{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
                auto n = cross3(ab, ac);
                auto ap = std::array<double, 3>{sphere_points[p * 3] - a[0], sphere_points[p * 3 + 1] - a[1], sphere_points[p * 3 + 2] - a[2]};
                if (dot3(n, ap) > eps) {
                    visible.push_back(fi);
                }
            }

            if (visible.empty()) continue;

            std::unordered_map<std::pair<int, int>, int, EdgeHash> edge_count;
            edge_count.reserve(visible.size() * 3);
            for (size_t fi : visible) {
                const auto& f = faces[fi];
                int a = f[0];
                int b = f[1];
                int c = f[2];
                auto e0 = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
                auto e1 = (b < c) ? std::make_pair(b, c) : std::make_pair(c, b);
                auto e2 = (c < a) ? std::make_pair(c, a) : std::make_pair(a, c);
                edge_count[e0]++;
                edge_count[e1]++;
                edge_count[e2]++;
            }

            std::vector<std::pair<int, int>> horizon;
            horizon.reserve(edge_count.size());
            for (const auto& kv : edge_count) {
                if (kv.second == 1) {
                    horizon.push_back(kv.first);
                }
            }

            std::vector<Triangle> new_faces;
            new_faces.reserve(faces.size() - visible.size() + horizon.size());
            std::vector<char> removed(faces.size(), 0);
            for (size_t fi : visible) removed[fi] = 1;
            for (size_t fi = 0; fi < faces.size(); ++fi) {
                if (!removed[fi]) new_faces.push_back(faces[fi]);
            }

            for (const auto& e : horizon) {
                Triangle tri = {e.first, e.second, p};
                if (is_triangle_degenerate(point(tri[0]), point(tri[1]), point(tri[2]))) {
                    continue;
                }
                orient_face(tri);
                new_faces.push_back(tri);
            }

            faces.swap(new_faces);
        }

        for (auto& f : faces) {
            orient_outward(f, sphere_points);
        }
        return faces;
    };

    auto hull_faces = build_convex_hull();
    if (!hull_faces.empty()) {
        triangles = std::move(hull_faces);
        return;
    }

    auto north_tris = build_projection_tris(true);
    auto south_tris = build_projection_tris(false);

    std::unordered_set<std::array<int, 3>, TriangleKeyHash> seen;
    seen.reserve((north_tris.size() + south_tris.size()) * 2);

    auto add_if_new = [&](const Triangle& tri, std::vector<Triangle>& out) {
        auto key = make_key(tri);
        if (seen.insert(key).second) {
            out.push_back(tri);
            return true;
        }
        return false;
    };

    std::vector<Triangle> combined;
    combined.reserve(north_tris.size() + south_tris.size());

    for (const auto& tri : north_tris) {
        auto v0 = get_point(sphere_points, tri[0]);
        auto v1 = get_point(sphere_points, tri[1]);
        auto v2 = get_point(sphere_points, tri[2]);
        double cz = (v0[2] + v1[2] + v2[2]) / 3.0;
        if (cz <= 0.0) {
            add_if_new(tri, combined);
        }
    }

    for (const auto& tri : south_tris) {
        auto v0 = get_point(sphere_points, tri[0]);
        auto v1 = get_point(sphere_points, tri[1]);
        auto v2 = get_point(sphere_points, tri[2]);
        double cz = (v0[2] + v1[2] + v2[2]) / 3.0;
        if (cz >= 0.0) {
            add_if_new(tri, combined);
        }
    }

    std::vector<CandidateTri> candidates;
    candidates.reserve(north_tris.size() + south_tris.size());
    auto add_candidate = [this, &candidates](const Triangle& tri) {
        CandidateTri c;
        c.tri = tri;
        int a = tri[0];
        int b = tri[1];
        int cidx = tri[2];
        c.edges[0] = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
        c.edges[1] = (b < cidx) ? std::make_pair(b, cidx) : std::make_pair(cidx, b);
        c.edges[2] = (cidx < a) ? std::make_pair(cidx, a) : std::make_pair(a, cidx);
        c.quality = calculateTriangleQuality(tri);
        candidates.push_back(c);
    };

    for (const auto& tri : north_tris) {
        add_candidate(tri);
    }
    for (const auto& tri : south_tris) {
        add_candidate(tri);
    }

    std::unordered_map<std::pair<int, int>, std::vector<size_t>, EdgeHash> edge_to_candidates;
    edge_to_candidates.reserve(candidates.size() * 3);
    for (size_t i = 0; i < candidates.size(); ++i) {
        edge_to_candidates[candidates[i].edges[0]].push_back(i);
        edge_to_candidates[candidates[i].edges[1]].push_back(i);
        edge_to_candidates[candidates[i].edges[2]].push_back(i);
    }

    auto rebuild_edges = [](const std::vector<Triangle>& tris) {
        std::unordered_map<std::pair<int, int>, std::vector<size_t>, EdgeHash> edge_map;
        edge_map.reserve(tris.size() * 3);
        for (size_t i = 0; i < tris.size(); ++i) {
            const auto& tri = tris[i];
            int a = tri[0];
            int b = tri[1];
            int c = tri[2];
            auto e0 = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
            auto e1 = (b < c) ? std::make_pair(b, c) : std::make_pair(c, b);
            auto e2 = (c < a) ? std::make_pair(c, a) : std::make_pair(a, c);
            edge_map[e0].push_back(i);
            edge_map[e1].push_back(i);
            edge_map[e2].push_back(i);
        }
        return edge_map;
    };

    auto is_closed = [](const std::unordered_map<std::pair<int, int>, std::vector<size_t>, EdgeHash>& edge_map) {
        for (const auto& kv : edge_map) {
            if (kv.second.size() != 2) return false;
        }
        return true;
    };

    auto remove_duplicates = [&]() {
        std::unordered_map<std::array<int, 3>, std::pair<size_t, double>, TriangleKeyHash> best;
        best.reserve(combined.size() * 2);
        for (size_t i = 0; i < combined.size(); ++i) {
            auto key = make_key(combined[i]);
            double q = calculateTriangleQuality(combined[i]);
            auto it = best.find(key);
            if (it == best.end() || q > it->second.second) {
                best[key] = {i, q};
            }
        }

        std::vector<size_t> keep_indices;
        keep_indices.reserve(best.size());
        for (const auto& kv : best) {
            keep_indices.push_back(kv.second.first);
        }
        std::sort(keep_indices.begin(), keep_indices.end());

        std::vector<Triangle> deduped;
        deduped.reserve(keep_indices.size());
        seen.clear();
        seen.reserve(keep_indices.size() * 2);
        for (size_t idx : keep_indices) {
            const auto& tri = combined[idx];
            deduped.push_back(tri);
            seen.insert(make_key(tri));
        }
        combined.swap(deduped);
    };

    const int max_attempts = 10;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        bool changed = false;

        remove_duplicates();

        auto edge_map = rebuild_edges(combined);

        // Remove non-manifold edges by dropping lowest-quality triangles
        std::unordered_set<size_t> to_remove;
        for (const auto& kv : edge_map) {
            if (kv.second.size() > 2) {
                std::vector<std::pair<double, size_t>> ranked;
                ranked.reserve(kv.second.size());
                for (size_t idx : kv.second) {
                    ranked.emplace_back(calculateTriangleQuality(combined[idx]), idx);
                }
                std::sort(ranked.begin(), ranked.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                for (size_t k = 2; k < ranked.size(); ++k) {
                    to_remove.insert(ranked[k].second);
                }
            }
        }

        if (!to_remove.empty()) {
            std::vector<size_t> idxs(to_remove.begin(), to_remove.end());
            std::sort(idxs.begin(), idxs.end(), std::greater<size_t>());
            for (size_t idx : idxs) {
                auto key = make_key(combined[idx]);
                seen.erase(key);
                combined.erase(combined.begin() + idx);
            }
            changed = true;
        }

        edge_map = rebuild_edges(combined);
        if (is_closed(edge_map) && !changed) {
            break;
        }

        // Stitch boundary edges using candidate triangles (keep manifold constraint)
        bool added_any = false;
        for (const auto& kv : edge_map) {
            if (kv.second.size() == 1) {
                auto it = edge_to_candidates.find(kv.first);
                if (it == edge_to_candidates.end()) continue;

                double best_q = -1.0;
                size_t best_idx = static_cast<size_t>(-1);
                for (size_t cand_idx : it->second) {
                    const auto& cand = candidates[cand_idx];
                    auto key = make_key(cand.tri);
                    if (seen.find(key) != seen.end()) continue;

                    bool valid = true;
                    for (const auto& e : cand.edges) {
                        auto eit = edge_map.find(e);
                        if (eit != edge_map.end() && eit->second.size() >= 2) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) continue;

                    if (cand.quality > best_q) {
                        best_q = cand.quality;
                        best_idx = cand_idx;
                    }
                }

                if (best_idx != static_cast<size_t>(-1)) {
                    add_if_new(candidates[best_idx].tri, combined);
                    added_any = true;
                }
            }
        }

        if (added_any) {
            changed = true;
        }

        if (!changed) {
            break;
        }
    }

    remove_duplicates();
    triangles = std::move(combined);
    filterLowQualityTriangles(1e-6);

    auto final_edge_map = rebuild_edges(triangles);
    if (is_closed(final_edge_map)) {
        improveQualityByEdgeFlipping(-1, 0.0, 10);
    }
}

size_t SphereTriangulation::size() const {
    return triangles.size();
}

void SphereTriangulation::getTriangleIndices(std::span<int> results) const {
    if (results.size() < triangles.size() * 3) return;
    for (size_t i = 0; i < triangles.size(); ++i) {
        results[i * 3] = triangles[i][0];
        results[i * 3 + 1] = triangles[i][1];
        results[i * 3 + 2] = triangles[i][2];
    }
}

void SphereTriangulation::exportToObj(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file " << filename << std::endl;
        return;
    }

    file << "# Generated by CPGEO Sphere Triangulation\n";
    file << "# Vertices: " << num_points << "\n";
    file << "# Triangles: " << triangles.size() << "\n\n";

    for (int i = 0; i < num_points; ++i) {
        file << "v " << sphere_points[i * 3] << " "
             << sphere_points[i * 3 + 1] << " "
             << sphere_points[i * 3 + 2] << "\n";
    }

    file << "\n";
    for (const auto& tri : triangles) {
        file << "f " << (tri[0] + 1) << " "
             << (tri[1] + 1) << " "
             << (tri[2] + 1) << "\n";
    }

    file.close();
}

} // namespace cpgeo