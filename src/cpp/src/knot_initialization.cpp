#include "cpgeo.h"
#include "cpgeo_mapping.h"
#include "mesh_utils.h"
#include "csr_cg.h"

#include <numbers>
#include <numeric>
#include <stdexcept>

namespace {

using UV = std::array<double, 2>;
using XYZ = std::array<double, 3>;
using Graph = std::vector<std::vector<int>>;
using Mask = std::vector<char>;

Graph mesh_graph(std::span<const int> faces, int n) {
    for (int v : faces)
        if (v < 0 || v >= n) throw std::invalid_argument("Invalid vertex index");
    const auto edges = cpgeo::extractEdgesWithNumber(faces);
    if (n + faces.size() / 3 != edges.size() + 2) throw std::invalid_argument("Expected spherical topology");
    Graph graph(n);
    for (const auto& [edge, count] : edges) {
        auto [a, b] = edge;
        if (a == b || count != 2) throw std::invalid_argument("Expected closed triangular mesh");
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    return graph;
}

struct Disk {
    Mask half;
    std::vector<int> boundary;
};

// All edges have unit cost. BFS gives the same distances as the former Dijkstra traversal.
Disk split_mesh(const Graph& graph, std::span<const int> faces) {
    const int n = static_cast<int>(graph.size());
    const int source = n > 50 ? 50 : 0;
    std::vector<int> distance(n, -1), queue{source};
    distance[source] = 0;
    for (size_t head = 0; head < queue.size(); ++head) {
        int v = queue[head];
        for (int w : graph[v]) {
            if (distance[w] >= 0) continue;
            distance[w] = distance[v] + 1;
            queue.push_back(w);
        }
    }
    if (queue.size() != graph.size()) throw std::invalid_argument("Disconnected control mesh");
    int middle = distance[queue[n / 2 + (n % 4 == 3)]]; // Python round(n/2), ties to even
    for (int t = middle; t >= 2; t = static_cast<int>(t / 1.5)) {
        Mask half(n);
        for (int v = 0; v < n; ++v)
            half[v] = distance[v] < t;
        std::vector<int> patch;
        for (size_t i = 0; i < faces.size(); i += 3)
            if (half[faces[i]] && half[faces[i + 1]] && half[faces[i + 2]])
                patch.insert(patch.end(), faces.begin() + i, faces.begin() + i + 3);
        auto loops = cpgeo::extractBoundaryLoops(patch);
        if (loops.size() != 1) continue;
        if (2 * std::count(half.begin(), half.end(), 1) < n)
            for (auto& h : half)
                h = !h;
        return {std::move(half), std::move(loops.front())};
    }
    return {}; // A coarse mesh may have no disk cut with interior vertices on both sides.
}

// Minimizing sum_edges ||uv_i - uv_j||^2 with Dirichlet data gives the
// unweighted graph Laplacian. Assemble its free rows once for both coordinates.
void harmonic_disk(std::vector<UV>& uv, const Graph& graph, const Mask& free, std::span<const int> faces) {
    std::vector<int> index(uv.size(), -1), vertices;
    for (size_t i = 0; i < uv.size(); ++i) {
        if (!free[i]) continue;
        index[i] = static_cast<int>(vertices.size());
        vertices.push_back(i);
    }
    if (vertices.size() == uv.size()) throw std::runtime_error("Ungrounded chart");
    std::vector<int> offsets{0}, columns;
    std::vector<double> values;
    std::array<std::vector<double>, 2> rhs;
    for (int i : vertices) {
        columns.push_back(index[i]);
        values.push_back(graph[i].size());
        UV boundary{};
        for (int j : graph[i]) {
            if (free[j]) {
                columns.push_back(index[j]);
                values.push_back(-1);
            } else {
                boundary[0] += uv[j][0];
                boundary[1] += uv[j][1];
            }
        }
        for (int d = 0; d < 2; ++d)
            rhs[d].push_back(boundary[d]);
        offsets.push_back(static_cast<int>(columns.size()));
    }
    for (int d = 0; d < 2; ++d) {
        const auto& b = rhs[d];
        const double norm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
        std::vector<double> x;
        if (!cpgeo::conjugate_gradient(offsets, columns, values, b, x,
                                       std::max(100, 2 * static_cast<int>(vertices.size())),
                                       1e-14 * std::max(norm, 1.0), 0.0))
            throw std::runtime_error("Harmonic solve did not converge");
        for (size_t i = 0; i < vertices.size(); ++i)
            uv[vertices[i]][d] = x[i];
    }
    auto a = uv[faces[0]], b = uv[faces[1]], c = uv[faces[2]];
    if ((b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0]))
        for (auto& p : uv)
            p[0] = -p[0];
}

void change_chart(std::vector<UV>& uv, bool reflect, bool cycle) {
    for (auto& p : uv) {
        auto s = cpgeo::stereographicProjection2_3(p);
        if (reflect) s[2] = -s[2];
        if (cycle) s = {s[1], s[2], s[0]};
        p = s[2] == 1.0 ? UV{} : cpgeo::stereographicProjection3_2(s);
    }
}

void circle(std::vector<UV>& uv, std::span<const int> loop, double radius) {
    for (size_t i = 0; i < loop.size(); ++i) {
        double angle = (double(i) / loop.size()) * (2 * std::numbers::pi);
        uv[loop[i]] = {radius * std::cos(angle), radius * std::sin(angle)};
    }
}

// When a chart collapses, place a vertex at the north pole and embed the
// remaining disk in a convex polygon. Radial projection of the disk plane
// preserves triangle winding and the pole closes the complementary cap.
std::vector<XYZ> polar_disk(const Graph& graph, std::span<const int> faces) {
    const int pole = faces.front();
    std::vector<int> fan;
    for (size_t i = 0; i < faces.size(); i += 3) {
        auto tri = faces.subspan(i, 3);
        if (std::find(tri.begin(), tri.end(), pole) != tri.end())
            fan.insert(fan.end(), tri.begin(), tri.end());
    }
    auto loops = cpgeo::extractBoundaryLoops(fan);
    if (loops.size() != 1 || loops.front().size() != graph[pole].size())
        throw std::invalid_argument("Expected a manifold vertex fan");
    Mask free(graph.size(), true);
    free[pole] = false;
    for (int v : loops.front())
        free[v] = false;
    std::vector<UV> uv(graph.size());
    circle(uv, loops.front(), 1.0);
    harmonic_disk(uv, graph, free, faces);
    std::vector<XYZ> knots(graph.size());
    for (size_t i = 0; i < uv.size(); ++i) {
        auto [x, y] = uv[i];
        double r = std::sqrt(x * x + y * y + 1.0);
        knots[i] = {x / r, y / r, -1.0 / r};
    }
    knots[pole] = {0, 0, 1};
    return knots;
}

std::vector<XYZ> initialize(const Graph& graph, std::span<const int> faces) {
    const int n = static_cast<int>(graph.size());
    auto [half, loop] = split_mesh(graph, faces);
    std::vector<UV> uv(n);
    Mask free = half;
    for (int v : loop)
        free[v] = false;
    const int interior = std::count(free.begin(), free.end(), 1);
    if (!interior || interior + loop.size() == graph.size()) return {};
    for (int i = 0; i < n; ++i)
        uv[i].fill(-1.0 + i * (2.0 / (n - 1)));
    double phi = std::acos(1.0 - 2.0 * interior / n);
    double radius = 2.0 * std::sin(phi) / (1.0 + std::cos(phi));
    circle(uv, loop, radius);
    harmonic_disk(uv, graph, free, faces);
    for (int i = 0; i < n; ++i)
        free[i] = !free[i] || std::hypot(uv[i][0], uv[i][1]) > 0.7 * radius;
    if (std::count(free.begin(), free.end(), 0) < 3) return {};
    change_chart(uv, true, false);
    harmonic_disk(uv, graph, free, faces);

    // Four overlapping charts, in the original x+, x-, y+, y- order.
    std::vector<double> radii(n);
    for (int pass = 0; pass < 4; ++pass) {
        change_chart(uv, pass != 0, pass == 0 || pass == 2);
        for (int i = 0; i < n; ++i)
            radii[i] = std::hypot(uv[i][0], uv[i][1]);
        auto cutoff = radii.begin() + static_cast<size_t>(0.9 * n);
        std::nth_element(radii.begin(), cutoff, radii.end());
        for (int i = 0; i < n; ++i)
            free[i] = std::hypot(uv[i][0], uv[i][1]) < *cutoff;
        if (std::count(free.begin(), free.end(), 0) >= 3) harmonic_disk(uv, graph, free, faces);
    }
    std::vector<XYZ> knots(n);
    for (int i = 0; i < n; ++i) {
        auto s = cpgeo::stereographicProjection2_3(uv[i]);
        knots[i] = {s[1], s[2], s[0]};
    }
    return knots;
}

bool orient_sphere(std::vector<XYZ>& knots, std::span<int> faces) {
    double area = 0;
    size_t inward = 0;
    for (size_t i = 0; i < faces.size(); i += 3) {
        const auto& a = knots[faces[i]];
        const auto& b = knots[faces[i + 1]];
        const auto& c = knots[faces[i + 2]];
        double det = cpgeo::sphere_orientation(a.data(), b.data(), c.data());
        if (!std::isfinite(det) || det == 0.0) return false;
        inward += det < 0.0;
        double denominator = 1.0;
        for (int d = 0; d < 3; ++d)
            denominator += a[d] * b[d] + b[d] * c[d] + c[d] * a[d];
        area += 2.0 * std::atan2(std::abs(det), denominator);
    }
    // Only commit an orientation after verifying a single covering of the sphere.
    if (!(std::abs(area - 4.0 * std::numbers::pi) < 1e-8)) return false;
    if (inward == faces.size() / 3) {
        // The chart reversed the entire mesh. Reflect the knots to preserve the
        // physical face winding and the handedness of the surface derivatives.
        for (auto& p : knots) p[0] = -p[0];
    } else if (inward != 0) {
        // Retain support for input meshes with inconsistent face winding.
        for (size_t i = 0; i < faces.size(); i += 3) {
            if (cpgeo::sphere_orientation(knots[faces[i]].data(), knots[faces[i + 1]].data(),
                                          knots[faces[i + 2]].data()) < 0.0)
                std::swap(faces[i + 1], faces[i + 2]);
        }
    }
    return true;
}

} // namespace

extern "C" CPGEO_API int cpgeo_initialize_knots(int* faces, int num_faces, int num_vertices,
                                                double* out_knots) {
    if (!faces || !out_knots || num_faces < 4 || num_vertices < 4) return -1;
    try {
        std::span<int> triangles(faces, 3 * static_cast<size_t>(num_faces));
        auto graph = mesh_graph(triangles, num_vertices);
        auto knots = initialize(graph, triangles);
        if (knots.empty() || !orient_sphere(knots, triangles)) {
            knots = polar_disk(graph, triangles);
            if (!orient_sphere(knots, triangles)) return -2;
        }
        for (size_t i = 0; i < knots.size(); ++i)
            std::copy_n(knots[i].begin(), 3, out_knots + 3 * i);
        return 0;
    } catch (const std::invalid_argument&) {
        return -1;
    } catch (...) {
        return -2;
    }
}
