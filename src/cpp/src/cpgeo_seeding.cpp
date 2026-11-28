#include "csr_cg.h"
#include "cpgeo_seeding.h"
#include <algorithm>
#include <numeric>
#include <condition_variable>
#include <functional>
#include <deque>
#include <set>
#include <mutex>
#include <thread>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <stdexcept>

namespace cpgeo {

    static const int order = 4;

    static void export_mesh(std::span<const double> vertices, std::span<const int> faces, const std::string& filename) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        // Write vertices
        for (size_t i = 0; i < vertices.size() / 3; ++i) {
            ofs << "v " << vertices[i * 3 + 0] << " " << vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << "\n";
        }

        // Write faces (OBJ format is 1-indexed)
        for (size_t i = 0; i < faces.size() / 3; ++i) {
            ofs << "f " << (faces[i * 3 + 0] + 1) << " " << (faces[i * 3 + 1] + 1) << " " << (faces[i * 3 + 2] + 1) << "\n";
        }

        ofs.close();
    
    }

static bool can_collapse(int a, int b, const std::array<double, 3>& midpoint,
                         const std::vector<std::vector<int>>& neighbors,
                         const std::vector<std::vector<int>>& incident_faces,
                         const std::vector<double>& vertices, const std::vector<int>& faces) {
    // On a closed triangular manifold the two opposite vertices must be the
    // only common neighbors. Collapsing a tetrahedron would duplicate a face.
    if (neighbors[a].size() == 3 && neighbors[b].size() == 3)
        return false;
    int common = 0;
    for (int v : neighbors[a]) {
        if (std::find(neighbors[b].begin(), neighbors[b].end(), v) != neighbors[b].end())
            ++common;
    }
    if (common != 2)
        return false;

    for (int endpoint : {a, b}) {
        for (int face : incident_faces[endpoint]) {
            const int* tri = faces.data() + 3 * face;
            int i = std::find(tri, tri + 3, endpoint) - tri;
            int v = tri[(i + 1) % 3], w = tri[(i + 2) % 3];
            if (v == a || v == b || w == a || w == b)
                continue;
            const double* p = vertices.data() + 3 * v;
            const double* q = vertices.data() + 3 * w;
            double before = sphere_orientation(vertices.data() + 3 * endpoint, p, q);
            double after = sphere_orientation(midpoint.data(), p, q);
            if (!(before * after > 0.0))
                return false;
        }
    }
    return true;
}

static std::vector<int> insert_delete_points(std::vector<double>& vertices_sphere,
                                             std::span<const double> control_points, SpaceTree& tree,
                                             double seed_size, int fixed_count) {
    if (!(seed_size > 0.0) || !std::isfinite(seed_size))
        throw std::invalid_argument("seed_size must be finite and positive");

    SphereTriangulation triangulator(vertices_sphere);
    triangulator.triangulate();
    std::vector<int> faces(triangulator.size() * 3);
    triangulator.getTriangleIndices(faces);

    if (fixed_count > 0) {
        std::vector<bool> retained(fixed_count);
        for (int v : faces)
            if (v < fixed_count) retained[v] = true;
        if (std::find(retained.begin(), retained.end(), false) != retained.end())
            throw std::runtime_error("Triangulation omitted a fixed control seed");
    }

    // A split batch can grow the mesh by at most half its current vertices.
    // Pure coarsening rounds strictly reduce the vertex count.
    constexpr int max_split_rounds = 10;
    int split_rounds = 0;
    while (!faces.empty()) {
        auto mapped = map_points_batch(vertices_sphere, tree, control_points);
        faces = mesh_optimize_by_edge_flipping(mapped, 3, faces, 100, vertices_sphere);
        if (split_rounds == max_split_rounds)
            break;

        const int n = static_cast<int>(vertices_sphere.size() / 3);
        std::vector<std::vector<int>> neighbors(n), incident_faces(n);
        for (size_t i = 0; i < faces.size(); ++i)
            incident_faces[faces[i]].push_back(i / 3);

        struct Candidate {
            double length;
            int a, b;
        };
        std::vector<Candidate> short_edges, long_edges;
        for (const auto& [edge, count] : extractEdgesWithNumber(faces)) {
            auto [a, b] = edge;
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
            // New seeds are free controls. Leave edges between two anchors intact.
            if (count != 2 || (a < fixed_count && b < fixed_count))
                continue;
            const double* p = mapped.data() + 3 * a;
            const double* q = mapped.data() + 3 * b;
            double length = std::hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
            if (a >= fixed_count && b >= fixed_count && length < 0.5 * seed_size)
                short_edges.push_back({length, a, b});
            if (length > 1.5 * seed_size)
                long_edges.push_back({length, a, b});
        }
        auto by_length = [](const Candidate& x, const Candidate& y) {
            return std::tie(x.length, x.a, x.b) < std::tie(y.length, y.a, y.b);
        };
        std::sort(short_edges.begin(), short_edges.end(), by_length);
        std::sort(long_edges.rbegin(), long_edges.rend(), by_length);

        std::vector<bool> locked(n, false);
        std::vector<int> collapse_to(n, -1);
        std::unordered_map<std::pair<int, int>, int, EdgeHash> split_at;
        bool changed = false;
        auto apply_batch = [&](const std::vector<Candidate>& candidates, bool collapse) {
            for (const auto& [length, a, b] : candidates) {
                if (locked[a] || locked[b])
                    continue;
                std::array<double, 3> mid;
                for (int d = 0; d < 3; ++d)
                    mid[d] = vertices_sphere[3 * a + d] + vertices_sphere[3 * b + d];
                double norm = std::hypot(mid[0], mid[1], mid[2]);
                if (!(norm > 1e-12))
                    continue;
                for (double& x : mid)
                    x /= norm;

                if (collapse) {
                    if (!can_collapse(a, b, mid, neighbors, incident_faces, vertices_sphere, faces))
                        continue;
                    collapse_to[b] = a;
                    std::copy(mid.begin(), mid.end(), vertices_sphere.begin() + 3 * a);
                    for (int v : neighbors[a])
                        locked[v] = true;
                    for (int v : neighbors[b])
                        locked[v] = true;
                } else {
                    split_at[{a, b}] = static_cast<int>(vertices_sphere.size() / 3);
                    vertices_sphere.insert(vertices_sphere.end(), mid.begin(), mid.end());
                }
                locked[a] = locked[b] = true;
                changed = true;
            }
        };
        apply_batch(short_edges, true);
        apply_batch(long_edges, false);
        if (!changed)
            break;
        if (!split_at.empty())
            ++split_rounds;

        // A face can contain at most one selected split. Both adjacent faces
        // use the same midpoint and keep their original winding.
        std::vector<int> next_faces;
        next_faces.reserve(faces.size() + 6 * split_at.size());
        for (size_t i = 0; i < faces.size(); i += 3) {
            std::array<int, 3> tri{faces[i], faces[i + 1], faces[i + 2]};
            bool split = false;
            for (int j = 0; j < 3; ++j) {
                int a = tri[j], b = tri[(j + 1) % 3], c = tri[(j + 2) % 3];
                auto it = split_at.find(std::minmax(a, b));
                if (it == split_at.end())
                    continue;
                next_faces.insert(next_faces.end(), {a, it->second, c, it->second, b, c});
                split = true;
                break;
            }
            if (split)
                continue;
            for (int& v : tri)
                if (collapse_to[v] >= 0)
                    v = collapse_to[v];
            if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0])
                next_faces.insert(next_faces.end(), tri.begin(), tri.end());
        }

        std::vector<int> new_index(vertices_sphere.size() / 3);
        int kept = 0;
        for (int v = 0; v < static_cast<int>(new_index.size()); ++v) {
            if (v < n && collapse_to[v] >= 0)
                continue;
            new_index[v] = kept;
            for (int d = 0; d < 3; ++d)
                vertices_sphere[3 * kept + d] = vertices_sphere[3 * v + d];
            ++kept;
        }
        vertices_sphere.resize(3 * kept);
        for (int& v : next_faces)
            v = new_index[v];
        faces = std::move(next_faces);
    }
    return faces;
}

struct SmoothingSystem {
    std::vector<double> gradient;
    std::vector<int> row_offsets;
    std::vector<int> columns;
    std::vector<double> values;
};

static SmoothingSystem assemble_uv_system(
    std::span<const double> r, std::span<const double> jacobian,
    std::span<const double> second_derivative, std::span<const int> edges,
    int active_count)
{
    auto [loss, spatial_gradient, indices, values] = closure_edge_length_derivative2(r, 3, edges, order);
    SmoothingSystem system;
    system.gradient.resize(2 * active_count);
    using Block = std::array<double, 4>;
    using BlockKey = std::pair<int, int>;
    using BlockMap = std::unordered_map<BlockKey, Block, EdgeHash>;
    const int threads = std::min(omp_get_max_threads(), std::max(1, active_count / 128));
    std::vector<BlockMap> thread_blocks(threads);

    #pragma omp parallel if(threads > 1) num_threads(threads)
    {
        auto& blocks = thread_blocks[omp_get_thread_num()];
        // Each pair of active vertices contributes one 2x2 UV block, J_i^T H_ij J_j.
        #pragma omp for schedule(static) nowait
        for (int k = 0; k < static_cast<int>(values.size()); ++k) {
            int i = indices[4 * k], j = indices[4 * k + 2];
            if (i >= active_count || j >= active_count) continue;
            auto& block = blocks[{i, j}];
            const double* ji = jacobian.data() + 6 * i + indices[4 * k + 1];
            const double* jj = jacobian.data() + 6 * j + indices[4 * k + 3];
            for (int u = 0; u < 2; ++u) {
                for (int v = 0; v < 2; ++v) {
                    block[2 * u + v] += values[k] * ji[3 * u] * jj[3 * v];
                }
            }
        }

        #pragma omp for
        for (int i = 0; i < active_count; ++i) {
            auto& diagonal = blocks[{i, i}];
            for (int d = 0; d < 3; ++d) {
                double g = spatial_gradient[3 * i + d];
                for (int u = 0; u < 2; ++u) {
                    system.gradient[2 * i + u] += g * jacobian[6 * i + 3 * u + d];
                    for (int v = 0; v < 2; ++v) {
                        int entry = 2 * u + v;
                        diagonal[entry] += g * second_derivative[12 * i + 3 * entry + d];
                    }
                }
            }
        }
    }

    BlockMap merged;
    for (const auto& blocks : thread_blocks) {
        for (const auto& [key, block] : blocks) {
            auto& sum = merged[key];
            for (int k = 0; k < 4; ++k) sum[k] += block[k];
        }
    }

    // Sort vertex pairs once and emit both scalar CSR rows in column order.
    std::vector<std::pair<BlockKey, Block>> entries(merged.begin(), merged.end());
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    system.columns.reserve(4 * entries.size());
    system.values.reserve(4 * entries.size());
    system.row_offsets.reserve(2 * active_count + 1);
    system.row_offsets.push_back(0);
    size_t end = 0;
    for (int row = 0; row < active_count; ++row) {
        size_t begin = end;
        while (end < entries.size() && entries[end].first.first == row) ++end;
        for (int u = 0; u < 2; ++u) {
            for (size_t k = begin; k < end; ++k) {
                const auto& [key, block] = entries[k];
                for (int v = 0; v < 2; ++v) {
                    double value = block[2 * u + v];
                    if (std::abs(value) < 1e-8) continue;
                    system.columns.push_back(2 * key.second + v);
                    system.values.push_back(value);
                }
            }
            system.row_offsets.push_back(static_cast<int>(system.values.size()));
        }
    }
    return system;
}

// Partition the six cube charts by rank. Two successive median cuts (u, v)
// form four children. Stop as soon as a child fits the free-point budget.
static std::vector<std::vector<int>> smoothing_cores(std::span<const double> vertices, int fixed_count) {
    constexpr int target = 64;
    std::array<std::vector<int>, 6> roots;
    for (int i = fixed_count; i < static_cast<int>(vertices.size() / 3); ++i) {
        const double* p = vertices.data() + 3*i;
        int axis = 0;
        for (int d = 1; d < 3; ++d)
            if (std::abs(p[d]) > std::abs(p[axis])) axis = d;
        roots[2*axis + (p[axis] < 0)].push_back(i);
    }
    std::vector<std::vector<int>> cores;
    for (int face = 0; face < 6; ++face) {
        auto split = [&](auto&& self, std::span<int> ids, int direction) -> void {
            if (ids.empty()) return;
            if (ids.size() <= target) {
                cores.emplace_back(ids.begin(), ids.end());
                return;
            }
            int normal = face / 2, axis = (normal + direction + 1) % 3;
            auto coordinate = [&](int id) {
                return vertices[3*id+axis] / std::abs(vertices[3*id+normal]);
            };
            size_t half = ids.size() / 2;
            std::nth_element(ids.begin(), ids.begin()+half, ids.end(), [&](int a, int b) {
                return std::pair{coordinate(a), a} < std::pair{coordinate(b), b};
            });
            self(self, ids.first(half), 1-direction);
            self(self, ids.subspan(half), 1-direction);
        };
        split(split, roots[face], 0);
    }
    return cores;
}

struct PatchFace {
    std::array<int, 3> local;
    int mesh_index;
};

struct SmoothingPatch {
    std::vector<int> ids, edges;
    std::vector<PatchFace> faces;
    int active_count;
    double volume_budget6 = 0.0;
};

// Movable vertices come first. All triangles and springs incident on them
// belong to the local problem, including springs anchored across the boundary.
static SmoothingPatch make_patch(const std::vector<int>& core, int fixed_count,
                                 const std::vector<std::vector<int>>& neighbors,
                                 const std::vector<std::vector<int>>& incident,
                                 std::span<const int> triangles, std::vector<int>& index,
                                 std::vector<int>& face_uses) {
    SmoothingPatch patch;
    auto add = [&](int v) {
        if (index[v] < 0) {
            index[v] = static_cast<int>(patch.ids.size());
            patch.ids.push_back(v);
        }
    };
    for (int v : core) add(v);
    size_t begin = 0;
    for (int ring = 0; ring < 2; ++ring) {
        const size_t end = patch.ids.size();
        for (size_t i = begin; i < end; ++i)
            for (int v : neighbors[patch.ids[i]])
                if (v >= fixed_count) add(v);
        begin = end;
    }
    patch.active_count = static_cast<int>(patch.ids.size());
    for (int i = 0; i < patch.active_count; ++i) {
        int v = patch.ids[i];
        for (int w : neighbors[v]) {
            add(w);
            if (index[w] > i)
                patch.edges.insert(patch.edges.end(), {i, index[w]});
        }
    }
    for (int i = 0; i < patch.active_count; ++i) {
        for (int f : incident[patch.ids[i]]) {
            int a = index[triangles[3*f]], b = index[triangles[3*f+1]], c = index[triangles[3*f+2]];
            if (std::min({a,b,c}) == i) {
                patch.faces.push_back({{a,b,c}, f});
                ++face_uses[f];
            }
        }
    }
    for (int v : patch.ids) index[v] = -1;
    return patch;
}

using Point3 = std::array<double, 3>;

static double dot(const Point3& a, const Point3& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static Point3 cross(const Point3& a, const Point3& b) {
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}

// Maximize the smallest stereographic denominator 1 - pole.dot(point) over
// the movable vertices. Keep the same chart for all Newton steps in the patch.
static SphereFrame patch_frame(std::span<const double> points) {
    Point3 pole{}, opposite_center{};
    for (size_t i = 0; i < points.size(); ++i) opposite_center[i % 3] -= points[i];
    double best = -1.0;
    auto consider = [&](Point3 candidate) {
        const double length = std::sqrt(dot(candidate, candidate));
        if (length < 1e-12) return;
        for (double& x : candidate) x /= length;
        double margin = 2.0;
        for (size_t i = 0; i < points.size(); i += 3) {
            const double alignment = std::inner_product(candidate.begin(), candidate.end(), points.data()+i, 0.0);
            margin = std::min(margin, 1.0 - alignment);
        }
        if (margin > best) {
            best = margin;
            pole = candidate;
        }
    };
    consider(opposite_center);
    for (int d = 0; d < 3; ++d) {
        Point3 axis{};
        axis[d] = 1.0;
        consider(axis);
        axis[d] = -1.0;
        consider(axis);
    }
    // Coarse patches can span both ends of all three coordinate axes.
    consider({1.0, 1.0, 1.0});
    Point3 axis{};
    const auto least_aligned = std::min_element(pole.begin(), pole.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    axis[least_aligned - pole.begin()] = 1.0;
    auto tangent = cross(axis, pole);
    double length = std::sqrt(dot(tangent, tangent));
    for (double& x : tangent) x /= length;
    return {{tangent, cross(pole, tangent), pole}};
}

static std::vector<double> newton_direction(SmoothingSystem system) {
    const int n = static_cast<int>(system.gradient.size());
    std::vector<double> scale(n, 1.0), rhs(n);
    for (int i = 0; i < n; ++i) {
        for (int k = system.row_offsets[i]; k < system.row_offsets[i+1]; ++k)
            if (system.columns[k] == i && system.values[k] != 0.0)
                scale[i] = 1.0 / std::sqrt(std::abs(system.values[k]));
        rhs[i] = -system.gradient[i] * scale[i];
    }
    for (int i = 0; i < n; ++i)
        for (int k = system.row_offsets[i]; k < system.row_offsets[i+1]; ++k)
            system.values[k] *= scale[i] * scale[system.columns[k]];

    std::vector<double> step;
    conjugate_gradient(system.row_offsets, system.columns, system.values, rhs, step, 10000, 1e-6);
    for (int i = 0; i < n; ++i) {
        step[i] *= scale[i];
        if (!std::isfinite(step[i])) step[i] = 0.0;
    }
    if (std::inner_product(step.begin(), step.end(), system.gradient.begin(), 0.0) > 0.0)
        for (double& x : step) x = -x;
    return step;
}

static double face_determinant(std::span<const double> vertices, const int* tri) {
    return sphere_orientation(vertices.data()+3*tri[0], vertices.data()+3*tri[1], vertices.data()+3*tri[2]);
}

enum class PatchStop { small_step, line_search_failed, iteration_limit };

struct PatchResult {
    PatchStop reason;
    int accepted_steps;
};

static PatchResult smooth_patch(const SmoothingPatch& patch, std::vector<double>& local,
                                SpaceTree& tree, std::span<const double> controls) {
    constexpr int max_newton_steps = 500;
    const int n = patch.active_count;
    const auto active = [&](const std::vector<double>& points) {
        return std::span<const double>(points).first(3*n);
    };
    const auto frame = patch_frame(active(local));
    auto mapped = map_points_batch(local, tree, controls);
    auto trial = local;
    auto trial_mapped = mapped;
    std::vector<std::array<double,2>> uv(n);
    // Only faces incident on movable vertices contribute to this patch's change.
    // The common apex is the parameter sphere origin; the factor 1/6 cancels.
    std::vector<double> initial_faces(patch.faces.size());
    for (size_t f = 0; f < initial_faces.size(); ++f)
        initial_faces[f] = face_determinant(local, patch.faces[f].local.data());
    const double initial_volume6 = std::accumulate(initial_faces.begin(), initial_faces.end(), 0.0);
    const double volume_tolerance6 = std::min(patch.volume_budget6, 0.05 * std::abs(initial_volume6));
    double loss = closure_edge_length_derivative0(mapped, 3, patch.edges, order);
    for (int iter = 0; iter < max_newton_steps; ++iter) {
        auto [r, jacobian, hessian] = map_points_batch_derivative2(active(local), tree, controls, frame);
        // Fixed neighbors contribute springs but have no UV unknowns.
        std::copy(r.begin(), r.end(), mapped.begin());
        auto step = newton_direction(assemble_uv_system(mapped, jacobian, hessian, patch.edges, n));
        const double norm2 = std::inner_product(step.begin(), step.end(), step.begin(), 0.0);
        if (norm2 < 0.1 * 0.1) return {PatchStop::small_step, iter};

        for (int i = 0; i < n; ++i)
            uv[i] = stereographicProjection3_2(frame.local(std::span<const double,3>(local.data()+3*i, 3)));

        bool accepted = false;
        double alpha = 1.0;
        for (int bt = 0; bt < 30; ++bt, alpha *= 0.5) {
            for (int i = 0; i < n; ++i) {
                std::array<double,2> q{uv[i][0]+alpha*step[2*i], uv[i][1]+alpha*step[2*i+1]};
                auto p = frame.world(stereographicProjection2_3(q));
                double length = std::sqrt(dot(p, p));
                for (int d = 0; d < 3; ++d) trial[3*i+d] = p[d]/length;
            }
            bool oriented = true;
            double trial_volume6 = 0.0;
            for (size_t f = 0; f < initial_faces.size(); ++f) {
                double after = face_determinant(trial, patch.faces[f].local.data());
                if (!(initial_faces[f]*after > 0.0)) {
                    oriented = false;
                    break;
                }
                trial_volume6 += after;
            }
            if (!oriented || !(std::abs(trial_volume6 - initial_volume6) < volume_tolerance6))
                continue;
            auto moved = map_points_batch(active(trial), tree, controls);
            std::copy(moved.begin(), moved.end(), trial_mapped.begin());
            const double next_loss = closure_edge_length_derivative0(trial_mapped, 3, patch.edges, order);
            if (!(next_loss < loss)) continue;

            loss = next_loss;
            local.swap(trial);
            mapped.swap(trial_mapped);
            accepted = true;
            break;
        }
        if (!accepted) return {PatchStop::line_search_failed, iter};
    }
    return {PatchStop::iteration_limit, max_newton_steps};
}

struct PatchOrder {
    std::vector<std::vector<int>> successors;
    std::vector<int> pending;
    int widest_level = 1;
};

// Preserve the serial order of every read/write and write/write conflict.
// Two patches may share any number of read-only boundary vertices.
static PatchOrder patch_order(const std::vector<SmoothingPatch>& patches, size_t vertex_count) {
    const int count = static_cast<int>(patches.size());
    PatchOrder order;
    order.successors.resize(count);
    order.pending.resize(count);
    std::vector<int> writer(vertex_count, -1), seen(count, -1), level(count), width(count);
    std::vector<std::vector<int>> readers(vertex_count);
    for (int i = 0; i < count; ++i) {
        auto depend_on = [&](int earlier) {
            if (earlier < 0 || seen[earlier] == i) return;
            seen[earlier] = i;
            order.successors[earlier].push_back(i);
            ++order.pending[i];
            level[i] = std::max(level[i], level[earlier]+1);
        };
        const auto& patch = patches[i];
        for (size_t j = 0; j < patch.ids.size(); ++j) {
            const int v = patch.ids[j];
            depend_on(writer[v]);
            if (j < static_cast<size_t>(patch.active_count)) {
                for (int reader : readers[v]) depend_on(reader);
                readers[v].clear();
                writer[v] = i;
            } else {
                readers[v].push_back(i);
            }
        }
        order.widest_level = std::max(order.widest_level, ++width[level[i]]);
    }
    return order;
}

struct PatchCandidate {
    std::vector<double> local;
    PatchResult result{};
};

static PatchCandidate solve_patch(const SmoothingPatch& patch, std::span<const double> vertices,
                                  SpaceTree& tree, std::span<const double> controls) {
    PatchCandidate candidate;
    candidate.local.resize(3*patch.ids.size());
    for (size_t i = 0; i < patch.ids.size(); ++i)
        std::copy_n(vertices.data()+3*patch.ids[i], 3, candidate.local.data()+3*i);
    candidate.result = smooth_patch(patch, candidate.local, tree, controls);
    return candidate;
}

// The producer owns geometry commits. Workers publish only private candidates.
class PatchWorkers {
public:
    using Solve = std::function<PatchCandidate(int)>;

    PatchWorkers(int threads, Solve solve) : solve_(std::move(solve)) {
        try {
            for (int i = 0; i < threads; ++i)
                workers_.emplace_back([this] { consume(); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~PatchWorkers() { shutdown(); }

    void submit(int index) {
        {
            std::lock_guard lock(mutex_);
            ready_.insert(index);
        }
        available_.notify_one();
    }

    std::pair<int, PatchCandidate> take() {
        std::unique_lock lock(mutex_);
        completed_.wait(lock, [&] { return error_ || !finished_.empty(); });
        if (error_) std::rethrow_exception(error_);
        auto finished = std::move(finished_.front());
        finished_.pop_front();
        return finished;
    }

private:
    void consume() {
        // Use the worker budget for independent patches, not nested OpenMP teams.
        omp_set_num_threads(1);
        try {
            while (true) {
                std::unique_lock lock(mutex_);
                available_.wait(lock, [&] { return stopping_ || !ready_.empty(); });
                if (stopping_) return;
                const int index = *ready_.begin();
                ready_.erase(ready_.begin());
                lock.unlock();
                auto candidate = solve_(index);
                lock.lock();
                finished_.emplace_back(index, std::move(candidate));
                lock.unlock();
                completed_.notify_one();
            }
        } catch (...) {
            std::lock_guard lock(mutex_);
            error_ = std::current_exception();
            completed_.notify_one();
        }
    }

    void shutdown() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        available_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    Solve solve_;
    std::mutex mutex_;
    std::condition_variable available_, completed_;
    std::set<int> ready_;
    std::deque<std::pair<int, PatchCandidate>> finished_;
    std::exception_ptr error_;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

static std::vector<PatchResult> run_patch_sweep(const std::vector<SmoothingPatch>& patches,
                                                std::vector<double>& vertices, SpaceTree& tree,
                                                std::span<const double> controls) {
    const int count = static_cast<int>(patches.size());
    std::vector<PatchResult> results(count);
    auto solve = [&](int i) { return solve_patch(patches[i], vertices, tree, controls); };
    auto publish = [&](int i, const PatchCandidate& candidate) {
        results[i] = candidate.result;
        if (candidate.result.accepted_steps == 0) return;
        for (int j = 0; j < patches[i].active_count; ++j)
            std::copy_n(candidate.local.data()+3*j, 3, vertices.data()+3*patches[i].ids[j]);
    };
    if (omp_get_max_threads() == 1) {
        for (int i = 0; i < count; ++i) publish(i, solve(i));
        return results;
    }

    auto order = patch_order(patches, vertices.size()/3);
    PatchWorkers pool(std::min(omp_get_max_threads(), order.widest_level), solve);
    for (int i = 0; i < count; ++i)
        if (order.pending[i] == 0) pool.submit(i);

    // Once all conflicting predecessors have published, both the coordinates
    // and the volume baseline for the successor match the serial sweep.
    for (int finished = 0; finished < count; ++finished) {
        auto [i, candidate] = pool.take();
        publish(i, candidate);
        for (int successor : order.successors[i])
            if (--order.pending[successor] == 0) pool.submit(successor);
    }
    return results;
}

void vertice_smoothing(std::vector<double>& vertices_sphere, std::span<int> faces,
                       std::span<const double> control_points, SpaceTree& tree) {
    vertice_smoothing(vertices_sphere, faces, control_points, tree, 0);
}

void vertice_smoothing(std::vector<double>& vertices, std::span<int> faces,
                       std::span<const double> controls, SpaceTree& tree, int fixed_count) {
    const auto cores = smoothing_cores(vertices, fixed_count);
    if (cores.empty()) return;
    std::vector<std::vector<int>> neighbors(vertices.size()/3), incident(vertices.size()/3);
    for (const auto& [edge, count] : extractEdgesWithNumber(faces)) {
        auto [a,b] = edge;
        neighbors[a].push_back(b);
        neighbors[b].push_back(a);
    }
    for (size_t i = 0; i < faces.size(); i += 3) {
        for (int d = 0; d < 3; ++d) incident[faces[i+d]].push_back(static_cast<int>(i/3));
    }
    std::vector<int> index(vertices.size()/3, -1);
    std::vector<int> face_uses(faces.size()/3);
    std::vector<SmoothingPatch> patches;
    patches.reserve(cores.size());
    for (const auto& core : cores)
        patches.push_back(make_patch(core, fixed_count, neighbors, incident, faces, index, face_uses));

    // Allocate each movable face's allowance exactly once, using its signed
    // tetrahedral contribution at sweep entry. The parameter triangles have
    // consistent winding, so their magnitudes add to the mesh volume (times 6).
    std::vector<double> face_share6(face_uses.size());
    for (size_t f = 0; f < face_uses.size(); ++f) {
        if (face_uses[f] > 0)
            face_share6[f] = 0.05 * std::abs(face_determinant(vertices, faces.data()+3*f)) / face_uses[f];
    }
    for (auto& patch : patches)
        for (const auto& face : patch.faces)
            patch.volume_budget6 += face_share6[face.mesh_index];

    size_t converged = 0, stalled = 0, steps = 0;
    for (const auto& result : run_patch_sweep(patches, vertices, tree, controls)) {
        converged += result.reason == PatchStop::small_step;
        stalled += result.reason == PatchStop::line_search_failed;
        steps += result.accepted_steps;
    }
    std::cout << "  - Smoothing sweep: " << converged << "/" << cores.size()
              << " patches reached the step tolerance, " << stalled << " line searches stalled, "
              << cores.size()-converged-stalled << " at iteration limit, " << steps
              << " accepted Newton steps." << std::endl;
}

std::tuple<std::vector<double>, std::vector<int>> uniformlyMesh(
    std::span<double> init_vertices_sphere, std::span<const double> control_points,
    SpaceTree& tree, double seed_size, int max_iterations) {
    return uniformlyMesh(init_vertices_sphere, control_points, tree, seed_size, max_iterations, 0);
}

std::tuple<std::vector<double>, std::vector<int>> uniformlyMesh(
    std::span<double> init_vertices_sphere, 
    std::span<const double> control_points,
    SpaceTree& tree,
    double seed_size,
    int max_iterations, int fixed_count
){
    if (fixed_count < 0 || static_cast<size_t>(fixed_count) > init_vertices_sphere.size() / 3)
        throw std::invalid_argument("Invalid fixed seed count");

    std::vector<double> vertices_sphere(init_vertices_sphere.begin(), init_vertices_sphere.end());

    // insert/delete points
    auto faces = insert_delete_points(vertices_sphere, control_points, tree, seed_size, fixed_count);

    for (int round = 0; round < max_iterations; ++round) {
        std::cout << "  - Mesh uniforming round " << round+1 << "/" << max_iterations << std::endl;
        vertice_smoothing(vertices_sphere, faces, control_points, tree, fixed_count);
        const auto mapped = map_points_batch(vertices_sphere, tree, control_points);
        faces = mesh_optimize_by_edge_flipping(mapped, 3, faces, 10000, vertices_sphere);
    }
    return {vertices_sphere, faces};
}

} // namespace cpgeo
