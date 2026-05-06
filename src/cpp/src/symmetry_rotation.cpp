#include "symmetry_rotation.h"
#include "mesh_utils.h"
#include "mesh_edge_flip.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <set>
#include <map>
#include <queue>
#include <cassert>
#include <cstring>
#include <functional>

namespace cpgeo {

// ======================================================================
// rot_z — rotate points around Z axis
// ======================================================================
std::vector<double> rot_z(std::span<const double> points, double angle) {
    int64_t n = static_cast<int64_t>(points.size()) / 3;
    double c = std::cos(angle);
    double s = std::sin(angle);
    std::vector<double> out(points.size());
    for (int64_t i = 0; i < n; ++i) {
        double x = points[i * 3 + 0];
        double y = points[i * 3 + 1];
        double z = points[i * 3 + 2];
        out[i * 3 + 0] = c * x - s * y;
        out[i * 3 + 1] = s * x + c * y;
        out[i * 3 + 2] = z;
    }
    return out;
}

// ======================================================================
// Internal helpers for zipper_stitch
// ======================================================================
namespace {

double vec_norm3(const double* p) {
    return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
}

void vec_sub3(const double* a, const double* b, double* out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

void vec_cross3(const double* a, const double* b, double* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

double vec_dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double min_angle_of_triangle(const double* p0, const double* p1, const double* p2) {
    double e01[3], e12[3], e20[3];
    vec_sub3(p1, p2, e01);
    vec_sub3(p0, p2, e12);
    vec_sub3(p0, p1, e20);
    double a = vec_norm3(e01);
    double b = vec_norm3(e12);
    double c = vec_norm3(e20);
    double ang1 = std::acos(std::clamp((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0));
    double ang2 = std::acos(std::clamp((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0));
    double ang3 = std::acos(std::clamp((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0));
    return std::min({ang1, ang2, ang3});
}

void tri_normal(const double* p0, const double* p1, const double* p2, double* n) {
    double v1[3], v2[3];
    vec_sub3(p1, p0, v1);
    vec_sub3(p2, p0, v2);
    vec_cross3(v1, v2, n);
    double ln = vec_norm3(n);
    if (ln > 1e-12) {
        n[0] /= ln; n[1] /= ln; n[2] /= ln;
    } else {
        n[0] = n[1] = n[2] = 0.0;
    }
}

} // anonymous namespace

// ======================================================================
// zipper_stitch
// ======================================================================
std::vector<int64_t> zipper_stitch(
    std::span<const int64_t> right_ids,
    std::span<const double> right_pts,
    std::span<const int64_t> left_ids,
    std::span<const double> left_pts,
    double dihedral_angle_threshold,
    bool debug)
{
    if (right_ids.size() < 2 || left_ids.size() < 2)
        return {};

    // Order chains
    std::vector<int64_t> rid; std::vector<double> rpt;
    std::vector<int64_t> lid; std::vector<double> lpt;
    order_seam_chains(right_ids, right_pts, left_ids, left_pts, rid, rpt, lid, lpt);

    int64_t nr = static_cast<int64_t>(rid.size());
    int64_t nl = static_cast<int64_t>(lid.size());
    double max_dihedral = dihedral_angle_threshold * M_PI / 180.0;

    auto min_angle = [](const double* p) -> double {
        return min_angle_of_triangle(p, p + 3, p + 6);
    };

    auto get_normal = [](const double* p, double* n) {
        tri_normal(p, p + 3, p + 6, n);
    };

    auto dihedral_ok = [&](const double* prev_n, const double* curr_n) -> bool {
        if (!prev_n || vec_norm3(prev_n) < 1e-12 || vec_norm3(curr_n) < 1e-12)
            return true;
        double angle = std::acos(std::clamp(vec_dot3(prev_n, curr_n), -1.0, 1.0));
        return angle < max_dihedral;
    };

    // ---- Greedy baseline ----
    std::vector<bool> greedy_path;
    int64_t gi = 0, gj = 0;
    double g_prev_n[3] = {0, 0, 0};
    double* g_prev_n_ptr = nullptr;
    double g_min_angle = std::numeric_limits<double>::infinity();

    while (gi < nr - 1 || gj < nl - 1) {
        bool c_i = gi < nr - 1;
        bool c_j = gj < nl - 1;
        if (c_i && c_j) {
            double pa[9] = {rpt[gi*3], rpt[gi*3+1], rpt[gi*3+2],
                            rpt[(gi+1)*3], rpt[(gi+1)*3+1], rpt[(gi+1)*3+2],
                            lpt[gj*3], lpt[gj*3+1], lpt[gj*3+2]};
            double pb[9] = {rpt[gi*3], rpt[gi*3+1], rpt[gi*3+2],
                            lpt[(gj+1)*3], lpt[(gj+1)*3+1], lpt[(gj+1)*3+2],
                            lpt[gj*3], lpt[gj*3+1], lpt[gj*3+2]};
            double ma = min_angle(pa);
            double mb = min_angle(pb);
            double na[3], nb[3];
            get_normal(pa, na);
            get_normal(pb, nb);
            bool ok_a = dihedral_ok(g_prev_n_ptr, na);
            bool ok_b = dihedral_ok(g_prev_n_ptr, nb);

            bool pick_a;
            if (ok_a && ok_b) pick_a = ma >= mb;
            else if (ok_a) pick_a = true;
            else if (ok_b) pick_a = false;
            else pick_a = ma >= mb;

            if (pick_a) {
                greedy_path.push_back(true);
                g_prev_n[0] = na[0]; g_prev_n[1] = na[1]; g_prev_n[2] = na[2];
                g_prev_n_ptr = g_prev_n;
                g_min_angle = std::min(g_min_angle, ma);
                gi++;
            } else {
                greedy_path.push_back(false);
                g_prev_n[0] = nb[0]; g_prev_n[1] = nb[1]; g_prev_n[2] = nb[2];
                g_prev_n_ptr = g_prev_n;
                g_min_angle = std::min(g_min_angle, mb);
                gj++;
            }
        } else if (c_i) {
            double pts[9] = {rpt[gi*3], rpt[gi*3+1], rpt[gi*3+2],
                             rpt[(gi+1)*3], rpt[(gi+1)*3+1], rpt[(gi+1)*3+2],
                             lpt[gj*3], lpt[gj*3+1], lpt[gj*3+2]};
            greedy_path.push_back(true);
            get_normal(pts, g_prev_n);
            g_prev_n_ptr = g_prev_n;
            g_min_angle = std::min(g_min_angle, min_angle(pts));
            gi++;
        } else {
            double pts[9] = {rpt[gi*3], rpt[gi*3+1], rpt[gi*3+2],
                             lpt[(gj+1)*3], lpt[(gj+1)*3+1], lpt[(gj+1)*3+2],
                             lpt[gj*3], lpt[gj*3+1], lpt[gj*3+2]};
            greedy_path.push_back(false);
            get_normal(pts, g_prev_n);
            g_prev_n_ptr = g_prev_n;
            g_min_angle = std::min(g_min_angle, min_angle(pts));
            gj++;
        }
    }

    std::vector<bool> best_path = greedy_path;
    double best_min_angle = g_min_angle;

    // ---- DFS + branch-and-bound ----
    constexpr int64_t kMaxNodes = 100000;
    int64_t node_count = 0;

    std::function<void(int64_t, int64_t, std::vector<bool>&, const double*, double)> dfs;
    dfs = [&](int64_t i, int64_t j, std::vector<bool>& path,
              const double* prev_n, double cur_min) {
        node_count++;
        if (node_count > kMaxNodes) return;

        if (i == nr - 1 && j == nl - 1) {
            if (cur_min > best_min_angle) {
                best_min_angle = cur_min;
                best_path = path;
            }
            return;
        }

        bool c_i = i < nr - 1;
        bool c_j = j < nl - 1;

        struct Candidate {
            bool is_i;
            int64_t ni, nj;
            double n[3];
            double ma;
        };
        std::vector<Candidate> candidates;

        if (c_i) {
            double pts[9] = {rpt[i*3], rpt[i*3+1], rpt[i*3+2],
                             rpt[(i+1)*3], rpt[(i+1)*3+1], rpt[(i+1)*3+2],
                             lpt[j*3], lpt[j*3+1], lpt[j*3+2]};
            double n[3];
            get_normal(pts, n);
            if (dihedral_ok(prev_n, n)) {
                candidates.push_back({true, i + 1, j, {n[0], n[1], n[2]}, min_angle(pts)});
            }
        }

        if (c_j) {
            double pts[9] = {rpt[i*3], rpt[i*3+1], rpt[i*3+2],
                             lpt[(j+1)*3], lpt[(j+1)*3+1], lpt[(j+1)*3+2],
                             lpt[j*3], lpt[j*3+1], lpt[j*3+2]};
            double n[3];
            get_normal(pts, n);
            if (dihedral_ok(prev_n, n)) {
                candidates.push_back({false, i, j + 1, {n[0], n[1], n[2]}, min_angle(pts)});
            }
        }

        // Try better min-angle first → find good paths early → tighter pruning
        // Use stable_sort to match Python's stable sort (preserves insertion order for ties)
        std::stable_sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.ma > b.ma; });

        for (const auto& cand : candidates) {
            double new_min = (cur_min <= cand.ma) ? cur_min : cand.ma;
            if (new_min <= best_min_angle) continue;
            path.push_back(cand.is_i);
            dfs(cand.ni, cand.nj, path, cand.n, new_min);
            path.pop_back();
        }
    };

    std::vector<bool> init_path;
    dfs(0, 0, init_path, nullptr, std::numeric_limits<double>::infinity());

    // Reconstruct triangles
    std::vector<int64_t> tris;
    int64_t ci = 0, cj = 0;
    for (bool pick_i : best_path) {
        if (pick_i) {
            tris.insert(tris.end(), {rid[ci], rid[ci + 1], lid[cj]});
            ci++;
        } else {
            tris.insert(tris.end(), {rid[ci], lid[cj + 1], lid[cj]});
            cj++;
        }
    }

    return tris;
}

// ======================================================================
// decide_pole_trim_count
// ======================================================================
int decide_pole_trim_count(
    std::span<const int64_t> left_side,
    std::span<const int64_t> right_side,
    std::span<const double> sector_vertices,
    double mean_edge)
{
    int64_t nl = static_cast<int64_t>(left_side.size());
    int64_t nr = static_cast<int64_t>(right_side.size());
    if (nl < 6 || nr < 6) return 0;

    auto get_pt = [&](int64_t idx) -> const double* {
        return &sector_vertices[idx * 3];
    };

    auto dist = [](const double* a, const double* b) -> double {
        double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    double ds = dist(get_pt(left_side[0]), get_pt(right_side[0]));
    double dn = dist(get_pt(left_side[nl - 1]), get_pt(right_side[nr - 1]));
    double span = std::max(ds, dn) / std::max(mean_edge, 1e-12);

    if (span > 3.0 && nl >= 12 && nr >= 12) return 2;
    if (span > 1.8) return 1;
    return 0;
}

// ======================================================================
// order_seam_chains
// ======================================================================
void order_seam_chains(
    std::span<const int64_t> right_ids_in,
    std::span<const double> right_pts_in,
    std::span<const int64_t> left_ids_in,
    std::span<const double> left_pts_in,
    std::vector<int64_t>& out_rid, std::vector<double>& out_rpt,
    std::vector<int64_t>& out_lid, std::vector<double>& out_lpt)
{
    out_rid.assign(right_ids_in.begin(), right_ids_in.end());
    out_rpt.assign(right_pts_in.begin(), right_pts_in.end());
    out_lid.assign(left_ids_in.begin(), left_ids_in.end());
    out_lpt.assign(left_pts_in.begin(), left_pts_in.end());

    // Ensure south->north direction (increasing z)
    // z[i] is at index i*3 + 2.  Last point's z is at (n-1)*3 + 2 = n*3 - 1 = size() - 1
    if (!out_rpt.empty()) {
        double first_z = out_rpt[2];
        double last_z = out_rpt[out_rpt.size() - 1];
        if (first_z > last_z) {
            std::reverse(out_rid.begin(), out_rid.end());
            int64_t npts = static_cast<int64_t>(out_rpt.size()) / 3;
            for (int64_t i = 0; i < npts / 2; ++i) {
                for (int k = 0; k < 3; ++k)
                    std::swap(out_rpt[i*3 + k], out_rpt[(npts - 1 - i)*3 + k]);
            }
        }
    }
    if (!out_lpt.empty()) {
        double first_z = out_lpt[2];
        double last_z = out_lpt[out_lpt.size() - 1];
        if (first_z > last_z) {
            std::reverse(out_lid.begin(), out_lid.end());
            int64_t npts = static_cast<int64_t>(out_lpt.size()) / 3;
            for (int64_t i = 0; i < npts / 2; ++i) {
                for (int k = 0; k < 3; ++k)
                    std::swap(out_lpt[i*3 + k], out_lpt[(npts - 1 - i)*3 + k]);
            }
        }
    }
}

// ======================================================================
// face_components_by_edges
// ======================================================================
std::vector<std::vector<int64_t>> face_components_by_edges(std::span<const int64_t> faces_flat, int64_t num_faces) {
    if (num_faces == 0) return {};

    std::map<std::pair<int64_t, int64_t>, std::vector<int64_t>> edge_to_faces;
    for (int64_t i = 0; i < num_faces; ++i) {
        int64_t a = faces_flat[i * 3 + 0];
        int64_t b = faces_flat[i * 3 + 1];
        int64_t c = faces_flat[i * 3 + 2];
        edge_to_faces[{std::min(a, b), std::max(a, b)}].push_back(i);
        edge_to_faces[{std::min(b, c), std::max(b, c)}].push_back(i);
        edge_to_faces[{std::min(c, a), std::max(c, a)}].push_back(i);
    }

    std::vector<std::vector<int64_t>> adj(num_faces);
    for (const auto& [edge, flist] : edge_to_faces) {
        if (flist.size() < 2) continue;
        for (size_t i = 0; i < flist.size(); ++i) {
            for (size_t j = i + 1; j < flist.size(); ++j) {
                adj[flist[i]].push_back(flist[j]);
                adj[flist[j]].push_back(flist[i]);
            }
        }
    }

    std::vector<bool> visited(num_faces, false);
    std::vector<std::vector<int64_t>> comps;
    for (int64_t i = 0; i < num_faces; ++i) {
        if (visited[i]) continue;
        std::vector<int64_t> stack = {i};
        visited[i] = true;
        std::vector<int64_t> cur;
        while (!stack.empty()) {
            int64_t u = stack.back(); stack.pop_back();
            cur.push_back(u);
            for (int64_t w : adj[u]) {
                if (!visited[w]) {
                    visited[w] = true;
                    stack.push_back(w);
                }
            }
        }
        comps.push_back(std::move(cur));
    }
    return comps;
}

// ======================================================================
// tri_area_sum
// ======================================================================
double tri_area_sum(std::span<const double> vertices, std::span<const int64_t> faces_flat, int64_t num_faces) {
    double total = 0.0;
    for (int64_t i = 0; i < num_faces; ++i) {
        const double* v0 = &vertices[faces_flat[i * 3 + 0] * 3];
        const double* v1 = &vertices[faces_flat[i * 3 + 1] * 3];
        const double* v2 = &vertices[faces_flat[i * 3 + 2] * 3];
        double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        double cross[3];
        vec_cross3(e1, e2, cross);
        total += 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    }
    return total;
}

// ======================================================================
// mean_edge_length
// ======================================================================
double mean_edge_length(std::span<const double> vertices, std::span<const int64_t> faces_flat, int64_t num_faces) {
    if (num_faces == 0) return 1.0;
    double total = 0.0;
    int64_t count = 0;
    for (int64_t i = 0; i < num_faces; ++i) {
        for (int k = 0; k < 3; ++k) {
            int64_t a = faces_flat[i * 3 + k];
            int64_t b = faces_flat[i * 3 + (k + 1) % 3];
            double dx = vertices[a*3+0] - vertices[b*3+0];
            double dy = vertices[a*3+1] - vertices[b*3+1];
            double dz = vertices[a*3+2] - vertices[b*3+2];
            total += std::sqrt(dx*dx + dy*dy + dz*dz);
            count++;
        }
    }
    return (count > 0) ? total / count : 1.0;
}

// ======================================================================
// axis_triangle_intersection
// ======================================================================
std::pair<bool, std::array<double, 3>> axis_triangle_intersection(
    std::span<const double> vertices, std::span<const int64_t> tri, double eps)
{
    const double* p0 = &vertices[tri[0] * 3];
    const double* p1 = &vertices[tri[1] * 3];
    const double* p2 = &vertices[tri[2] * 3];

    // Solve: x = w0*p0 + w1*p1 + w2*p2, with x_x=0, x_y=0, w0+w1+w2=1
    // Matrix: [p0_x p1_x p2_x] [w0]   [0]
    //         [p0_y p1_y p2_y] [w1] = [0]
    //         [  1    1    1  ] [w2]   [1]
    double m00 = p0[0], m01 = p1[0], m02 = p2[0];
    double m10 = p0[1], m11 = p1[1], m12 = p2[1];
    double m20 = 1.0,  m21 = 1.0,  m22 = 1.0;

    double det = m00 * (m11 * m22 - m12 * m21)
               - m01 * (m10 * m22 - m12 * m20)
               + m02 * (m10 * m21 - m11 * m20);

    if (std::abs(det) <= eps) return {false, {}};

    double inv_det = 1.0 / det;
    double rhs_x = 0.0, rhs_y = 0.0, rhs_z = 1.0;

    double w0 = ((m11 * m22 - m12 * m21) * rhs_x + (m02 * m21 - m01 * m22) * rhs_y + (m01 * m12 - m02 * m11) * rhs_z) * inv_det;
    double w1 = ((m12 * m20 - m10 * m22) * rhs_x + (m00 * m22 - m02 * m20) * rhs_y + (m02 * m10 - m00 * m12) * rhs_z) * inv_det;
    double w2 = ((m10 * m21 - m11 * m20) * rhs_x + (m01 * m20 - m00 * m21) * rhs_y + (m00 * m11 - m01 * m10) * rhs_z) * inv_det;

    if (w0 < -1e-8 || w0 > 1.0 + 1e-8 || w1 < -1e-8 || w1 > 1.0 + 1e-8 || w2 < -1e-8 || w2 > 1.0 + 1e-8)
        return {false, {}};

    std::array<double, 3> q;
    q[0] = w0 * p0[0] + w1 * p1[0] + w2 * p2[0];
    q[1] = w0 * p0[1] + w1 * p1[1] + w2 * p2[1];
    q[2] = w0 * p0[2] + w1 * p1[2] + w2 * p2[2];
    return {true, q};
}

// ======================================================================
// find_axis_pole_triangles
// ======================================================================
PoleTriangles find_axis_pole_triangles(std::span<const double> vertices, std::span<const int64_t> faces_flat, int64_t num_faces) {
    PoleTriangles result;
    struct Hit {
        double z;
        int64_t tri[3];
        double q[3];
    };
    std::vector<Hit> hits;

    for (int64_t i = 0; i < num_faces; ++i) {
        std::array<int64_t, 3> tri = {faces_flat[i*3], faces_flat[i*3+1], faces_flat[i*3+2]};
        auto [ok, q] = axis_triangle_intersection(vertices, tri);
        if (!ok) continue;
        hits.push_back({q[2], {tri[0], tri[1], tri[2]}, {q[0], q[1], q[2]}});
    }

    if (hits.empty()) return result;

    auto south_it = std::min_element(hits.begin(), hits.end(),
        [](const Hit& a, const Hit& b) { return a.z < b.z; });
    auto north_it = std::max_element(hits.begin(), hits.end(),
        [](const Hit& a, const Hit& b) { return a.z < b.z; });

    result.has_south = true;
    result.has_north = true;
    result.south_tri = {south_it->tri[0], south_it->tri[1], south_it->tri[2]};
    result.south_hit = {south_it->q[0], south_it->q[1], south_it->q[2]};
    result.north_tri = {north_it->tri[0], north_it->tri[1], north_it->tri[2]};
    result.north_hit = {north_it->q[0], north_it->q[1], north_it->q[2]};

    return result;
}

// ======================================================================
// virtual_anchor_from_triangle
// ======================================================================
std::array<double, 3> virtual_anchor_from_triangle(
    std::span<const double> tri_points,
    const std::array<double, 3>& axis_hit,
    double phase, double alpha, double target_r)
{
    double r_tri[3];
    for (int i = 0; i < 3; ++i)
        r_tri[i] = std::sqrt(tri_points[i*3]*tri_points[i*3] + tri_points[i*3+1]*tri_points[i*3+1]);
    double r_src = std::max(1e-9, *std::min_element(r_tri, r_tri + 3) * 0.55);
    double r = std::min(std::max(r_src, 0.60 * target_r), 1.30 * target_r);
    double theta = phase + 0.5 * alpha;
    return {r * std::cos(theta), r * std::sin(theta), axis_hit[2]};
}

// ======================================================================
// recover_removed_faces_for_interior_holes
// ======================================================================
std::vector<int64_t> recover_removed_faces_for_interior_holes(
    std::span<const int64_t> f_sector_flat, int64_t n_sector,
    std::span<const int64_t> f_comp_flat, int64_t n_comp,
    const std::unordered_set<int64_t>& interior_boundary_vertices)
{
    if (interior_boundary_vertices.empty()) return {};

    // Build set of component faces for quick lookup
    std::set<std::tuple<int64_t, int64_t, int64_t>> comp_set;
    for (int64_t i = 0; i < n_comp; ++i) {
        int64_t a = f_comp_flat[i*3], b = f_comp_flat[i*3+1], c = f_comp_flat[i*3+2];
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        comp_set.insert({a, b, c});
    }

    // Find removed faces
    std::vector<std::array<int64_t, 3>> removed;
    for (int64_t i = 0; i < n_sector; ++i) {
        int64_t a = f_sector_flat[i*3], b = f_sector_flat[i*3+1], c = f_sector_flat[i*3+2];
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        if (comp_set.find({a, b, c}) == comp_set.end()) {
            removed.push_back({f_sector_flat[i*3], f_sector_flat[i*3+1], f_sector_flat[i*3+2]});
        }
    }

    if (removed.empty()) return {};

    int64_t n = static_cast<int64_t>(removed.size());
    std::vector<bool> used(n, false);
    std::unordered_set<int64_t> frontier = interior_boundary_vertices;

    bool changed = true;
    while (changed) {
        changed = false;
        for (int64_t i = 0; i < n; ++i) {
            if (used[i]) continue;
            int64_t a = removed[i][0], b = removed[i][1], c = removed[i][2];
            if (frontier.count(a) || frontier.count(b) || frontier.count(c)) {
                used[i] = true;
                frontier.insert({a, b, c});
                changed = true;
            }
        }
    }

    std::vector<int64_t> result;
    for (int64_t i = 0; i < n; ++i) {
        if (used[i]) {
            result.push_back(removed[i][0]);
            result.push_back(removed[i][1]);
            result.push_back(removed[i][2]);
        }
    }
    return result;
}

// ======================================================================
// seal_loops_constrained
// ======================================================================
std::vector<int64_t> seal_loops_constrained(
    std::span<const int64_t> faces_flat, int64_t num_faces,
    std::span<const int64_t> loop_vertices,
    std::span<const int64_t> loop_starts,
    int64_t num_loops,
    int64_t max_loop_size)
{
    // When no loops provided, compute boundary loops internally (matching Python's loops=None behavior)
    std::vector<int64_t> computed_loop_verts;
    std::vector<int64_t> computed_loop_starts;
    if (num_loops == 0) {
        auto [lv, ls] = extract_boundary_loops_flat(faces_flat, num_faces);
        if (ls.size() <= 1) {
            // No boundary loops found; return faces unchanged
            return std::vector<int64_t>(faces_flat.begin(), faces_flat.begin() + num_faces * 3);
        }
        computed_loop_verts = std::move(lv);
        computed_loop_starts = std::move(ls);
        loop_vertices = computed_loop_verts;
        loop_starts = computed_loop_starts;
        num_loops = static_cast<int64_t>(computed_loop_starts.size()) - 1;
    }

    // Build edge count map from existing faces
    std::map<std::pair<int64_t, int64_t>, int> edge_count;
    for (int64_t i = 0; i < num_faces; ++i) {
        int64_t a = faces_flat[i*3], b = faces_flat[i*3+1], c = faces_flat[i*3+2];
        for (auto [u, v] : {std::pair{a,b}, {b,c}, {c,a}}) {
            auto key = std::make_pair(std::min(u, v), std::max(u, v));
            edge_count[key]++;
        }
    }

    auto can_add = [&](int64_t a, int64_t b, int64_t c) -> bool {
        for (auto [u, v] : {std::pair{a,b}, {b,c}, {c,a}}) {
            auto key = std::make_pair(std::min(u, v), std::max(u, v));
            if (edge_count[key] >= 2) return false;
        }
        return true;
    };

    auto add_triangle = [&](int64_t a, int64_t b, int64_t c) {
        for (auto [u, v] : {std::pair{a,b}, {b,c}, {c,a}}) {
            edge_count[std::make_pair(std::min(u, v), std::max(u, v))]++;
        }
    };

    std::vector<int64_t> added;
    for (int64_t li = 0; li < num_loops; ++li) {
        int64_t start = loop_starts[li];
        int64_t end = (li + 1 < num_loops) ? loop_starts[li + 1] : static_cast<int64_t>(loop_vertices.size());
        int64_t ring_size = end - start;

        if (ring_size < 3 || ring_size > max_loop_size) continue;

        std::vector<int64_t> poly(loop_vertices.begin() + start, loop_vertices.begin() + end);
        while (poly.size() > 2) {
            int64_t n = static_cast<int64_t>(poly.size());
            bool ear_found = false;
            for (int64_t i = 0; i < n; ++i) {
                int64_t a = poly[(i - 1 + n) % n];
                int64_t b = poly[i];
                int64_t c = poly[(i + 1) % n];
                if (a == b || b == c || a == c) continue;
                if (!can_add(a, b, c)) continue;
                added.push_back(a); added.push_back(b); added.push_back(c);
                add_triangle(a, b, c);
                poly.erase(poly.begin() + i);
                ear_found = true;
                break;
            }
            if (!ear_found) break;
        }
    }

    if (added.empty()) {
        return std::vector<int64_t>(faces_flat.begin(), faces_flat.end());
    }
    std::vector<int64_t> result(faces_flat.begin(), faces_flat.end());
    result.insert(result.end(), added.begin(), added.end());
    return result;
}

// ======================================================================
// split_seam_paths
// ======================================================================
std::pair<std::vector<int64_t>, std::vector<int64_t>> split_seam_paths(
    std::span<const int64_t> loop,
    std::span<const double> vertices,
    int64_t preferred_south_idx,
    int64_t preferred_north_idx,
    const double* preferred_south_point,
    const double* preferred_north_point)
{
    int64_t n = static_cast<int64_t>(loop.size());

    auto pick_from_preferred = [&](int64_t pref_idx, const double* pref_pt) -> int64_t {
        if (pref_idx >= 0) {
            for (int64_t i = 0; i < n; ++i) {
                if (loop[i] == pref_idx) return i;
            }
            if (!pref_pt && pref_idx < static_cast<int64_t>(vertices.size()) / 3) {
                pref_pt = &vertices[pref_idx * 3];
            }
        }
        if (pref_pt) {
            int64_t best = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (int64_t i = 0; i < n; ++i) {
                double dx = vertices[loop[i]*3+0] - pref_pt[0];
                double dy = vertices[loop[i]*3+1] - pref_pt[1];
                double dz = vertices[loop[i]*3+2] - pref_pt[2];
                double d = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (d < best_d) { best_d = d; best = i; }
            }
            return best;
        }
        return -1L;
    };

    int64_t i_s = pick_from_preferred(preferred_south_idx, preferred_south_point);
    int64_t i_n = pick_from_preferred(preferred_north_idx, preferred_north_point);

    std::vector<int64_t> path1, path2;
    if (i_s <= i_n) {
        path1.assign(loop.begin() + i_s, loop.begin() + i_n + 1);
        // path2 = boundary[i_s::-1] + boundary[:i_n-1:-1]
        for (int64_t i = i_s; i >= 0; --i) path2.push_back(loop[i]);
        // Python slice boundary[:i_n-1:-1] with negative step:
        // start defaults to n-1, stop is i_n-1 (exclusive, Python negative wrap)
        int64_t stop_idx = i_n - 1;
        if (stop_idx < 0) stop_idx += n;
        for (int64_t i = n - 1; i > stop_idx; --i) path2.push_back(loop[i]);
    } else {
        path1.insert(path1.end(), loop.begin() + i_s, loop.end());
        path1.insert(path1.end(), loop.begin(), loop.begin() + i_n + 1);
        // Python slice boundary[i_s:i_n-1:-1]: start=i_s, stop=i_n-1 (exclusive, Python negative wrap)
        int64_t stop_idx = i_n - 1;
        if (stop_idx < 0) stop_idx += n;
        for (int64_t i = i_s; i > stop_idx; --i) path2.push_back(loop[i]);
    }

    if (path1.size() < 2 || path2.size() < 2) return {{}, {}};
    return {path1, path2};
}

// ======================================================================
// loop_perimeter
// ======================================================================
double loop_perimeter(std::span<const double> vertices, std::span<const int64_t> loop) {
    int64_t n = static_cast<int64_t>(loop.size());
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t a = loop[i];
        int64_t b = loop[(i + 1) % n];
        double dx = vertices[a*3+0] - vertices[b*3+0];
        double dy = vertices[a*3+1] - vertices[b*3+1];
        double dz = vertices[a*3+2] - vertices[b*3+2];
        total += std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return total;
}

// ======================================================================
// rotate_cycle
// ======================================================================
std::vector<int64_t> rotate_cycle(std::span<const int64_t> ids, int64_t start_idx) {
    if (ids.empty()) return {};
    int64_t k = start_idx % static_cast<int64_t>(ids.size());
    std::vector<int64_t> result;
    result.insert(result.end(), ids.begin() + k, ids.end());
    result.insert(result.end(), ids.begin(), ids.begin() + k);
    return result;
}

// ======================================================================
// zipper_stitch_closed
// ======================================================================
std::vector<int64_t> zipper_stitch_closed(
    std::span<const int64_t> loop_a,
    std::span<const int64_t> loop_b,
    std::span<const double> vertices)
{
    int64_t na = static_cast<int64_t>(loop_a.size());
    int64_t nb = static_cast<int64_t>(loop_b.size());
    if (na < 3 || nb < 3) return {};

    // Find best starting point on b to match a[0]
    const double* pa0 = &vertices[loop_a[0] * 3];
    int64_t start_b = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < nb; ++j) {
        const double* pb = &vertices[loop_b[j] * 3];
        double dx = pa0[0] - pb[0], dy = pa0[1] - pb[1], dz = pa0[2] - pb[2];
        double d = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (d < best_d) { best_d = d; start_b = j; }
    }

    std::vector<int64_t> b_rot = rotate_cycle(loop_b, start_b);

    // Decide direction
    const double* pb0 = &vertices[b_rot[0] * 3];
    const double* pa1 = &vertices[loop_a[1 % na] * 3];
    const double* pb1 = &vertices[b_rot[1 % nb] * 3];
    const double* pb_last = &vertices[b_rot[nb - 1] * 3];

    double d_same = std::sqrt((pa0[0]-pb0[0])*(pa0[0]-pb0[0]) + (pa0[1]-pb0[1])*(pa0[1]-pb0[1]) + (pa0[2]-pb0[2])*(pa0[2]-pb0[2]))
                  + std::sqrt((pa1[0]-pb1[0])*(pa1[0]-pb1[0]) + (pa1[1]-pb1[1])*(pa1[1]-pb1[1]) + (pa1[2]-pb1[2])*(pa1[2]-pb1[2]));
    double d_flip = std::sqrt((pa0[0]-pb0[0])*(pa0[0]-pb0[0]) + (pa0[1]-pb0[1])*(pa0[1]-pb0[1]) + (pa0[2]-pb0[2])*(pa0[2]-pb0[2]))
                  + std::sqrt((pa1[0]-pb_last[0])*(pa1[0]-pb_last[0]) + (pa1[1]-pb_last[1])*(pa1[1]-pb_last[1]) + (pa1[2]-pb_last[2])*(pa1[2]-pb_last[2]));

    if (d_flip < d_same) {
        std::reverse(b_rot.begin(), b_rot.end());
        // Re-find best start
        start_b = 0;
        best_d = std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < nb; ++j) {
            const double* pb = &vertices[b_rot[j] * 3];
            double dx = pa0[0] - pb[0], dy = pa0[1] - pb[1], dz = pa0[2] - pb[2];
            double d = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d < best_d) { best_d = d; start_b = j; }
        }
        b_rot = rotate_cycle(b_rot, start_b);
    }

    // Stitch
    std::vector<int64_t> tris;
    int64_t i = 0, j = 0, step_i = 0, step_j = 0;

    while (step_i < na || step_j < nb) {
        bool can_i = step_i < na;
        bool can_j = step_j < nb;
        int64_t ai = loop_a[i % na];
        int64_t bj = b_rot[j % nb];

        if (can_i && can_j) {
            int64_t ai1 = loop_a[(i + 1) % na];
            int64_t bj1 = b_rot[(j + 1) % nb];
            const double* pai1 = &vertices[ai1 * 3];
            const double* pbj = &vertices[bj * 3];
            const double* pai = &vertices[ai * 3];
            const double* pbj1 = &vertices[bj1 * 3];

            double ci_len = std::sqrt((pai1[0]-pbj[0])*(pai1[0]-pbj[0]) + (pai1[1]-pbj[1])*(pai1[1]-pbj[1]) + (pai1[2]-pbj[2])*(pai1[2]-pbj[2]));
            double cj_len = std::sqrt((pai[0]-pbj1[0])*(pai[0]-pbj1[0]) + (pai[1]-pbj1[1])*(pai[1]-pbj1[1]) + (pai[2]-pbj1[2])*(pai[2]-pbj1[2]));

            if (ci_len <= cj_len) {
                tris.insert(tris.end(), {ai, ai1, bj});
                i++; step_i++;
            } else {
                tris.insert(tris.end(), {ai, bj1, bj});
                j++; step_j++;
            }
        } else if (can_i) {
            int64_t ai1 = loop_a[(i + 1) % na];
            tris.insert(tris.end(), {ai, ai1, bj});
            i++; step_i++;
        } else {
            int64_t bj1 = b_rot[(j + 1) % nb];
            tris.insert(tris.end(), {ai, bj1, bj});
            j++; step_j++;
        }
    }

    return tris;
}

// ======================================================================
// sample_loop_ids
// ======================================================================
std::vector<int64_t> sample_loop_ids(std::span<const int64_t> loop, int periods) {
    int64_t n = static_cast<int64_t>(loop.size());
    if (periods <= 0 || n == 0) return {};
    if (n == periods) return std::vector<int64_t>(loop.begin(), loop.end());
    std::vector<int64_t> result(periods);
    for (int p = 0; p < periods; ++p) {
        int64_t idx = static_cast<int64_t>(std::floor(static_cast<double>(p) * n / periods));
        idx = std::clamp(idx, int64_t(0), n - 1);
        result[p] = loop[idx];
    }
    return result;
}

// ======================================================================
// build_polar_candidate_ring
// ======================================================================
std::pair<std::vector<double>, bool> build_polar_candidate_ring(
    std::span<const double> vertices,
    std::span<const int64_t> loop,
    int periods,
    double mean_edge,
    double far_factor)
{
    auto ids = sample_loop_ids(loop, periods);
    if (static_cast<int64_t>(ids.size()) != periods) return {{}, false};

    std::vector<double> cand(periods * 3);
    bool pulled = false;
    double target_r = std::max(mean_edge, 1e-9);

    for (int i = 0; i < periods; ++i) {
        const double* v = &vertices[ids[i] * 3];
        cand[i*3+0] = v[0];
        cand[i*3+1] = v[1];
        cand[i*3+2] = v[2];
    }

    for (int i = 0; i < periods; ++i) {
        double r = std::sqrt(cand[i*3]*cand[i*3] + cand[i*3+1]*cand[i*3+1]);
        if (r > far_factor * target_r) {
            double theta = std::atan2(cand[i*3+1], cand[i*3+0]);
            cand[i*3+0] = target_r * std::cos(theta);
            cand[i*3+1] = target_r * std::sin(theta);
            pulled = true;
        }
    }

    return {cand, pulled};
}

// ======================================================================
// insert_polar_quality_ring
// ======================================================================
std::pair<std::vector<double>, std::vector<int64_t>> insert_polar_quality_ring(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    std::span<const int64_t> loop,
    int periods,
    double mean_edge,
    bool upward)
{
    int64_t nv = static_cast<int64_t>(vertices.size()) / 3;
    int64_t ring_size = static_cast<int64_t>(loop.size());
    if (ring_size < 3) return {std::vector<double>(vertices.begin(), vertices.end()),
                                std::vector<int64_t>(faces_flat.begin(), faces_flat.begin() + num_faces * 3)};

    std::vector<double> verts(vertices.begin(), vertices.end());
    std::vector<int64_t> f(faces_flat.begin(), faces_flat.begin() + num_faces * 3);

    // Compute mean z and r of loop
    double z_loop = 0.0, r_loop = 0.0;
    for (int64_t i = 0; i < ring_size; ++i) {
        const double* p = &verts[loop[i] * 3];
        z_loop += p[2];
        r_loop += std::sqrt(p[0]*p[0] + p[1]*p[1]);
    }
    z_loop /= ring_size;
    r_loop /= ring_size;
    if (r_loop <= 1e-12) return {verts, f};

    auto [cand_ring, pulled] = build_polar_candidate_ring(verts, loop, periods, mean_edge, 2.0);

    std::vector<double> new_ring;
    double dz;
    double z_new;

    if (static_cast<int64_t>(cand_ring.size()) / 3 == periods && pulled) {
        new_ring = cand_ring;
        z_new = 0.0;
        for (int i = 0; i < periods; ++i) z_new += new_ring[i*3+2];
        z_new /= periods;
        dz = std::max(0.35 * mean_edge, 0.08 * std::max(r_loop, 1e-9));
    } else {
        double r_new = std::max(0.35 * r_loop, 0.8 * mean_edge);
        dz = std::max(0.35 * mean_edge, 0.08 * r_loop);
        z_new = z_loop + (upward ? dz : -dz);
        double ang0 = std::atan2(verts[loop[0]*3+1], verts[loop[0]*3+0]);
        new_ring.resize(periods * 3);
        for (int i = 0; i < periods; ++i) {
            double alpha = ang0 + i * (2.0 * M_PI / periods);
            new_ring[i*3+0] = r_new * std::cos(alpha);
            new_ring[i*3+1] = r_new * std::sin(alpha);
            new_ring[i*3+2] = z_new;
        }
    }

    // Center point
    std::array<double, 3> center = {0.0, 0.0, z_new + (upward ? 0.5 * dz : -0.5 * dz)};

    int64_t base = nv;
    std::vector<int64_t> ring_ids(periods);
    for (int i = 0; i < periods; ++i) ring_ids[i] = base + i;
    int64_t center_id = base + periods;

    // Add new vertices
    verts.insert(verts.end(), new_ring.begin(), new_ring.end());
    verts.insert(verts.end(), {center[0], center[1], center[2]});

    // Bridge between original loop and new ring
    auto bridge = zipper_stitch_closed(loop, ring_ids, verts);

    // Cap
    std::vector<int64_t> cap;
    for (int i = 0; i < periods; ++i) {
        int64_t a = ring_ids[i];
        int64_t b = ring_ids[(i + 1) % periods];
        if (upward) {
            cap.insert(cap.end(), {a, b, center_id});
        } else {
            cap.insert(cap.end(), {a, center_id, b});
        }
    }

    f.insert(f.end(), bridge.begin(), bridge.end());
    f.insert(f.end(), cap.begin(), cap.end());

    return {verts, f};
}

// ======================================================================
// mesh_polar_holes
// ======================================================================
std::pair<std::vector<double>, std::vector<int64_t>> mesh_polar_holes(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    int periods,
    double mean_edge)
{
    std::vector<double> verts(vertices.begin(), vertices.end());
    std::vector<int64_t> f(faces_flat.begin(), faces_flat.begin() + num_faces * 3);

    auto [loop_verts, loop_starts] = extract_boundary_loops_flat(f, static_cast<int64_t>(f.size()) / 3);
    int64_t num_loops = static_cast<int64_t>(loop_starts.size()) - 1;
    if (num_loops == 0) return {verts, f};

    // Select which loops to process
    std::vector<std::pair<int64_t, int64_t>> loops_to_use; // (start, end)
    if (num_loops == 1) {
        loops_to_use.push_back({loop_starts[0], loop_starts[1]});
    } else {
        // Find loops with min and max z
        double z_min = std::numeric_limits<double>::infinity();
        double z_max = -std::numeric_limits<double>::infinity();
        int64_t low_i = -1, high_i = -1;
        for (int64_t i = 0; i < num_loops; ++i) {
            int64_t s = loop_starts[i];
            int64_t e = loop_starts[i + 1];
            double z_sum = 0;
            for (int64_t j = s; j < e; ++j) z_sum += verts[loop_verts[j] * 3 + 2];
            double z_mean = z_sum / (e - s);
            if (z_mean < z_min) { z_min = z_mean; low_i = i; }
            if (z_mean > z_max) { z_max = z_mean; high_i = i; }
        }
        if (low_i >= 0) loops_to_use.push_back({loop_starts[low_i], loop_starts[low_i + 1]});
        if (high_i >= 0 && high_i != low_i) loops_to_use.push_back({loop_starts[high_i], loop_starts[high_i + 1]});
    }

    for (const auto& [ls, le] : loops_to_use) {
        std::vector<int64_t> lp(loop_verts.begin() + ls, loop_verts.begin() + le);
        double perim = loop_perimeter(verts, lp);
        int64_t lp_size = static_cast<int64_t>(lp.size());

        bool is_large = (lp_size >= periods + 1) && (perim > 2.0 * periods * mean_edge);
        bool is_small = (lp_size <= periods) && (perim < 0.3 * periods * mean_edge);

        if (is_large) {
            double zc = 0;
            for (auto idx : lp) zc += verts[idx * 3 + 2];
            zc /= lp_size;
            double z_median = 0;
            {
                std::vector<double> all_z;
                for (int64_t vi = 0; vi < static_cast<int64_t>(verts.size()) / 3; ++vi)
                    all_z.push_back(verts[vi * 3 + 2]);
                std::sort(all_z.begin(), all_z.end());
                size_t nz = all_z.size();
                if (nz % 2 == 0)
                    z_median = (all_z[nz/2 - 1] + all_z[nz/2]) * 0.5;
                else
                    z_median = all_z[nz/2];
            }
            bool up = zc >= z_median;
            auto [new_v, new_f] = insert_polar_quality_ring(verts, f, static_cast<int64_t>(f.size()) / 3, lp, periods, mean_edge, up);
            verts = std::move(new_v);
            f = std::move(new_f);
        } else if (is_small) {
            double zc = 0;
            for (auto idx : lp) zc += verts[idx * 3 + 2];
            zc /= lp_size;
            double z_median = 0;
            {
                std::vector<double> all_z;
                for (int64_t vi = 0; vi < static_cast<int64_t>(verts.size()) / 3; ++vi)
                    all_z.push_back(verts[vi * 3 + 2]);
                std::sort(all_z.begin(), all_z.end());
                size_t nz = all_z.size();
                if (nz % 2 == 0)
                    z_median = (all_z[nz/2 - 1] + all_z[nz/2]) * 0.5;
                else
                    z_median = all_z[nz/2];
            }
            bool up = zc >= z_median;

            // Remove faces touching the ring
            std::unordered_set<int64_t> lp_set(lp.begin(), lp.end());
            std::vector<int64_t> f_cut;
            int64_t nf = static_cast<int64_t>(f.size()) / 3;
            for (int64_t fi = 0; fi < nf; ++fi) {
                int64_t a = f[fi*3], b = f[fi*3+1], c = f[fi*3+2];
                if (lp_set.count(a) || lp_set.count(b) || lp_set.count(c)) continue;
                f_cut.push_back(a); f_cut.push_back(b); f_cut.push_back(c);
            }
            if (f_cut.empty()) continue;
            f = std::move(f_cut);

            // Recompute boundary
            auto [new_lv, new_ls] = extract_boundary_loops_flat(f, static_cast<int64_t>(f.size()) / 3);
            int64_t new_nl = static_cast<int64_t>(new_ls.size()) - 1;
            if (new_nl == 0) continue;

            std::vector<int64_t> new_lp;
            if (new_nl == 1) {
                new_lp.assign(new_lv.begin() + new_ls[0], new_lv.begin() + new_ls[1]);
            } else {
                int64_t best_i = up ? 0 : 0;
                double best_z = up ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                for (int64_t li = 0; li < new_nl; ++li) {
                    double z_sum = 0;
                    for (int64_t j = new_ls[li]; j < new_ls[li+1]; ++j)
                        z_sum += verts[new_lv[j] * 3 + 2];
                    double z_mean = z_sum / (new_ls[li+1] - new_ls[li]);
                    if ((up && z_mean > best_z) || (!up && z_mean < best_z)) {
                        best_z = z_mean;
                        best_i = li;
                    }
                }
                new_lp.assign(new_lv.begin() + new_ls[best_i], new_lv.begin() + new_ls[best_i + 1]);
            }

            // Seal the new ring
            std::vector<int64_t> loop_starts_vec = {0, static_cast<int64_t>(new_lp.size())};
            f = seal_loops_constrained(f, static_cast<int64_t>(f.size()) / 3, new_lp, loop_starts_vec, 1, 512);
        } else {
            // Neither large nor small: seal directly
            f = seal_loops_constrained(f, static_cast<int64_t>(f.size()) / 3,
                                       std::span<const int64_t>(), std::span<const int64_t>(), 0, 16);
        }
    }

    if (f.empty()) return {verts, f};

    // Compact vertex indexing
    std::unordered_set<int64_t> used_set;
    int64_t nf = static_cast<int64_t>(f.size()) / 3;
    for (int64_t i = 0; i < nf; ++i) {
        used_set.insert(f[i*3]);
        used_set.insert(f[i*3+1]);
        used_set.insert(f[i*3+2]);
    }
    std::vector<int64_t> used(used_set.begin(), used_set.end());
    std::sort(used.begin(), used.end());

    int64_t nv_old = static_cast<int64_t>(verts.size()) / 3;
    std::vector<int64_t> remap(nv_old, -1);
    for (int64_t i = 0; i < static_cast<int64_t>(used.size()); ++i)
        remap[used[i]] = i;

    std::vector<double> new_verts(used.size() * 3);
    for (int64_t i = 0; i < static_cast<int64_t>(used.size()); ++i)
        std::memcpy(&new_verts[i * 3], &verts[used[i] * 3], 3 * sizeof(double));

    for (int64_t i = 0; i < nf; ++i) {
        f[i*3+0] = remap[f[i*3+0]];
        f[i*3+1] = remap[f[i*3+1]];
        f[i*3+2] = remap[f[i*3+2]];
    }

    return {new_verts, f};
}

// ======================================================================
// extract_sector
// ======================================================================
SectorResult extract_sector(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    double alpha, double threshold, double tol)
{
    int64_t nv = static_cast<int64_t>(vertices.size()) / 3;

    // Compute r_xy and theta for all vertices
    std::vector<double> r_xy(nv), theta(nv);
    for (int64_t i = 0; i < nv; ++i) {
        double x = vertices[i*3+0], y = vertices[i*3+1];
        r_xy[i] = std::sqrt(x*x + y*y);
        theta[i] = std::fmod(std::atan2(y, x) + 2.0 * M_PI, 2.0 * M_PI);
    }

    // radial_ref
    double radial_ref = 1.0;
    {
        double sum_r = 0; int64_t cnt = 0;
        for (int64_t i = 0; i < nv; ++i) {
            if (r_xy[i] > tol) { sum_r += r_xy[i]; cnt++; }
        }
        if (cnt > 0) radial_ref = sum_r / cnt;
    }

    double ang_eps = std::max(tol, threshold / std::max(radial_ref, 1e-12));
    ang_eps = std::min(0.24 * alpha, ang_eps);
    ang_eps = std::max(1e-6, ang_eps);

    double axis_eps = std::max(tol, 0.5 * threshold);

    PoleTriangles poles = find_axis_pole_triangles(vertices, faces_flat, num_faces);

    int phase_samples = std::max(8, static_cast<int>(std::ceil(2.0 * M_PI / alpha)));

    SectorResult best;
    double best_key_val = std::numeric_limits<double>::infinity();

    for (int ps = 0; ps < phase_samples; ++ps) {
        double phase = alpha * static_cast<double>(ps) / phase_samples;
        double eps_try = ang_eps;

        for (int iter = 0; iter < 18; ++iter) {
            // Determine which vertices are in sector
            std::vector<bool> keep(nv, false);
            int64_t keep_count = 0;
            for (int64_t i = 0; i < nv; ++i) {
                double th = std::fmod(theta[i] - phase + 2.0 * M_PI, 2.0 * M_PI);
                if ((th > eps_try && th < alpha - eps_try) || r_xy[i] <= axis_eps) {
                    keep[i] = true;
                    keep_count++;
                }
            }
            if (keep_count == 0) { eps_try *= 0.5; continue; }

            // Filter faces
            std::vector<int64_t> f_sector;
            for (int64_t fi = 0; fi < num_faces; ++fi) {
                if (keep[faces_flat[fi*3]] && keep[faces_flat[fi*3+1]] && keep[faces_flat[fi*3+2]]) {
                    f_sector.push_back(faces_flat[fi*3]);
                    f_sector.push_back(faces_flat[fi*3+1]);
                    f_sector.push_back(faces_flat[fi*3+2]);
                }
            }
            if (f_sector.empty()) { eps_try *= 0.5; continue; }

            int64_t n_sector = static_cast<int64_t>(f_sector.size()) / 3;
            auto comps = face_components_by_edges(f_sector, n_sector);
            if (comps.empty()) { eps_try *= 0.5; continue; }

            double side_band = std::max(3.0 * eps_try, 5.0 * tol);
            std::vector<int64_t> best_comp_faces;
            double best_comp_key_val = -std::numeric_limits<double>::infinity();

            for (const auto& comp : comps) {
                std::vector<int64_t> f_comp;
                for (int64_t fi : comp) {
                    f_comp.push_back(f_sector[fi*3]);
                    f_comp.push_back(f_sector[fi*3+1]);
                    f_comp.push_back(f_sector[fi*3+2]);
                }

                // Find unique vertex ids in this component
                std::unordered_set<int64_t> comp_ids_set;
                for (int64_t fi : comp) {
                    comp_ids_set.insert(f_sector[fi*3]);
                    comp_ids_set.insert(f_sector[fi*3+1]);
                    comp_ids_set.insert(f_sector[fi*3+2]);
                }
                std::vector<int64_t> comp_ids(comp_ids_set.begin(), comp_ids_set.end());

                int64_t touch_left = 0, touch_right = 0;
                for (int64_t vid : comp_ids) {
                    double th = std::fmod(theta[vid] - phase + 2.0 * M_PI, 2.0 * M_PI);
                    if (th <= side_band) touch_left++;
                    if ((alpha - th) <= side_band) touch_right++;
                }
                int64_t touch_both = (touch_left >= 2 && touch_right >= 2) ? 1 : 0;
                double area = tri_area_sum(vertices, f_comp, static_cast<int64_t>(f_comp.size()) / 3);
                double key_val = static_cast<double>(touch_both) * 1e20 + area;
                if (key_val > best_comp_key_val) {
                    best_comp_key_val = key_val;
                    best_comp_faces = std::move(f_comp);
                }
            }

            if (best_comp_faces.empty()) { eps_try *= 0.5; continue; }

            // Build local vertex set for the best component
            int64_t n_comp = static_cast<int64_t>(best_comp_faces.size()) / 3;
            std::unordered_set<int64_t> ids_comp_set;
            for (int64_t i = 0; i < n_comp; ++i) {
                ids_comp_set.insert(best_comp_faces[i*3]);
                ids_comp_set.insert(best_comp_faces[i*3+1]);
                ids_comp_set.insert(best_comp_faces[i*3+2]);
            }
            std::vector<int64_t> ids_comp(ids_comp_set.begin(), ids_comp_set.end());
            std::sort(ids_comp.begin(), ids_comp.end());

            std::vector<int64_t> remap(nv, -1);
            for (int64_t i = 0; i < static_cast<int64_t>(ids_comp.size()); ++i)
                remap[ids_comp[i]] = i;

            std::vector<int64_t> f_local(n_comp * 3);
            for (int64_t i = 0; i < n_comp; ++i) {
                f_local[i*3+0] = remap[best_comp_faces[i*3+0]];
                f_local[i*3+1] = remap[best_comp_faces[i*3+1]];
                f_local[i*3+2] = remap[best_comp_faces[i*3+2]];
            }

            std::vector<double> v_local(ids_comp.size() * 3);
            for (int64_t i = 0; i < static_cast<int64_t>(ids_comp.size()); ++i)
                std::memcpy(&v_local[i*3], &vertices[ids_comp[i]*3], 3 * sizeof(double));

            // Extract boundary loops of the local sector
            auto [loop_v, loop_s] = extract_boundary_loops_flat(f_local, n_comp);
            int64_t num_loops_local = static_cast<int64_t>(loop_s.size()) - 1;
            if (num_loops_local == 0) { eps_try *= 0.5; continue; }

            // Compute local theta
            std::vector<double> theta_local(ids_comp.size());
            for (int64_t i = 0; i < static_cast<int64_t>(ids_comp.size()); ++i)
                theta_local[i] = std::fmod(theta[ids_comp[i]] - phase + 2.0 * M_PI, 2.0 * M_PI);

            // Find the best seam loop
            int64_t seam_idx = -1;
            double seam_score_best = -1;
            for (int64_t li = 0; li < num_loops_local; ++li) {
                int64_t s = loop_s[li], e = loop_s[li + 1];
                int64_t lcnt = 0, rcnt = 0;
                for (int64_t j = s; j < e; ++j) {
                    double th = theta_local[loop_v[j]];
                    if (th <= side_band) lcnt++;
                    if ((alpha - th) <= side_band) rcnt++;
                }
                double score = std::min(lcnt, rcnt) * 1e8 + (lcnt + rcnt) * 1e4 + (e - s);
                if (score > seam_score_best) {
                    seam_score_best = score;
                    seam_idx = li;
                }
            }
            if (seam_idx < 0) { eps_try *= 0.5; continue; }

            int64_t seam_s = loop_s[seam_idx], seam_e = loop_s[seam_idx + 1];
            std::vector<int64_t> seam_loop(loop_v.begin() + seam_s, loop_v.begin() + seam_e);
            double mean_edge_local = mean_edge_length(v_local, f_local, n_comp);

            // Preferred south/north anchors
            int64_t pref_s_idx = -1, pref_n_idx = -1;
            double pref_s_pt[3] = {0, 0, 0}, pref_n_pt[3] = {0, 0, 0};
            bool has_pref_s_pt = false, has_pref_n_pt = false;

            if (poles.has_south) {
                std::vector<int64_t> local_s;
                for (int64_t g : poles.south_tri) {
                    if (g >= 0 && g < nv && remap[g] >= 0) local_s.push_back(remap[g]);
                }
                if (!local_s.empty()) {
                    double best_r = std::numeric_limits<double>::infinity();
                    for (int64_t idx : local_s) {
                        double r = std::sqrt(v_local[idx*3]*v_local[idx*3] + v_local[idx*3+1]*v_local[idx*3+1]);
                        if (r < best_r) { best_r = r; pref_s_idx = idx; }
                    }
                } else {
                    auto anchor = virtual_anchor_from_triangle(
                        std::span<const double>(&vertices[poles.south_tri[0]*3], 9),
                        poles.south_hit, phase, alpha, mean_edge_local);
                    pref_s_pt[0] = anchor[0]; pref_s_pt[1] = anchor[1]; pref_s_pt[2] = anchor[2];
                    has_pref_s_pt = true;
                }
            }

            if (poles.has_north) {
                std::vector<int64_t> local_n;
                for (int64_t g : poles.north_tri) {
                    if (g >= 0 && g < nv && remap[g] >= 0) local_n.push_back(remap[g]);
                }
                if (!local_n.empty()) {
                    double best_r = std::numeric_limits<double>::infinity();
                    for (int64_t idx : local_n) {
                        double r = std::sqrt(v_local[idx*3]*v_local[idx*3] + v_local[idx*3+1]*v_local[idx*3+1]);
                        if (r < best_r) { best_r = r; pref_n_idx = idx; }
                    }
                } else {
                    auto anchor = virtual_anchor_from_triangle(
                        std::span<const double>(&vertices[poles.north_tri[0]*3], 9),
                        poles.north_hit, phase, alpha, mean_edge_local);
                    pref_n_pt[0] = anchor[0]; pref_n_pt[1] = anchor[1]; pref_n_pt[2] = anchor[2];
                    has_pref_n_pt = true;
                }
            }

            auto [path1, path2] = split_seam_paths(
                seam_loop, v_local,
                pref_s_idx, pref_n_idx,
                has_pref_s_pt ? pref_s_pt : nullptr,
                has_pref_n_pt ? pref_n_pt : nullptr);

            if (path1.size() < 2 || path2.size() < 2) { eps_try *= 0.5; continue; }

            // Determine left/right by mean theta
            double mean1 = 0, mean2 = 0;
            for (auto idx : path1) mean1 += theta_local[idx];
            for (auto idx : path2) mean2 += theta_local[idx];
            mean1 /= path1.size();
            mean2 /= path2.size();

            std::vector<int64_t> left, right;
            if (mean1 <= mean2) { left = path1; right = path2; }
            else { left = path2; right = path1; }

            // Handle interior loops
            std::vector<int64_t> interior_loops_data;
            std::vector<int64_t> interior_loop_starts = {0};
            for (int64_t li = 0; li < num_loops_local; ++li) {
                if (li == seam_idx) continue;
                interior_loop_starts.push_back(interior_loop_starts.back() + (loop_s[li+1] - loop_s[li]));
                interior_loops_data.insert(interior_loops_data.end(),
                    loop_v.begin() + loop_s[li], loop_v.begin() + loop_s[li+1]);
            }

            if (!interior_loops_data.empty()) {
                std::unordered_set<int64_t> interior_global;
                for (auto idx : interior_loops_data)
                    interior_global.insert(ids_comp[idx]);

                auto f_patch = recover_removed_faces_for_interior_holes(
                    f_sector, n_sector, best_comp_faces, n_comp, interior_global);

                if (!f_patch.empty()) {
                    // Extend ids_comp if needed
                    std::unordered_set<int64_t> old_set(ids_comp.begin(), ids_comp.end());
                    for (int64_t i = 0; i < static_cast<int64_t>(f_patch.size()); i += 3) {
                        for (int k = 0; k < 3; ++k) {
                            if (!old_set.count(f_patch[i+k])) {
                                old_set.insert(f_patch[i+k]);
                                ids_comp.push_back(f_patch[i+k]);
                            }
                        }
                    }
                    std::sort(ids_comp.begin(), ids_comp.end());

                    // Rebuild remap
                    remap.assign(nv, -1);
                    for (int64_t i = 0; i < static_cast<int64_t>(ids_comp.size()); ++i)
                        remap[ids_comp[i]] = i;

                    // Rebuild best_comp_faces
                    n_comp = static_cast<int64_t>(best_comp_faces.size()) / 3;
                    int64_t patch_n = static_cast<int64_t>(f_patch.size()) / 3;
                    best_comp_faces.insert(best_comp_faces.end(), f_patch.begin(), f_patch.end());
                    n_comp += patch_n;

                    // Rebuild f_local
                    f_local.resize(n_comp * 3);
                    for (int64_t i = 0; i < n_comp; ++i) {
                        f_local[i*3+0] = remap[best_comp_faces[i*3+0]];
                        f_local[i*3+1] = remap[best_comp_faces[i*3+1]];
                        f_local[i*3+2] = remap[best_comp_faces[i*3+2]];
                    }

                    // Rebuild v_local
                    v_local.resize(ids_comp.size() * 3);
                    for (int64_t i = 0; i < static_cast<int64_t>(ids_comp.size()); ++i)
                        std::memcpy(&v_local[i*3], &vertices[ids_comp[i]*3], 3 * sizeof(double));
                }
            }

            double area_local = tri_area_sum(v_local, f_local, n_comp);
            int64_t num_interior_loops = static_cast<int64_t>(interior_loop_starts.size()) - 1;
            double candidate_key = static_cast<double>(num_interior_loops) * 1e12
                                 + std::abs(static_cast<double>(left.size()) - right.size()) * 1e6
                                 - area_local;

            if (candidate_key < best_key_val) {
                best_key_val = candidate_key;
                best.vertices = std::move(v_local);
                best.faces = std::move(f_local);
                best.left_side = std::move(left);
                best.right_side = std::move(right);
                best.valid = true;
            }

            break; // Success - exit the eps_try loop
        }
    }

    if (!best.valid) {
        throw std::runtime_error("Failed to extract a clean rotational sector. Consider decreasing threshold/tol.");
    }
    return best;
}

// ======================================================================
// extract_boundary_loops_flat (wrapper)
// ======================================================================
std::pair<std::vector<int64_t>, std::vector<int64_t>> extract_boundary_loops_flat(
    std::span<const int64_t> faces_flat, int64_t num_faces)
{
    // Convert to int32 for the existing extractBoundaryLoops
    std::vector<int> faces_i32(faces_flat.begin(), faces_flat.end());
    auto loops = cpgeo::extractBoundaryLoops(std::span<const int>(faces_i32.data(), faces_i32.size()));

    std::vector<int64_t> all_v;
    std::vector<int64_t> starts = {0};
    for (const auto& loop : loops) {
        for (int v : loop) all_v.push_back(static_cast<int64_t>(v));
        starts.push_back(static_cast<int64_t>(all_v.size()));
    }
    return {all_v, starts};
}

// ======================================================================
// get_mesh_edges_flat (wrapper)
// ======================================================================
std::vector<int64_t> get_mesh_edges_flat(std::span<const int64_t> faces_flat, int64_t num_faces) {
    std::vector<int> faces_i32(faces_flat.begin(), faces_flat.end());
    auto edge_map = cpgeo::extractEdgesWithNumber(std::span<const int>(faces_i32.data(), faces_i32.size()));
    std::vector<int64_t> result;
    for (const auto& [edge, count] : edge_map) {
        result.push_back(static_cast<int64_t>(edge.first));
        result.push_back(static_cast<int64_t>(edge.second));
        result.push_back(static_cast<int64_t>(count));
    }
    return result;
}

// ======================================================================
// optimize_mesh_by_edge_flipping_wrapper
// ======================================================================
std::vector<int64_t> optimize_mesh_by_edge_flipping_wrapper(
    std::span<const double> vertices,
    std::span<const int64_t> faces_flat, int64_t num_faces,
    int max_iterations)
{
    std::vector<int> faces_i32(faces_flat.begin(), faces_flat.end());
    auto result = cpgeo::mesh_optimize_by_edge_flipping(vertices, 3,
        std::span<const int>(faces_i32.data(), faces_i32.size()), max_iterations);
    std::vector<int64_t> out(result.begin(), result.end());
    return out;
}

// ======================================================================
// enforce_rotational_symmetry_z — main entry point
// ======================================================================
RotationalSymmetryResult enforce_rotational_symmetry_z(
    std::span<const double> vertices,
    std::span<const int64_t> faces,
    int periods,
    double threshold,
    double tol,
    bool return_match)
{
    if (periods < 2)
        throw std::invalid_argument("periods must be >= 2");

    int64_t num_faces = static_cast<int64_t>(faces.size()) / 3;

    // Compute mean edge length
    auto edges_flat = get_mesh_edges_flat(faces, num_faces);
    double sum_len = 0.0;
    int64_t edge_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(edges_flat.size()); i += 3) {
        int64_t a = edges_flat[i];
        int64_t b = edges_flat[i+1];
        double dx = vertices[a*3+0] - vertices[b*3+0];
        double dy = vertices[a*3+1] - vertices[b*3+1];
        double dz = vertices[a*3+2] - vertices[b*3+2];
        sum_len += std::sqrt(dx*dx + dy*dy + dz*dz);
        edge_count++;
    }
    double mean_edge = edge_count > 0 ? sum_len / edge_count : 1.0;

    if (threshold < 0)
        threshold = mean_edge * 0.2;

    double alpha = 2.0 * M_PI / periods;

    SectorResult sector = extract_sector(vertices, faces, num_faces, alpha, threshold, tol);

    int64_t m = static_cast<int64_t>(sector.vertices.size()) / 3;

    // Rotate and replicate sectors
    std::vector<double> verts_all;
    verts_all.reserve(m * 3 * periods);
    verts_all.insert(verts_all.end(), sector.vertices.begin(), sector.vertices.end());
    for (int k = 1; k < periods; ++k) {
        auto rotated = rot_z(sector.vertices, k * alpha);
        verts_all.insert(verts_all.end(), rotated.begin(), rotated.end());
    }

    // Pole trimming
    int trim_count = decide_pole_trim_count(sector.left_side, sector.right_side, sector.vertices, mean_edge);
    std::vector<int64_t> left_side = sector.left_side;
    std::vector<int64_t> right_side = sector.right_side;

    if (trim_count > 0) {
        int64_t ls = static_cast<int64_t>(left_side.size());
        int64_t rs = static_cast<int64_t>(right_side.size());
        if (ls > 2 * trim_count && rs > 2 * trim_count) {
            left_side = std::vector<int64_t>(left_side.begin() + trim_count, left_side.end() - trim_count);
            right_side = std::vector<int64_t>(right_side.begin() + trim_count, right_side.end() - trim_count);
        }
    }

    // Build faces for all sectors
    std::vector<int64_t> faces_all;
    for (int k = 0; k < periods; ++k) {
        int64_t off = k * m;
        for (int64_t i = 0; i < static_cast<int64_t>(sector.faces.size()); i += 3) {
            faces_all.push_back(sector.faces[i+0] + off);
            faces_all.push_back(sector.faces[i+1] + off);
            faces_all.push_back(sector.faces[i+2] + off);
        }
    }

    // Seam stitching
    std::vector<int64_t> left_global, right_global;
    for (int k = 0; k < periods; ++k) {
        int64_t off = k * m;
        for (auto id : left_side) left_global.push_back(id + off);
        for (auto id : right_side) right_global.push_back(id + off);
    }

    for (int k = 0; k < periods; ++k) {
        int nk = (k + 1) % periods;

        int64_t ls = static_cast<int64_t>(left_side.size());
        int64_t rs = static_cast<int64_t>(right_side.size());

        std::vector<int64_t> rid(right_global.begin() + k * rs, right_global.begin() + (k + 1) * rs);
        std::vector<int64_t> lid(left_global.begin() + nk * ls, left_global.begin() + (nk + 1) * ls);

        std::vector<double> rpts(rs * 3), lpts(ls * 3);
        for (int64_t i = 0; i < rs; ++i)
            std::memcpy(&rpts[i*3], &verts_all[rid[i]*3], 3 * sizeof(double));
        for (int64_t i = 0; i < ls; ++i)
            std::memcpy(&lpts[i*3], &verts_all[lid[i]*3], 3 * sizeof(double));

        auto seam = zipper_stitch(rid, rpts, lid, lpts);
        faces_all.insert(faces_all.end(), seam.begin(), seam.end());
    }

    // Mesh polar holes
    int64_t nf_all = static_cast<int64_t>(faces_all.size()) / 3;
    auto [v_polar, f_polar] = mesh_polar_holes(verts_all, faces_all, nf_all, periods, mean_edge);

    // Edge flip optimization
    auto f_opt = optimize_mesh_by_edge_flipping_wrapper(v_polar, f_polar, static_cast<int64_t>(f_polar.size()) / 3);

    RotationalSymmetryResult result;
    result.vertices = std::move(v_polar);
    result.faces = std::move(f_opt);

    if (return_match) {
        result.match.resize(periods * m);
        for (int k = 0; k < periods; ++k) {
            int64_t off = k * m;
            for (int64_t i = 0; i < m; ++i)
                result.match[k * m + i] = off + i;
        }
    }

    return result;
}

} // namespace cpgeo
