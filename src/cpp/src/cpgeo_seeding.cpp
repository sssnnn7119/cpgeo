#include "cpgeo_seeding.h"

namespace cpgeo {

static double triangleArea(
    std::span<const double, 3> v0,
    std::span<const double, 3> v1,
    std::span<const double, 3> v2
){
    std::array<double, 3> a{
        v1[0] - v0[0],
        v1[1] - v0[1],
        v1[2] - v0[2]
    };
    std::array<double, 3> b{
        v2[0] - v0[0],
        v2[1] - v0[1],
        v2[2] - v0[2]
    };

    std::array<double, 3> cross_product{
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };

    double area = 0.5 * std::sqrt(
        cross_product[0]*cross_product[0] +
        cross_product[1]*cross_product[1] +
        cross_product[2]*cross_product[2]
    );

    return area;
}

static std::vector<int> insert_delete_points(
    std::vector<double>& vertices_sphere,
    std::span<const double> control_points,
    SpaceTree& tree,
    double seed_size
){

    const double ANGLE_THRESHOLD = 150.0 * 3.141592653589793 / 180.0; // radians
    const int MAX_ITER = 10000; // safety guard to avoid pathological infinite loops
    int iter_count = 0;
    std::vector<int> faces;

    // create triangulator once and reuse, update points each iteration
    SphereTriangulation triangulator(vertices_sphere);

    while (true) {
        if (++iter_count > MAX_ITER) {
            // safety: stop if too many iterations
            break;
        }

        bool any_change = false;

        // update stored points each iteration to reflect modifications
        triangulator.set_points(std::span<const double>(vertices_sphere.data(), vertices_sphere.size()));
        triangulator.triangulate();
        faces.clear();
        faces.resize(triangulator.size() * 3);
        triangulator.getTriangleIndices(faces);

        if (faces.empty()) {
            break;
        }

        auto r = map_points_batch(vertices_sphere, tree, control_points);

        // Combined triangle checks: oversized triangle (split), very small area (merge), large-angle (split)
        double max_area = seed_size * seed_size * 1.2;
        double min_area = seed_size * seed_size * 0.2;
        double cos_threshold = std::cos(ANGLE_THRESHOLD);
        for (int tridx = 0; tridx < static_cast<int>(faces.size() / 3); ++tridx) {
            int v0_idx = faces[tridx * 3 + 0];
            int v1_idx = faces[tridx * 3 + 1];
            int v2_idx = faces[tridx * 3 + 2];

            // Use mapped/batch-mapped points for geometric tests to reflect current mapping
            std::span<double, 3> mv0(r.data() + v0_idx * 3, 3);
            std::span<double, 3> mv1(r.data() + v1_idx * 3, 3);
            std::span<double, 3> mv2(r.data() + v2_idx * 3, 3);

            double area = triangleArea(mv0, mv1, mv2);

            // 1) Oversized triangle -> insert centroid (use actual sphere coords for insertion)
            if (area > max_area) {
                std::span<double, 3> sv0(vertices_sphere.data() + v0_idx * 3, 3);
                std::span<double, 3> sv1(vertices_sphere.data() + v1_idx * 3, 3);
                std::span<double, 3> sv2(vertices_sphere.data() + v2_idx * 3, 3);

                std::array<double, 3> new_point{
                    (sv0[0] + sv1[0] + sv2[0]) / 3.0,
                    (sv0[1] + sv1[1] + sv2[1]) / 3.0,
                    (sv0[2] + sv1[2] + sv2[2]) / 3.0
                };
                double norm = std::sqrt(new_point[0]*new_point[0] + new_point[1]*new_point[1] + new_point[2]*new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }

                vertices_sphere.push_back(new_point[0]); vertices_sphere.push_back(new_point[1]); vertices_sphere.push_back(new_point[2]);
                any_change = true; break;
            }

            // 2) Very small area -> merge triple into averaged point
            if (area < min_area) {
                std::span<double, 3> sv0(vertices_sphere.data() + v0_idx * 3, 3);
                std::span<double, 3> sv1(vertices_sphere.data() + v1_idx * 3, 3);
                std::span<double, 3> sv2(vertices_sphere.data() + v2_idx * 3, 3);

                std::array<double, 3> new_point{ (sv0[0] + sv1[0] + sv2[0]) / 3.0, (sv0[1] + sv1[1] + sv2[1]) / 3.0, (sv0[2] + sv1[2] + sv2[2]) / 3.0 };
                double norm = std::sqrt(new_point[0]*new_point[0] + new_point[1]*new_point[1] + new_point[2]*new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }

                vertices_sphere.push_back(new_point[0]); vertices_sphere.push_back(new_point[1]); vertices_sphere.push_back(new_point[2]);

                // remove old points in descending index order
                if (v0_idx > v1_idx) std::swap(v0_idx, v1_idx);
                if (v1_idx > v2_idx) std::swap(v1_idx, v2_idx);
                if (v0_idx > v1_idx) std::swap(v0_idx, v1_idx);
                vertices_sphere.erase(vertices_sphere.begin() + v2_idx * 3, vertices_sphere.begin() + v2_idx * 3 + 3);
                vertices_sphere.erase(vertices_sphere.begin() + v1_idx * 3, vertices_sphere.begin() + v1_idx * 3 + 3);
                vertices_sphere.erase(vertices_sphere.begin() + v0_idx * 3, vertices_sphere.begin() + v0_idx * 3 + 3);

                any_change = true; break;
            }

            // 3) Large-angle splitting: check angles via cosine
            auto vec01 = std::array<double, 3>{ mv1[0] - mv0[0], mv1[1] - mv0[1], mv1[2] - mv0[2] };
            auto vec02 = std::array<double, 3>{ mv2[0] - mv0[0], mv2[1] - mv0[1], mv2[2] - mv0[2] };
            auto vec10 = std::array<double, 3>{ mv0[0] - mv1[0], mv0[1] - mv1[1], mv0[2] - mv1[2] };
            auto vec12 = std::array<double, 3>{ mv2[0] - mv1[0], mv2[1] - mv1[1], mv2[2] - mv1[2] };
            auto vec20 = std::array<double, 3>{ mv0[0] - mv2[0], mv0[1] - mv2[1], mv0[2] - mv2[2] };
            auto vec21 = std::array<double, 3>{ mv1[0] - mv2[0], mv1[1] - mv2[1], mv1[2] - mv2[2] };

            auto cos_between = [](const std::array<double,3>& u, const std::array<double,3>& v){
                double dot = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
                double nu = std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
                double nv = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
                if (nu <= 0 || nv <= 0) return 1.0;
                double c = dot / (nu * nv);
                if (c > 1.0) c = 1.0; if (c < -1.0) c = -1.0; return c;
            };

            double cos0 = cos_between(vec01, vec02);
            double cos1 = cos_between(vec10, vec12);
            double cos2 = cos_between(vec20, vec21);

            if (cos0 < cos_threshold) {
                std::span<double, 3> sv1(vertices_sphere.data() + v1_idx * 3, 3);
                std::span<double, 3> sv2(vertices_sphere.data() + v2_idx * 3, 3);
                std::array<double, 3> new_point{ (sv1[0] + sv2[0]) / 2.0, (sv1[1] + sv2[1]) / 2.0, (sv1[2] + sv2[2]) / 2.0 };
                double norm = std::sqrt(new_point[0]*new_point[0] + new_point[1]*new_point[1] + new_point[2]*new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }
                vertices_sphere.push_back(new_point[0]); vertices_sphere.push_back(new_point[1]); vertices_sphere.push_back(new_point[2]);
                any_change = true; break;
            }
            if (cos1 < cos_threshold) {
                std::span<double, 3> sv0(vertices_sphere.data() + v0_idx * 3, 3);
                std::span<double, 3> sv2(vertices_sphere.data() + v2_idx * 3, 3);
                std::array<double, 3> new_point{ (sv0[0] + sv2[0]) / 2.0, (sv0[1] + sv2[1]) / 2.0, (sv0[2] + sv2[2]) / 2.0 };
                double norm = std::sqrt(new_point[0]*new_point[0] + new_point[1]*new_point[1] + new_point[2]*new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }
                vertices_sphere.push_back(new_point[0]); vertices_sphere.push_back(new_point[1]); vertices_sphere.push_back(new_point[2]);
                any_change = true; break;
            }
            if (cos2 < cos_threshold) {
                std::span<double, 3> sv0(vertices_sphere.data() + v0_idx * 3, 3);
                std::span<double, 3> sv1(vertices_sphere.data() + v1_idx * 3, 3);
                std::array<double, 3> new_point{ (sv0[0] + sv1[0]) / 2.0, (sv0[1] + sv1[1]) / 2.0, (sv0[2] + sv1[2]) / 2.0 };
                double norm = std::sqrt(new_point[0]*new_point[0] + new_point[1]*new_point[1] + new_point[2]*new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }
                vertices_sphere.push_back(new_point[0]); vertices_sphere.push_back(new_point[1]); vertices_sphere.push_back(new_point[2]);
                any_change = true; break;
            }
        }


		// Edge-based small boundary edge merging
        double min_edge_length = seed_size * 0.5;
        auto edges = extractEdgesWithNumber(faces);
        for (const auto& [edge, count] : edges) {
            if (count != 1) continue; // only consider boundary edges
            int a = edge.first;
            int b = edge.second;

            // use mapped coordinates to measure current edge length
            std::span<double, 3> ma(r.data() + a * 3, 3);
            std::span<double, 3> mb(r.data() + b * 3, 3);

            double dx = ma[0] - mb[0];
            double dy = ma[1] - mb[1];
            double dz = ma[2] - mb[2];
            double edge_length = std::sqrt(dx*dx + dy*dy + dz*dz);

            // 4) Edge detection: small boundary edges merging (independent pass)
            if (edge_length < min_edge_length) {
                // merge into midpoint: keep lower index, erase higher index
                if (a > b) std::swap(a, b);

                std::span<double, 3> va(vertices_sphere.data() + a * 3, 3);
                std::span<double, 3> vb(vertices_sphere.data() + b * 3, 3);

                std::array<double, 3> new_point{ (va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0, (va[2] + vb[2]) / 2.0 };
                double norm = std::sqrt(new_point[0]*new_point[0] + new_point[1]*new_point[1] + new_point[2]*new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }

                // write midpoint to a, erase b
                vertices_sphere[a*3 + 0] = new_point[0];
                vertices_sphere[a*3 + 1] = new_point[1];
                vertices_sphere[a*3 + 2] = new_point[2];

                vertices_sphere.erase(vertices_sphere.begin() + b * 3, vertices_sphere.begin() + b * 3 + 3);

                any_change = true;
                break;
            }
        }

        if (any_change) continue;

        // if nothing changed this round, we're done
        if (!any_change) break;
    }
    return faces;
}

static auto closure_edge_length_surface2plane_derivative2(
    std::span<double> _r,
    std::span<double> _rdu,
    std::span<double> _rdu2,
    std::span<const int> edges,
    int order
) -> std::tuple<double, std::vector<double>, std::vector<int>, std::vector<double>> {
    auto [loss, Ldr, Ldr2_indices, Ldr2_values] = closure_edge_length_derivative2(_r, 3, edges, order);


    int num_points = static_cast<int>(_r.size()) / 3;
    int num_indices_Ldr2 = static_cast<int>(Ldr2_indices.size() / 4);

    std::vector<double> Ldu(num_points * 2, 0.0);

	TensorView rdu(std::array<const int, 3>{ num_points, 3, 2}, _rdu);

    std::unordered_map<std::array<int, 4>, double, Array4IntHash> Ldu2_map; // (pt_idx1, dim1, pt_idx2, dim2) -> value

#pragma omp parallel for
    for (int ptidx = 0; ptidx < num_points; ptidx++) {
        for (int iidx = 0; iidx < 3; iidx++) {
			for (int uidx = 0; uidx < 2; uidx++) {
                Ldu[ptidx * 2 + uidx] += Ldr[ptidx * 3 + iidx] * rdu[{ptidx, iidx, uidx}];
            }
        }
    }

	TensorView rdu2(std::array<const int, 4>{ num_points, 3, 2, 2}, _rdu2);

	// Ldu2 += Ldr2 * rdu * rdu + Ldr * rdu2
    for (int idx = 0; idx < num_indices_Ldr2; idx++) {
        int v0_idx = Ldr2_indices[idx * 4 + 0];
        int v0_dim = Ldr2_indices[idx * 4 + 1];
        int v1_idx = Ldr2_indices[idx * 4 + 2];
        int v1_dim = Ldr2_indices[idx * 4 + 3];
        double value = Ldr2_values[idx];

        for (int u0idx = 0; u0idx < 2; u0idx++) {
            for (int u1idx = 0; u1idx < 2; u1idx++) {
                Ldu2_map[{v0_idx, u0idx, v1_idx, u1idx}] += value * rdu[{v0_idx, v0_dim, u0idx}] * rdu[{v1_idx, v1_dim, u1idx}];
            }
        }
    }

	// add Ldr * rdu2
    for(int ptidx = 0; ptidx < num_points; ptidx++) {
        for (int idim = 0; idim < 3; idim++) {
            double Ldr_val = Ldr[ptidx * 3 + idim];
            for (int u0idx = 0; u0idx < 2; u0idx++) {
                for (int u1idx = 0; u1idx < 2; u1idx++) {
                    Ldu2_map[{ptidx, u0idx, ptidx, u1idx}] += Ldr_val * rdu2[{ptidx, idim, u0idx, u1idx}];
                }
            }
        }
	}

	// copy all entries from map to output vectors
	std::vector<int> Ldu2_indices;
	std::vector<double> Ldu2_values;

	Ldu2_indices.reserve(Ldu2_map.size() * 4);
	Ldu2_values.reserve(Ldu2_map.size());

    for (const auto& [key, value] : Ldu2_map) {
        Ldu2_indices.push_back(key[0]);
        Ldu2_indices.push_back(key[1]);
        Ldu2_indices.push_back(key[2]);
        Ldu2_indices.push_back(key[3]);
        Ldu2_values.push_back(value);
	}

	return { loss, Ldu, Ldu2_indices, Ldu2_values };
}

static void vertice_smoothing(
    std::vector<double>& vertices_sphere,
    std::span<int> faces,
    std::span<const double> control_points,
    SpaceTree& tree
){
	int num_points = static_cast<int>(vertices_sphere.size()) / 3;
	std::vector<double> _r(num_points * 3, 0.0);
	std::vector<double> _rdu(num_points * 3 * 2, 0.0);
	std::vector<double> _rdu2(num_points * 3 * 2 * 2, 0.0);

	TensorView r(std::array<const int, 2>{ num_points, 3 }, _r);
	TensorView rdu(std::array<const int, 3>{ num_points, 3, 2 }, _rdu);
	TensorView rdu2(std::array<const int, 4>{ num_points, 3, 2, 2 }, _rdu2);

	auto knots = tree.get_knots();
	auto thresholds = tree.get_thresholds();

    while (true) {



        for (int ptidx = 0; ptidx < num_points; ptidx++) {

			auto vertices_plane = stereographicProjection3_2(std::span<const double, 3>(vertices_sphere.data() + ptidx * 3, 3));

            auto indices_cp = tree.query_point(
                vertices_sphere[ptidx * 3 + 0],
                vertices_sphere[ptidx * 3 + 1],
                vertices_sphere[ptidx * 3 + 2]
            );

            auto [rdot, rdudot, rdu2dot] = get_weights_derivative2(indices_cp, knots, thresholds, vertices_plane);

            auto rnow = get_mapped_points(indices_cp, rdot, control_points);
			r[{ptidx, 0}] = rnow[0];
			r[{ptidx, 1}] = rnow[1];
			r[{ptidx, 2}] = rnow[2];

        }

    }

}



std::tuple<std::vector<double>, std::vector<int>> uniformlyMesh(
    std::span<double> init_vertices_sphere, 
    std::span<const double> control_points,
    SpaceTree& tree,
    double seed_size,
    int max_iterations
){

    std::vector<double> vertices_sphere(init_vertices_sphere.begin(), init_vertices_sphere.end());

    std::vector<int> faces;
    SphereTriangulation triangulator(vertices_sphere);
    triangulator.triangulate();
    faces.clear();
    faces.resize(triangulator.size() * 3);
    triangulator.getTriangleIndices(faces);

    for(int loop=0;loop<max_iterations;loop++){
        std::cout << "  - Mesh uniforming loop " << loop + 1 << "/" << max_iterations << std::endl;

        // refine mesh by edge flipping
        //faces = mesh_optimize_by_edge_flipping(vertices_sphere, 3, faces, 100);

        // refine mesh by vertex smoothing
        vertice_smoothing(vertices_sphere, faces, control_points, tree);

        // insert/delete points
        faces = insert_delete_points(vertices_sphere, control_points, tree, seed_size);

    }

    return {vertices_sphere, faces};
}






} // namespace cpgeo