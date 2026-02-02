#include "cpgeo_seeding.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>

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

    const int MAX_ITER = 5; // safety guard to avoid pathological infinite loops
    int iter_count = 0;
    std::vector<int> faces;

    while (true) {
        if (++iter_count > MAX_ITER) {
            // safety: stop if too many iterations
            break;
        }

        bool any_change = false;

        // update stored points each iteration to reflect modifications
        SphereTriangulation triangulator(vertices_sphere);
        triangulator.triangulate();
        faces.clear();
        faces.resize(triangulator.size() * 3);
        triangulator.getTriangleIndices(faces);

        if (faces.empty()) {
            break;
        }

        auto r = map_points_batch(vertices_sphere, tree, control_points);

        // refine mesh by edge flipping
        faces = mesh_optimize_by_edge_flipping(r, 3, faces, 100);

        // Edge-based small boundary edge merging + large edge splitting (batched)
        double min_edge_length = seed_size * 0.5;
        double max_edge_length = seed_size * 1.5;
        auto edges = extractEdgesWithNumber(faces);

        const int num_points = static_cast<int>(vertices_sphere.size() / 3);
        std::vector<bool> delete_flag(num_points, false);
        std::vector<std::array<double, 3>> insert_points;

        insert_points.reserve(edges.size());

        for (const auto& [edge, count] : edges) {
            int a = edge.first;
            int b = edge.second;

            // use mapped coordinates to measure current edge length
            std::span<double, 3> ma(r.data() + a * 3, 3);
            std::span<double, 3> mb(r.data() + b * 3, 3);

            double dx = ma[0] - mb[0];
            double dy = ma[1] - mb[1];
            double dz = ma[2] - mb[2];
            double edge_length = std::sqrt(dx * dx + dy * dy + dz * dz);

            // 1) Edge detection: small boundary edges merging (batch)
            if (edge_length < min_edge_length) {
                if (a > b) std::swap(a, b);

                if (delete_flag[a] || delete_flag[b]) {
                    // one of the endpoints is already marked for deletion
                    continue;
                }

                std::span<double, 3> va(vertices_sphere.data() + a * 3, 3);
                std::span<double, 3> vb(vertices_sphere.data() + b * 3, 3);

                std::array<double, 3> new_point{ (va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0, (va[2] + vb[2]) / 2.0 };
                double norm = std::sqrt(new_point[0] * new_point[0] + new_point[1] * new_point[1] + new_point[2] * new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }

                delete_flag[a] = true;
                delete_flag[b] = true;
                insert_points.push_back(new_point);
                any_change = true;
                continue;
            }

            // 2) Edge detection: large boundary edges splitting (batch)
            if (edge_length > max_edge_length) {
                if (a > b) std::swap(a, b);

                if (delete_flag[a] || delete_flag[b]) {
                    // one of the endpoints is already marked for deletion
                    continue;
                }

                std::span<double, 3> va(vertices_sphere.data() + a * 3, 3);
                std::span<double, 3> vb(vertices_sphere.data() + b * 3, 3);

                std::array<double, 3> new_point{ (va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0, (va[2] + vb[2]) / 2.0 };
                double norm = std::sqrt(new_point[0] * new_point[0] + new_point[1] * new_point[1] + new_point[2] * new_point[2]);
                if (norm > 0) { new_point[0] /= norm; new_point[1] /= norm; new_point[2] /= norm; }

                insert_points.push_back(new_point);
                any_change = true;
                continue;
            }
        }

        
        // if nothing changed this round, we're done
        if (!any_change) break;
        else {
            std::vector<double> new_vertices;
            new_vertices.reserve(vertices_sphere.size() + insert_points.size() * 3);

            for (int i = 0; i < num_points; i++) {
                if (delete_flag[i]) continue;
                new_vertices.push_back(vertices_sphere[i * 3 + 0]);
                new_vertices.push_back(vertices_sphere[i * 3 + 1]);
                new_vertices.push_back(vertices_sphere[i * 3 + 2]);
            }

            for (const auto& p : insert_points) {
                new_vertices.push_back(p[0]);
                new_vertices.push_back(p[1]);
                new_vertices.push_back(p[2]);
            }

            vertices_sphere = std::move(new_vertices);
        }

    }
    
    // // update stored points each iteration to reflect modifications
    SphereTriangulation triangulator(vertices_sphere);
    triangulator.triangulate();
    faces.clear();
    faces.resize(triangulator.size() * 3);
    triangulator.getTriangleIndices(faces);

    auto r = map_points_batch(vertices_sphere, tree, control_points);

    // refine mesh by edge flipping
    faces = mesh_optimize_by_edge_flipping(r, 3, faces, 100);
    
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

	// rdu layout: [num_points, 2 plane dims, 3 space dims]
	TensorView rdu(std::array<const int, 3>{ num_points, 2, 3}, _rdu);

    std::unordered_map<std::array<int, 4>, double, Array4IntHash> Ldu2_map; // (pt_idx1, dim1, pt_idx2, dim2) -> value

#pragma omp parallel for
    for (int ptidx = 0; ptidx < num_points; ptidx++) {
        for (int iidx = 0; iidx < 3; iidx++) {
			for (int uidx = 0; uidx < 2; uidx++) {
                Ldu[ptidx * 2 + uidx] += Ldr[ptidx * 3 + iidx] * rdu[{ptidx, uidx, iidx}];
            }
        }
    }

	// rdu2 layout: [num_points, 2 plane dims, 2 plane dims, 3 space dims]
	TensorView rdu2(std::array<const int, 4>{ num_points, 2, 2, 3}, _rdu2);

	// Ldu2 += Ldr2 * rdu * rdu + Ldr * rdu2
	// Use thread-local maps to avoid contention, then merge
	std::vector<std::unordered_map<std::array<int, 4>, double, Array4IntHash>> thread_maps;
	int num_threads = 1;
	#pragma omp parallel
	{
		#pragma omp single
		num_threads = omp_get_num_threads();
	}
	thread_maps.resize(num_threads);
	
	// Parallel: Ldr2 contribution
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		auto& local_map = thread_maps[tid];
		
		#pragma omp for
		for (int idx = 0; idx < num_indices_Ldr2; idx++) {
			int v0_idx = Ldr2_indices[idx * 4 + 0];
			int v0_dim = Ldr2_indices[idx * 4 + 1];
			int v1_idx = Ldr2_indices[idx * 4 + 2];
			int v1_dim = Ldr2_indices[idx * 4 + 3];
			double value = Ldr2_values[idx];

			for (int u0idx = 0; u0idx < 2; u0idx++) {
				for (int u1idx = 0; u1idx < 2; u1idx++) {
					local_map[{v0_idx, u0idx, v1_idx, u1idx}] += value * rdu[{v0_idx, u0idx, v0_dim}] * rdu[{v1_idx, u1idx, v1_dim}];
				}
			}
		}
		
		// Parallel: Ldr * rdu2 contribution
		#pragma omp for
		for(int ptidx = 0; ptidx < num_points; ptidx++) {
			for (int idim = 0; idim < 3; idim++) {
				double Ldr_val = Ldr[ptidx * 3 + idim];
				for (int u0idx = 0; u0idx < 2; u0idx++) {
					for (int u1idx = 0; u1idx < 2; u1idx++) {
						local_map[{ptidx, u0idx, ptidx, u1idx}] += Ldr_val * rdu2[{ptidx, u0idx, u1idx, idim}];
					}
				}
			}
		}
	}
	
	// Merge thread-local maps into global map
	for (const auto& local_map : thread_maps) {
		for (const auto& [key, val] : local_map) {
			Ldu2_map[key] += val;
		}
	}

	// copy all entries from map to output vectors
	std::vector<int> Ldu2_indices;
	std::vector<double> Ldu2_values;

	Ldu2_indices.reserve(Ldu2_map.size() * 4);
	Ldu2_values.reserve(Ldu2_map.size());

	// Collect entries and sort by key for deterministic output
	std::vector<std::pair<std::array<int,4>, double>> entries;
	entries.reserve(Ldu2_map.size());
	for (const auto& kv : Ldu2_map) {
		entries.push_back(kv);
	}
	std::sort(entries.begin(), entries.end(),
		[](const auto& a, const auto& b) {
			const auto& ka = a.first; const auto& kb = b.first;
			if (ka[0] != kb[0]) return ka[0] < kb[0];
			if (ka[1] != kb[1]) return ka[1] < kb[1];
			if (ka[2] != kb[2]) return ka[2] < kb[2];
			return ka[3] < kb[3];
		});

	for (const auto& [key, value] : entries) {
        if (std::abs(value) < 1e-8) continue; // skip near-zero entries
		Ldu2_indices.push_back(key[0]);
		Ldu2_indices.push_back(key[1]);
		Ldu2_indices.push_back(key[2]);
		Ldu2_indices.push_back(key[3]);
		Ldu2_values.push_back(value);
		//std::cout<< "indices: ("<< key[0]<<","<< key[1]<<","<< key[2]<<","<< key[3]<<") value: "<< value <<std::endl;
	}

	return { loss, Ldu, Ldu2_indices, Ldu2_values };
}

// COO to CSR sparse matrix conversion (simplified: COO already sorted and merged)
static void coo_to_csr(
    const std::vector<int>& coo_indices,    // [i0, j0, i1, j1, ...] flattened (pt_idx, dim, pt_idx, dim) tuples
    const std::vector<double>& coo_values,  // corresponding values
    int matrix_size,                         // total number of variables (num_points * 2)
    std::vector<int>& csr_row_ptr,          // output: row pointers [0, nnz_row0, nnz_row0+nnz_row1, ...]
    std::vector<int>& csr_col_idx,          // output: column indices
    std::vector<double>& csr_values         // output: values
) {
    int nnz = static_cast<int>(coo_values.size());
    
    // Convert 4-tuple indices to flat row/col indices (can parallelize)
    std::vector<int> rows(nnz);
    csr_col_idx.resize(nnz);
    
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        rows[i] = coo_indices[i * 4 + 0] * 2 + coo_indices[i * 4 + 1];
        csr_col_idx[i] = coo_indices[i * 4 + 2] * 2 + coo_indices[i * 4 + 3];
    }
    
    // Copy values directly (already merged in COO)
    csr_values = coo_values;
    
    // Build row pointers by scanning rows (COO is already sorted by row, then col)
    csr_row_ptr.assign(matrix_size + 1, 0);
    
    int current_row = -1;
    for (int i = 0; i < nnz; i++) {
        // Fill gaps for empty rows
        while (current_row < rows[i]) {
            csr_row_ptr[++current_row] = i;
        }
    }
    // Fill remaining empty rows
    while (current_row < matrix_size) {
        csr_row_ptr[++current_row] = nnz;
    }
}

// Sparse matrix-vector multiplication: y = A * x (CSR format)
static void sparse_matvec(
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_idx,
    const std::vector<double>& csr_values,
    const std::vector<double>& x,
    std::vector<double>& y
) {
    int n = static_cast<int>(x.size());
    y.assign(n, 0.0);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++) {
            y[i] += csr_values[j] * x[csr_col_idx[j]];
        }
    }
}

// Conjugate Gradient solver for symmetric positive definite systems (optimized)
// Solves: A * x = b, where A is given in CSR format
static bool conjugate_gradient(
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_idx,
    const std::vector<double>& csr_values,
    const std::vector<double>& b,
    std::vector<double>& x,
    int max_iter = 1000,
    double tol = 1e-6
) {
    int n = static_cast<int>(b.size());
    x.assign(n, 0.0);
    
    std::vector<double> r(n);
    std::vector<double> p(n);
    std::vector<double> Ap(n);
    
    // r = b - A*x (initially r = b since x = 0)
    r = b;
    p = r;
    
    // Compute initial residual norm with parallel reduction
    double rs_old = 0.0;
    #pragma omp parallel for reduction(+:rs_old)
    for (int i = 0; i < n; i++) {
        rs_old += r[i] * r[i];
    }
    
    if (std::sqrt(rs_old) < tol) {
        return true; // Already solved
    }
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        sparse_matvec(csr_row_ptr, csr_col_idx, csr_values, p, Ap);
        
        // alpha = rs_old / (p^T * Ap) - parallel reduction
        double pAp = 0.0;
        #pragma omp parallel for reduction(+:pAp)
        for (int i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }
        
        if (std::abs(pAp) < 1e-20) {
            return false; // Numerical breakdown
        }
        
        double alpha = rs_old / pAp;
        
        // Fused update: x += alpha*p, r -= alpha*Ap, compute rs_new
        double rs_new = 0.0;
        #pragma omp parallel for reduction(+:rs_new)
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rs_new += r[i] * r[i];  // fuse residual computation
        }
        
        if (std::sqrt(rs_new) < tol) {
            return true; // Converged
        }
        
        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;
        
        // p = r + beta * p
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        rs_old = rs_new;

        //if (iter % 50 == 0) {
        //    std::cout << "      CG iter " << iter << ": residual = " << std::sqrt(rs_old) << std::endl;
        //}
    }
    
    return false; // Did not converge
}

// Compute loss only (without derivatives)
static double compute_loss_only(
    std::vector<double>& vertices_sphere,
    SpaceTree& tree,
    const std::span<const double> control_points,
    std::span<const int> edges,
    bool north_pole
) {
    auto r = map_points_batch(vertices_sphere, tree, control_points, north_pole);
    double loss = closure_edge_length_derivative0(std::span<double>(r.data(), r.size()), 3, edges, order);
    return loss;
}

// Evaluate volume of the mesh
static double evaluate_volume(
    std::span<const double> vertices_sphere,
    std::span<const int> faces){
    
    double volume = 0.0;
    int num_faces = static_cast<int>(faces.size()) / 3;
    for (int fidx = 0; fidx < num_faces; fidx++) {
        int v0_idx = faces[fidx * 3 + 0];
        int v1_idx = faces[fidx * 3 + 1];
        int v2_idx = faces[fidx * 3 + 2];

        std::span<const double, 3> v0(vertices_sphere.data() + v0_idx * 3, 3);
        std::span<const double, 3> v1(vertices_sphere.data() + v1_idx * 3, 3);
        std::span<const double, 3> v2(vertices_sphere.data() + v2_idx * 3, 3);

        // Volume contribution from tetrahedron formed by triangle and origin
        double vol = (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) -
                      v0[1] * (v1[0] * v2[2] - v1[2] * v2[0]) +
                      v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])) / 6.0;
        volume += vol;
    }
    return std::abs(volume);
}

// Backtracking line search with volume preservation
// Returns: (step_accepted, final_step_size, final_loss)
static std::tuple<bool, double, double> backtracking_line_search(
    std::vector<double>& vertices_sphere,
    const std::vector<double>& vertices_backup,
    const std::vector<double>& delta_uv,
    double current_loss,
    double initial_volume,
    std::span<int> faces,
    SpaceTree& tree,
    const std::span<const double> control_points,
    std::span<const int> edges,
    bool north_pole,
    int max_backtracks = 30,
    double beta = 0.5,
    bool check_volume = true
) {
    int num_points = static_cast<int>(vertices_sphere.size()) / 3;
    double step_size = 1.0;
    
    for (int bt = 0; bt < max_backtracks; bt++) {
        // Apply update: convert delta_uv to sphere coordinates
        vertices_sphere = vertices_backup; // restore
        
        for (int ptidx = 0; ptidx < num_points; ptidx++) {
            // Get current sphere position
            std::array<double, 3> sphere_pos = {
                vertices_sphere[ptidx * 3 + 0],
                vertices_sphere[ptidx * 3 + 1],
                vertices_sphere[ptidx * 3 + 2]
            };
            
            // Project to plane
            auto uv_pos = stereographicProjection3_2(
                std::span<const double, 3>(sphere_pos.data(), 3),
                north_pole
            );
            
            // Apply step in UV space
            uv_pos[0] += step_size * delta_uv[ptidx * 2 + 0];
            uv_pos[1] += step_size * delta_uv[ptidx * 2 + 1];
            
            // Project back to sphere
            auto new_sphere_pos = stereographicProjection2_3(
                std::span<const double, 2>(uv_pos.data(), 2),
                north_pole
            );
            
            // Normalize to unit sphere
            double norm = std::sqrt(
                new_sphere_pos[0] * new_sphere_pos[0] +
                new_sphere_pos[1] * new_sphere_pos[1] +
                new_sphere_pos[2] * new_sphere_pos[2]
            );
            
            if (norm > 1e-10) {
                vertices_sphere[ptidx * 3 + 0] = new_sphere_pos[0] / norm;
                vertices_sphere[ptidx * 3 + 1] = new_sphere_pos[1] / norm;
                vertices_sphere[ptidx * 3 + 2] = new_sphere_pos[2] / norm;
            }
        }
        
        // Compute new loss
        double new_loss = compute_loss_only(vertices_sphere, tree, control_points, edges, north_pole);
        
        // Check volume preservation if requested
        bool volume_ok = true;
        if (check_volume) {
            double current_volume = evaluate_volume(std::span<const double>(vertices_sphere.data(), vertices_sphere.size()), faces);
            double volume_ratio = std::abs(initial_volume - current_volume) / initial_volume;
            volume_ok = (volume_ratio < 0.05);
        }
        
        // Accept step if loss decreased and volume is preserved
        if (new_loss < current_loss && volume_ok) {
            return {true, step_size, new_loss};
        }
        
        // Reduce step size
        step_size *= beta;
    }
    
    return {false, 0.0, current_loss};
}

static std::vector<double> newton_step(
    std::vector<double>& vertices_sphere, 
    SpaceTree& tree, 
    std::span<const double> control_points, 
    std::span<const int> edges, 
    bool north_pole
){

    auto [r, rdu, rdu2] = map_points_batch_derivative2(vertices_sphere, tree, control_points, north_pole);
    auto [loss, Ldu, Ldu2_indices, Ldu2_values] = closure_edge_length_surface2plane_derivative2(
        std::span<double>(r.data(), r.size()),
        std::span<double>(rdu.data(), rdu.size()),
        std::span<double>(rdu2.data(), rdu2.size()),
        edges,
        order
    );
    
    int num_points = static_cast<int>(vertices_sphere.size()) / 3;
    int num_variables = num_points * 2; // each point has 2 UV variables
    
    // Build index_remain: filter points based on z coordinate
    // North pole: keep z < 0.5, South pole: keep z > -0.5
    std::vector<bool> point_keep(num_points, false);
    int num_keep_points = 0;
    
    for (int ptidx = 0; ptidx < num_points; ptidx++) {
        double z = vertices_sphere[ptidx * 3 + 2];
        bool keep = north_pole ? (z < 0.5) : (z > -0.5);
        point_keep[ptidx] = keep;
        if (keep) num_keep_points++;
    }
    
    int num_keep_variables = num_keep_points * 2;
    
    // Build mapping: old variable index -> new variable index
    // Variable index: ptidx * 2 + dim (dim = 0 for u, 1 for v)
    std::vector<int> old_to_new(num_variables, -1);
    int new_var_idx = 0;
    
    for (int ptidx = 0; ptidx < num_points; ptidx++) {
        if (point_keep[ptidx]) {
            old_to_new[ptidx * 2 + 0] = new_var_idx++;
            old_to_new[ptidx * 2 + 1] = new_var_idx++;
        }
    }
    
    // Filter first derivative (gradient)
    std::vector<double> Ldu_filtered(num_keep_variables, 0.0);
    for (int ptidx = 0; ptidx < num_points; ptidx++) {
        if (point_keep[ptidx]) {
            for (int dim = 0; dim < 2; dim++) {
                int old_idx = ptidx * 2 + dim;
                int new_idx = old_to_new[old_idx];
                Ldu_filtered[new_idx] = Ldu[old_idx];
            }
        }
    }
    
    // Filter second derivative (Hessian) and remap indices
    std::vector<int> Ldu2_indices_filtered;
    std::vector<double> Ldu2_values_filtered;
    
    int num_hessian_entries = static_cast<int>(Ldu2_values.size());
    for (int idx = 0; idx < num_hessian_entries; idx++) {
        int pt0 = Ldu2_indices[idx * 4 + 0];
        int dim0 = Ldu2_indices[idx * 4 + 1];
        int pt1 = Ldu2_indices[idx * 4 + 2];
        int dim1 = Ldu2_indices[idx * 4 + 3];
        
        // Keep only if both points are kept
        if (point_keep[pt0] && point_keep[pt1]) {
            int old_var0 = pt0 * 2 + dim0;
            int old_var1 = pt1 * 2 + dim1;
            int new_var0 = old_to_new[old_var0];
            int new_var1 = old_to_new[old_var1];
            
            // Convert to flat indices for CSR conversion
            Ldu2_indices_filtered.push_back(new_var0 / 2); // new point index
            Ldu2_indices_filtered.push_back(new_var0 % 2); // dimension
            Ldu2_indices_filtered.push_back(new_var1 / 2); // new point index
            Ldu2_indices_filtered.push_back(new_var1 % 2); // dimension
            Ldu2_values_filtered.push_back(Ldu2_values[idx]);
        }
    }
    
    // Convert filtered COO to CSR format for Hessian
    std::vector<int> csr_row_ptr, csr_col_idx;
    std::vector<double> csr_values;
    coo_to_csr(Ldu2_indices_filtered, Ldu2_values_filtered, num_keep_variables, csr_row_ptr, csr_col_idx, csr_values);

    
    // Set up RHS: -gradient (using filtered gradient)
    std::vector<double> rhs(num_keep_variables);
    for (int i = 0; i < num_keep_variables; i++) {
        rhs[i] = -Ldu_filtered[i];
    }
    
    // Solve Hessian * delta_uv = -gradient using conjugate gradient
    std::vector<double> delta_uv_filtered;
    bool converged = conjugate_gradient(csr_row_ptr, csr_col_idx, csr_values, rhs, delta_uv_filtered, 10000, 1e-5);

    
    
    // if (!converged) {
    //     std::cout << "    Warning: CG did not converge, using gradient descent step" << std::endl;
    //     // Fallback to gradient descent
    //     delta_uv_filtered = rhs;
    //     double scale = 0.01; // small step size
    //     for (auto& v : delta_uv_filtered) v *= scale;
    // }
    
    // Expand delta_uv back to full size (set filtered-out variables to 0)
    std::vector<double> delta_uv(num_variables, 0.0);
    for (int ptidx = 0; ptidx < num_points; ptidx++) {
        if (point_keep[ptidx]) {
            for (int dim = 0; dim < 2; dim++) {
                int old_idx = ptidx * 2 + dim;
                int new_idx = old_to_new[old_idx];
                delta_uv[old_idx] = delta_uv_filtered[new_idx];
            }
        }
    }

    // Sanitize optimization direction: replace NaN or infinite values with 0.0
    for (auto &v : delta_uv) {
        if (std::isnan(v) || std::isinf(v)) {
            v = 0.0;
        }
    }

    // If Newton direction is obtuse to gradient direction, reverse Newton direction
    double dot_grad = 0.0;
    for (int i = 0; i < num_variables; ++i) {
        dot_grad += delta_uv[i] * Ldu[i];
    }
    if (dot_grad > 0.0) {
        for (auto& v : delta_uv) {
            v = -v;
        }
    }
    
    return delta_uv;
}

void vertice_smoothing(
    std::vector<double>& vertices_sphere,
    std::span<int> faces,
    std::span<const double> control_points,
    SpaceTree& tree
){
	int num_points = static_cast<int>(vertices_sphere.size()) / 3;

	auto edges_map = extractEdgesWithNumber(faces);

	std::vector<int> edges;
	edges.reserve(edges_map.size() * 2);
    for (const auto& [edge, count] : edges_map) {
        edges.push_back(edge.first);
        edges.push_back(edge.second);
	}

	auto knots = tree.get_knots();
	auto thresholds = tree.get_thresholds();

    const int max_newton_iters = 50;
    const double loss_tol = 1e-6;
    const double grad_tol = 1e-2;
    
    double prev_loss = std::numeric_limits<double>::infinity();

    // Initial volume
    double initial_volume = evaluate_volume(std::span<const double>(vertices_sphere.data(), vertices_sphere.size()), faces);
    
    for(int loop = 0; loop < max_newton_iters; loop++){
        bool north_pole = loop % 2 == 0;

        
        
        // Compute newton step (returns delta in UV space)
        auto delta_uv = newton_step(vertices_sphere, tree, control_points, std::span<const int>(edges.data(), edges.size()), north_pole);


        
        // Compute current loss
        double current_loss = compute_loss_only(vertices_sphere, tree, control_points, std::span<const int>(edges.data(), edges.size()), north_pole);
        
        std::cout << "    Newton iter " << loop << ": loss = " << current_loss;
        
        // Check gradient norm for convergence
        double grad_norm = 0.0;
        for (double v : delta_uv) {
            grad_norm += v * v;
        }
        grad_norm = std::sqrt(grad_norm);
        std::cout << ", grad_norm = " << grad_norm;
        
        if (grad_norm < grad_tol) {
            std::cout << " -> Converged (small gradient)" << std::endl;
            break;
        }
        
        // Backtracking line search
        const double beta = 0.5;  // step size reduction factor
        const int max_backtracks = 30;
        
        std::vector<double> vertices_backup = vertices_sphere;
        
        auto [step_accepted, step_size, new_loss] = backtracking_line_search(
            vertices_sphere, vertices_backup, delta_uv, current_loss, initial_volume,
            faces, tree, control_points, std::span<const int>(edges.data(), edges.size()),
            north_pole, max_backtracks, beta, true
        );
        
        if (step_accepted) {
            std::cout << " -> step_size = " << step_size << ", new_loss = " << new_loss << std::endl;
            prev_loss = new_loss;
        } else {
            std::cout << " -> Line search failed, no step taken." << std::endl;
            vertices_sphere = vertices_backup;
            break; // No progress possible
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

    // insert/delete points
    auto faces = insert_delete_points(vertices_sphere, control_points, tree, seed_size);

    std::vector<double> r;

    for(int loop=0;loop<max_iterations;loop++){
        std::cout << "  - Mesh uniforming loop " << loop + 1 << "/" << max_iterations << std::endl;

        // refine mesh by vertex smoothing
        vertice_smoothing(vertices_sphere, faces, control_points, tree);

        // refine mesh by edge flipping
        auto faces_new = insert_delete_points(vertices_sphere, control_points, tree, seed_size);


        bool any_change = false;
        for (int i = 0; i < static_cast<int>(faces.size()); i++) {
            if (faces[i] != faces_new[i]) {
                any_change = true;
                break;
            }
        }
        if (!any_change) {
            std::cout << "    No edge flipping changes, stopping early." << std::endl;
            break;
        }
        faces = std::move(faces_new);
    }

    return {vertices_sphere, faces};
}

} // namespace cpgeo