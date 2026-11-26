#pragma once
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace cpgeo {

// Solve A*x = b from a zero initial guess. tol is an absolute residual norm.
inline bool conjugate_gradient(
    const std::vector<int>& csr_row_ptr, const std::vector<int>& csr_col_idx,
    const std::vector<double>& csr_values, const std::vector<double>& b,
    std::vector<double>& x, int max_iter = 1000, double tol = 1e-6,
    double curvature_floor = 1e-20)
{
    const int n = static_cast<int>(b.size());
    x.assign(n, 0.0);
    std::vector<double> r = b, p = b, Ap(n);

    double rs_old = 0.0;
    for (double value : r) rs_old += value * value;
    if (std::sqrt(rs_old) < tol) return true;

    // Each mesh row is short. Give each worker enough rows to amortize
    // the dot-product barriers, and keep one team for the whole solve.
    constexpr int rows_per_thread = 4096;
    const int threads = std::min(omp_get_max_threads(), std::max(1, n / rows_per_thread));
    double pAp = 0.0, rs_new = 0.0;
    bool converged = false;
    #pragma omp parallel if(threads > 1) num_threads(threads)
    {
        for (int iter = 0; iter < max_iter; ++iter) {
            #pragma omp for schedule(static) reduction(+:pAp)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j) {
                    sum += csr_values[j] * p[csr_col_idx[j]];
                }
                Ap[i] = sum;
                pAp += p[i] * sum;
            }
            if (!std::isfinite(pAp) || pAp == 0.0 || std::abs(pAp) < curvature_floor) break;

            const double alpha = rs_old / pAp;
            #pragma omp for schedule(static) reduction(+:rs_new)
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                rs_new += r[i] * r[i];
            }
            if (std::sqrt(rs_new) < tol) {
                #pragma omp single nowait
                {
                    converged = true;
                }
                break;
            }
            const double beta = rs_new / rs_old;
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];

            #pragma omp single
            {
                rs_old = rs_new;
                rs_new = 0.0;
                pAp = 0.0;
            }
        }
    }
    return converged;
}

} // namespace cpgeo
