#include "pch.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <boost/multiprecision/cpp_bin_float.hpp>

typedef boost::multiprecision::cpp_bin_float_50 lf;

void get_thresholds(double* results, const double* knots, int n, int k) {
    if (n <= k) {
        return;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::vector<lf> distances;
        distances.reserve(n - 1);

        lf x1 = knots[i * 3], y1 = knots[i * 3 + 1], z1 = knots[i * 3 + 2];

        for (int j = 0; j < n; j++) {
            if (i != j) {
                lf x2 = knots[j * 3], y2 = knots[j * 3 + 1], z2 = knots[j * 3 + 2];
                lf dist_squared = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
                distances.push_back(dist_squared);
            }
        }

        // 只对前k个元素排序，更高效
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

        // 第k个最近的点的距离
		results[i] = sqrt(distances[k - 1]).convert_to<double>();
    }

}