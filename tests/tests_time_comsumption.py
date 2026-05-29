
import sys
import os
import time
from contextlib import contextmanager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import cpgeo
import numpy as np
import cpgeo.capi as capi

@contextmanager
def timer(name: str, iterations: int = 10):
    """计时器context manager，用于测量代码执行时间"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f'{name}: {elapsed:.6f} seconds ({iterations} iterations)')

num_knots = 10
knot_limit = 2.0

knots = np.stack(np.meshgrid(np.linspace(-knot_limit, knot_limit, num_knots), np.linspace(-knot_limit, knot_limit, num_knots), np.linspace(-knot_limit, knot_limit, num_knots), indexing='ij'), axis=-1).astype(np.float64).reshape(-1, 3)
threshold = 1.2 * np.ones((knots.shape[0],), dtype=np.float64)

knots[:, 0] += 0.01
num_repeat = 1000
points = np.array([[1.0, 0.0, 0.0]], dtype=np.float64).repeat(num_repeat, axis=0)
points_plane = np.array([[2., 0.]], dtype=np.float64).repeat(num_repeat, axis=0)

t0 = time.time()

tree = cpgeo.capi.space_tree_create(knots=knots, thresholds=threshold)

indices_cps, indices_pts = cpgeo.capi.get_space_tree_query(tree, query_points_sphere=points)
w, wdu, wdu_2 = cpgeo.capi.get_weights_derivative2(indices_cps=indices_cps,
                                                indices_pts=indices_pts,
                                                knots=knots,
                                                thresholds=threshold,
                                                query_points=points_plane)

controlpoints = knots.copy()
controlpoints[:, 2] += 10.


def compute_mapped_points(indices_cps, indices_pts, weights, controlpoints, num_queries):
    """计算映射后的点坐标（Numba JIT加速版本）"""
    mapped_points = np.zeros((num_queries, 3), dtype=np.float64)
    
    for i in range(num_queries):
        start = indices_pts[i]
        end = indices_pts[i + 1]
        
        for j in range(start, end):
            idx_cp = indices_cps[j]
            for k in range(3):
                mapped_points[i, k] += weights[j] * controlpoints[idx_cp, k]
    
    return mapped_points

times_iterations = 100

with timer("Mapping time (C++)"):
    for _ in range(times_iterations):
        mapped_points_cpp = cpgeo.capi.get_mapped_points(indices_cps, indices_pts, w, controlpoints, points.shape[0])

# with timer("Mapping time (Plain Python)"):
#     for _ in range(times_iterations):
#         mapped_points_plain = compute_mapped_points(indices_cps, indices_pts, w, controlpoints, points.shape[0])

# 验证结果一致性
print("\nVerifying results consistency:")
print("C++ vs Plain Python max diff:", np.max(np.abs(mapped_points_cpp - mapped_points_plain)))

consistency_checks = [
    np.allclose(mapped_points_cpp, mapped_points_plain, atol=1e-10)
]


print("All results consistent:", all(consistency_checks))