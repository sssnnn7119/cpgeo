import torch
import sys
import os
import time
from contextlib import contextmanager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import cpgeo
import numpy as np
import numba as nb
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

indices_cps_torch = torch.from_numpy(indices_cps).to(torch.int32)
indices_pts_torch = torch.from_numpy(indices_pts).to(torch.int32)
w_torch = torch.from_numpy(w).to(torch.float64)
controlpoints_torch = torch.from_numpy(controlpoints).to(torch.float64)

weights_per_query = indices_pts_torch[1:] - indices_pts_torch[:-1]
# 使用repeat_interleave创建查询点索引
query_indices_torch = torch.repeat_interleave(torch.arange(points.shape[0], dtype=torch.long), weights_per_query)

# 计算映射后的点（Torch JIT兼容版本）
@torch.jit.script
def compute_mapped_points_jit(indices_cps: torch.Tensor, indices_pts: torch.Tensor,
                            weights: torch.Tensor, controlpoints: torch.Tensor,
                            num_queries: int) -> torch.Tensor:
    """计算映射后的点坐标（Torch JIT兼容版本）"""
    mapped_points = torch.zeros((num_queries, 3), dtype=torch.float64)

    for i in range(num_queries):
        start = int(indices_pts[i])
        end = int(indices_pts[i + 1])

        # 获取该查询点的权重和控制点索引
        w_subset = weights[start:end]
        cp_indices = indices_cps[start:end]

        # 使用Torch的高级索引进行向量化累加
        selected_points = controlpoints[cp_indices]  # (num_weights, 3)
        weighted_points = w_subset.unsqueeze(-1) * selected_points  # (num_weights, 3)
        mapped_points[i] = torch.sum(weighted_points, dim=0)

    return mapped_points

# 计算映射后的点（Torch Compile版本 - PyTorch 2.0+）
@torch.compile
def compute_mapped_points_compile(indices_cps: torch.Tensor, indices_pts: torch.Tensor,
                                    weights: torch.Tensor, controlpoints: torch.Tensor,
                                    num_queries: int) -> torch.Tensor:
    """计算映射后的点坐标（Torch Compile版本）"""
    mapped_points = torch.zeros((num_queries, 3), dtype=weights.dtype)

    for i in range(num_queries):
        start = indices_pts[i]
        end = indices_pts[i + 1]

        # 获取该查询点的权重和控制点索引
        w_subset = weights[start:end]
        cp_indices = indices_cps[start:end]

        # 使用Torch的高级索引进行向量化累加
        selected_points = controlpoints[cp_indices]  # (num_weights, 3)
        weighted_points = w_subset.unsqueeze(-1) * selected_points  # (num_weights, 3)
        mapped_points[i] = torch.sum(weighted_points, dim=0)

    return mapped_points

# 计算映射后的点（完全向量化Torch版本）
def compute_mapped_points_torch_vectorized(indices_cps: torch.Tensor, indices_pts: torch.Tensor,
                                        weights: torch.Tensor, controlpoints: torch.Tensor,
                                        num_queries: int) -> torch.Tensor:
    """计算映射后的点坐标（完全向量化Torch版本）"""
    # 创建查询点索引数组（向量化版本）
    # 计算每个查询点的权重数量
    

    # 扩展权重和控制点
    weights_expanded = weights.unsqueeze(-1).expand(-1, 3)  # (num_weights, 3)
    selected_points = controlpoints[indices_cps]  # (num_weights, 3)
    weighted_contributions = weights_expanded * selected_points  # (num_weights, 3)

    # 使用scatter_add累加到结果
    mapped_points = torch.zeros((num_queries, 3), dtype=torch.float64)
    mapped_points.scatter_add_(0, query_indices_torch.unsqueeze(-1).expand(-1, 3), weighted_contributions)

    return mapped_points

@nb.jit(nopython=True, parallel=True)
def compute_mapped_points_numba(indices_cps, indices_pts, weights, controlpoints, num_queries):
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

times_iterations = 50

with timer("Mapping time (Torch JIT)"):
    for _ in range(times_iterations):
        mapped_points = compute_mapped_points_jit(indices_cps_torch, indices_pts_torch, w_torch, controlpoints_torch, points.shape[0])

compute_mapped_points_compile(indices_cps_torch, indices_pts_torch, w_torch, controlpoints_torch, points.shape[0])  # 预热编译
with timer("Mapping time (Torch Compile)"):
    for _ in range(times_iterations):
        mapped_points_compile = compute_mapped_points_compile(indices_cps_torch, indices_pts_torch, w_torch, controlpoints_torch, points.shape[0])

with timer("Mapping time (Torch Vectorized)"):
    for _ in range(times_iterations):
        mapped_points_torch_vec = compute_mapped_points_torch_vectorized(indices_cps_torch, indices_pts_torch, w_torch, controlpoints_torch, points.shape[0])

compute_mapped_points_numba(indices_cps, indices_pts, w, controlpoints, points.shape[0])  # 预热JIT编译
with timer("Mapping time (Numba)"):
    for _ in range(times_iterations):
        mapped_points_numba = compute_mapped_points_numba(indices_cps, indices_pts, w, controlpoints, points.shape[0])

with timer("Mapping time (C++)"):
    for _ in range(times_iterations):
        mapped_points_cpp = cpgeo.capi.get_mapped_points(indices_cps, indices_pts, w, controlpoints, points.shape[0])

with timer("Mapping time (Plain Python)"):
    for _ in range(times_iterations):
        mapped_points_plain = compute_mapped_points(indices_cps, indices_pts, w, controlpoints, points.shape[0])

print("Mapped points shape:", mapped_points.shape)
print("First few mapped points:")
print(mapped_points[:5])

# 验证结果一致性
print("\nVerifying results consistency:")
print("Torch JIT vs Torch Vectorized max diff:", torch.max(torch.abs(mapped_points - mapped_points_torch_vec)).item())
print("Torch JIT vs Torch Compile max diff:", torch.max(torch.abs(mapped_points - mapped_points_compile)).item())
print("Torch Vectorized vs Numba max diff:", np.max(np.abs(mapped_points_torch_vec.numpy() - mapped_points_numba)))
print("Numba vs C++ max diff:", np.max(np.abs(mapped_points_numba - mapped_points_cpp)))
print("C++ vs Plain Python max diff:", np.max(np.abs(mapped_points_cpp - mapped_points_plain)))

consistency_checks = [
    torch.allclose(mapped_points, mapped_points_torch_vec, atol=1e-10),
    np.allclose(mapped_points_torch_vec.numpy(), mapped_points_numba, atol=1e-10),
    np.allclose(mapped_points_numba, mapped_points_cpp, atol=1e-10),
    np.allclose(mapped_points_cpp, mapped_points_plain, atol=1e-10)
]

consistency_checks.append(torch.allclose(mapped_points, mapped_points_compile, atol=1e-10))

print("All results consistent:", all(consistency_checks))