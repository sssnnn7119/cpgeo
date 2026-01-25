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

def test_space_tree_query_and_weight_computation():
    """测试空间树查询和权重计算的C API接口"""



    num_knots = 10
    knot_limit = 2.0

    knots = np.stack(np.meshgrid(np.linspace(-knot_limit, knot_limit, num_knots), np.linspace(-knot_limit, knot_limit, num_knots), np.linspace(-knot_limit, knot_limit, num_knots), indexing='ij'), axis=-1).astype(np.float64).reshape(-1, 3)
    threshold = 1.2 * np.ones((knots.shape[0],), dtype=np.float64)

    knots[:, 0] += 0.01
    num_repeat = 1
    points = np.array([[1.0, 0.0, 0.0]], dtype=np.float64).repeat(num_repeat, axis=0)
    points_plane = np.array([[2., 0.]], dtype=np.float64).repeat(num_repeat, axis=0)


    with timer("Space tree creation"):
        tree = cpgeo.capi.space_tree_create(knots=knots, thresholds=threshold)

    with timer("Space tree query", iterations=1):
        indices_cps, indices_pts = cpgeo.capi.get_space_tree_query(tree, query_points_sphere=points)


    with timer("Weight computation", iterations=100):
        for _ in range(100):
            w, wdu, wdu_2 = cpgeo.capi.get_weights_derivative2(indices_cps=indices_cps,
                                                            indices_pts=indices_pts,
                                                            knots=knots,
                                                            thresholds=threshold,
                                                            query_points=points_plane)

    # verify direct get_weights matches derivative-based weights
    w_direct = cpgeo.capi.get_weights(indices_cps=indices_cps,
                                      indices_pts=indices_pts,
                                      knots=knots,
                                      thresholds=threshold,
                                      query_points=points_plane)
    assert np.allclose(w_direct, w), f"get_weights mismatch with derivative: {w_direct} vs {w}"

    controlpoints = knots.copy()
    controlpoints[:, 2] += 10.

    with timer("Mapped points computation C++", iterations=1000):
        
        mapped_points_cpp = cpgeo.capi.get_mapped_points(indices_cps, indices_pts, w, controlpoints, points.shape[0])


    print("Mapped points C++:\n", mapped_points_cpp)


    # raise NotImplementedError("The user requested no output.")


def edges():
    faces = np.array([[0, 1, 2],
                      [2, 3, 0],
                        [0, 2, 4]], dtype=np.int32)
    
    with timer("CAPI get_mesh_edges"):
        edges = capi.get_mesh_edges(faces)

    print("faces:\n", faces)
    print("edges:\n", edges)


if __name__ == "__main__":
    print("\n\n=================================== Testing begin ===================================\n")
    test_space_tree_query_and_weight_computation()
    print("\n=================================== Edges test ===================================\n\n")