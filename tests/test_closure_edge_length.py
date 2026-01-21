import numpy as np

from cpgeo.capi import get_mesh_closure_edge_length_derivative2
from cpgeo import capi


def test_refinement():
    print("=== Test Edge Refinement ===")

    data = np.load("tests/testdata.npz")
    vertices = data["control_points"].T
    faces = data["mesh_elements"]

    edges = capi.get_mesh_edges(faces)[:, :2]

    import time
    t0 = time.time()
    loss, ldr, ldr2_indices, ldr2_values = get_mesh_closure_edge_length_derivative2(vertices, edges)
    t1 = time.time()
    print(f"Computation time: {t1 - t0:.6f} seconds")

    # print(f"Loss: {loss}")
    # print("Ldr:")
    # print(ldr)
    # print("ldr2 in COO format:")

    # import scipy.sparse as sp

    # ldr2_matrix = sp.coo_matrix((ldr2_values, (ldr2_indices[:, 0]*2 + ldr2_indices[:, 1], ldr2_indices[:, 2]*2 + ldr2_indices[:, 3])), shape=(vertices.size, vertices.size))
    # ldr2_matrix = ldr2_matrix.tocsr().tocoo()
    # for i, j, v in zip(ldr2_matrix.row, ldr2_matrix.col, ldr2_matrix.data):
    #     print(f"({i//2}, {i%2}, {j//2}, {j%2}): {v}")
    


if __name__ == "__main__":
    test_refinement()