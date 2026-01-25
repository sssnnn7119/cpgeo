from contextlib import contextmanager
import cpgeo
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import time


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


data = np.load('tests/testdata.npz')
cps = data['control_points'].T
faces = data['mesh_elements']


surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)

t0 = time.time()
surf.initialize()

np.savetxt('knots.txt', surf._knots, delimiter=',')
np.savetxt('control_points.txt', surf.control_points, delimiter=',')

t1 = time.time()
print("Initialization time:", t1 - t0)


query_points_sphere = surf._knots
query_points_plane = surf.reference_to_curvilinear(query_points_sphere)


with timer("Map points init", iterations=100):
    indices_cps, indices_pts, w, wdu, wdu2 = surf.get_weights2(query_points_plane=query_points_plane, derivative=2)

    query_points_u0pos = query_points_plane.copy()
    query_points_u0pos[:, 0] += 1e-6

    # indices_cps_pos, indices_pts_pos, w_pos, wdu_pos, wdu2_pos = surf.get_weights2(query_points_plane=query_points_u0pos, derivative=2)

with timer("Map points", iterations=100):
    r0 = cpgeo.capi.get_mapped_points(indices_cps, indices_pts, w, surf.control_points, query_points_plane.shape[0])
    # r1 = cpgeo.capi.get_mapped_points(indices_cps_pos, indices_pts_pos, w_pos, surf.control_points, query_points_plane.shape[0])

    rdu0 = np.stack([
        cpgeo.capi.get_mapped_points(indices_cps, indices_pts, wdu[0], surf.control_points, query_points_plane.shape[0]),
        cpgeo.capi.get_mapped_points(indices_cps, indices_pts, wdu[1], surf.control_points, query_points_plane.shape[0])
    ], axis=-1)

with timer("Map points derivative2 C++", iterations=100):
    _r, _rdu, _rdu2 = cpgeo.capi.map_points_derivative2(query_points=query_points_sphere,
                                                        tree_handle=surf._space_tree,
                                                        controlpoints=surf.control_points,)


# rdu0_fd = (r1 - r0) / 1e-6

surf.show()
surf.show_control_points()
surf.show_knots()

raise NotImplementedError("Test case under construction.")