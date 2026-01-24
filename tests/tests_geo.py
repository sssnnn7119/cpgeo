import cpgeo
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import time
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



query_points = np.array([[2.0, 0.0]])


indices_cps, indices_pts, w, wdu, wdu2 = surf.get_weights2(query_points_plane=query_points, derivative=2)

query_points_u0pos = query_points.copy()
query_points_u0pos[:, 0] += 1e-6

indices_cps_pos, indices_pts_pos, w_pos, wdu_pos, wdu2_pos = surf.get_weights2(query_points_plane=query_points_u0pos, derivative=2)

r0 = cpgeo.capi.get_mapped_points(indices_cps, indices_pts, w, surf.control_points, query_points.shape[0])
r1 = cpgeo.capi.get_mapped_points(indices_cps_pos, indices_pts_pos, w_pos, surf.control_points, query_points.shape[0])

rdu0 = np.stack([
    cpgeo.capi.get_mapped_points(indices_cps, indices_pts, wdu[0], surf.control_points, query_points.shape[0]),
    cpgeo.capi.get_mapped_points(indices_cps, indices_pts, wdu[1], surf.control_points, query_points.shape[0])
], axis=-1)

rdu0_fd = (r1 - r0) / 1e-6

surf.show()
surf.show_control_points()
surf.show_knots()

raise NotImplementedError("Test case under construction.")