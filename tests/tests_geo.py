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


# data = np.load('tests/testdata.npz')
# cps = data['control_points'].T
# faces = data['mesh_elements']

import pyvista as pv
obj = pv.read('tests/mesh_uniforming_loop_0_step1.obj')
cps = obj.points
faces = obj.faces.reshape((-1, 4))[:, 1:4].astype(np.int32)
edges = cpgeo.capi.get_mesh_edges(faces)

surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)

t0 = time.time()
surf.initialize()



# rdu0_fd = (r1 - r0) / 1e-6

# surf.show()
# surf.show_control_points()
# surf.show_knots()

# Run uniform remeshing and visualize the updated surface
with timer("Uniform remeshing", iterations=1):
    new_vertices, new_faces = surf.uniformly_mesh(seed_size=1.0, max_iterations=10, update_self=False)

print(f"  New vertices: {new_vertices.shape}, New faces: {new_faces.shape}")

# Map the new vertices via the model (to physical/control-point space)
with timer("Mapping new vertices", iterations=1):
    mapped_new = surf.map3(new_vertices)

print(f"  Mapped vertices: {mapped_new.shape}")

# Visualize mapped new surface with PyVista
try:
    import pyvista as pv
    mesh = pv.PolyData(mapped_new, np.hstack([np.full((new_faces.shape[0], 1), 3), new_faces]))
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightcoral', show_edges=True, opacity=1.0)
    plotter.show()
except Exception as e:
    print("PyVista plotting failed:", e)

raise NotImplementedError("Test case under construction.")