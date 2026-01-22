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
t1 = time.time()
print("Initialization time:", t1 - t0)

surf.show()
surf.show_control_points()
surf.show_knots()

raise NotImplementedError("Test case under construction.")