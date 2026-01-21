import cpgeo

import numpy as np

data = np.load('tests/testdata.npz')

cps = data['control_points'].T
faces = data['mesh_elements']


surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)

surf.initialize()

raise NotImplementedError("Test case under construction.")