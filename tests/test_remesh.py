"""
Test mesh partition and boundary extraction functionality.
"""

import numpy as np
import sys
sys.path.insert(0, 'src/python')

import cpgeo


if __name__ == "__main__":
    
    data = np.load('tests/testdata.npz')

    cps = data['control_points'].T
    faces = data['mesh_elements']


    surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)

    surf.initialize()

    r = surf.map3(surf._knots)

    edges = cpgeo.capi.get_mesh_edges(faces)[:, :2]

    length = ((r[edges[:, 0]] - r[edges[:, 1]])**2).sum(axis=1)**0.5

    assert False
