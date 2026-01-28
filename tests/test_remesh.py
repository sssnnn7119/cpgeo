"""
Test mesh partition and boundary extraction functionality.
"""

import numpy as np
import sys
sys.path.insert(0, 'src/python')

import cpgeo
import os


if __name__ == "__main__":
    
    # data = np.load('tests/testdata.npz')

    # cps = data['control_points'].T
    # faces = data['mesh_elements']
    # surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)
    # surf.initialize()

    surf = cpgeo.CPGEO.load('tests/Surface-1_iter-19.npz')
    print(os.getpid())
    surf.show_control_points()

    np.savetxt('src/cpp/tests/control_points.txt', surf.control_points, delimiter=',')
    np.savetxt('src/cpp/tests/knots.txt', surf._knots, delimiter=',')
    np.savetxt('src/cpp/tests/cp_faces.txt', surf._cp_faces, fmt='%d', delimiter=',')

    surf.initialize()

    surf.refine_surface(seed_size=1.0, max_iterations=5)

    surf.show_control_points()

    assert False
