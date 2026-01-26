"""
Test mesh partition and boundary extraction functionality.
"""

import numpy as np
import sys
sys.path.insert(0, 'src/python')

import cpgeo


if __name__ == "__main__":
    
    # data = np.load('tests/testdata.npz')

    # cps = data['control_points'].T
    # faces = data['mesh_elements']


    surf = cpgeo.CPGEO.load('tests/Surface-1_iter-4.npz')

    surf.initialize()

    surf.refine_surface(seed_size=1.0, max_iterations=5)

    surf.show()

    assert False
