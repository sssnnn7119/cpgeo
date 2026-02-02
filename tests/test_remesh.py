"""
Test mesh partition and boundary extraction functionality.
"""

import numpy as np
import sys
sys.path.insert(0, 'src/python')

import cpgeo
import os
import time
class Timer:
    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()  # user + system time
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        elapsed = end_time - self.start_time
        print(f"{self.name} took {elapsed:.4f} seconds.")

if __name__ == "__main__":
    
    # data = np.load('tests/testdata.npz')

    # cps = data['control_points'].T
    # faces = data['mesh_elements']
    # surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)
    # surf.initialize()

    surf = cpgeo.CPGEO.load('tests/Surface-1_iter-34.npz')
    print(os.getpid())
    

    np.savetxt('src/cpp/tests/control_points.txt', surf.control_points, delimiter=',')
    np.savetxt('src/cpp/tests/knots.txt', surf._knots, delimiter=',')
    np.savetxt('src/cpp/tests/cp_faces.txt', surf._cp_faces, fmt='%d', delimiter=',')

    surf.initialize()
    surf.show_knots()

    with Timer("Refinement"):
        surf.refine_surface(seed_size=1.0, max_iterations=5)

    surf.show_control_points()
    surf.show()
    surf.show_knots()

    assert False
