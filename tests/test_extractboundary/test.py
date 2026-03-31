import cpgeo

import numpy as np

data = np.loadtxt('tests/test_extractboundary/data1.txt')

result = cpgeo.capi.extract_boundary_loops(data)

print("Boundary extraction test completed.")

print("result:", result)