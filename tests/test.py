import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cpgeo
from cpgeo.utils import mesh, mlab_visualization

import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')
print(torch.__version__)
f = open('tests/r.txt', 'r')
data = f.read()
f.close()
r = torch.tensor(eval(data))

f = open('tests/elements.txt', 'r')
data = f.read()
f.close()
elements = torch.tensor(eval(data))

# index_remain = torch.where(r[2] > 50)[0]
# elements = mesh.adjust_faces(elements, index_remain)
# r = r[:, index_remain]

r1_ind = mesh.divide_mesh(elements)[0][0]
r1 = r[:, r1_ind]
e1 = mesh.adjust_faces(elements, r1_ind)

a = cpgeo.surface.Plane()
a.cp_vertices = r1
a.cp_elements = e1
a.initialize()
a.pre_load()
a.get_indices_2nd(a._pre_indices)
# a = CPGEO.surface.circle(1000, 40)

a.show()
a.reconstruction(1,True)
a.show()
a.show_knots()
a.show_P0()
print(os.getcwd())
print(os.path.dirname(__file__))

raise Exception('stop')