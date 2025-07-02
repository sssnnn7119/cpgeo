import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import CPGEO
from CPGEO.utils import mesh, mlab_visualization

import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')

a0 = CPGEO.surface.sphere(100, 4)
a  = CPGEO.surface.Sphere_Symmetry(symmetric=[1, [0]])
a._cp_vertices = a0.cp_vertices
a._cp_elements = a0.cp_elements
a.initialize()


# a = CPGEO.surface.circle(1000, 40)
a.pre_load()
def show_normal():
    result = a.map(derivative=1)
    r = result[0]
    rdu = result[1]

    normal = torch.cross(rdu[:, 0], rdu[:, 1])
    normal = normal / torch.norm(normal, dim=0)

    mlab_visualization.show_quiver3d(r, normal)

def show_normal_mesh(vertices, faces):
    r = vertices[:, faces]

    normal = torch.cross(r[:, :, 1] - r[:, :, 0],
                        r[:, :, 2] - r[:, :, 0])
    normal = normal / torch.norm(normal, dim=0)

    r_mean = torch.mean(r, dim=2)

    mlab_visualization.show_quiver3d(r_mean, normal)

show_normal_mesh(a.cp_vertices, a.cp_elements)
show_normal_mesh(a.knots, a.cp_elements)
a.reconstruction(1,True)
show_normal_mesh(a.cp_vertices, a.cp_elements)
show_normal_mesh(a.knots, a.cp_elements)

a.show_knots()
a.show_P0()
print(os.getcwd())
print(os.path.dirname(__file__))

raise Exception('stop')