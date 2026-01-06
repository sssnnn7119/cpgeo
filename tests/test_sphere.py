import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cpgeo
from cpgeo.utils import mesh, mlab_visualization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')


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

    mlab_visualization.show_quiver3d(r_mean, normal, hold=True)
a = cpgeo.surface.Sphere()

data = np.load('tests/Surface-1_iter-8.npz')
a.cp_vertices = torch.tensor(data['control_points'])
a.cp_elements = torch.tensor(data['mesh_elements'])
a.initialize()

a.reconstruction(1.)

a.uniformly_mesh(1.5)

a.show_P0()
a.show()

show_normal_mesh(a.cp_vertices, a.cp_elements)


print(os.getcwd())
print(os.path.dirname(__file__))

raise Exception('stop')