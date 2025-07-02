import numpy as np
import torch
from ..utils import mesh as _mesh_methods
from .Sphere import Sphere
from .Plane import Plane
from .Cylinder import Cylinder
from .base import _Surface_base
from .Sphere_Symmetry import Sphere_Symmetry
def sphere(num_points: int, radius: float = 1.0):
    n = torch.arange(num_points) + 1
    phi = (np.sqrt(5) - 1) / 2
    zn = (2 * n - 1) / num_points - 1
    xn = torch.sqrt(1 - zn**2) * torch.cos(2 * torch.pi * n * phi)
    yn = torch.sqrt(1 - zn**2) * torch.sin(2 * torch.pi * n * phi)

    r = torch.stack([xn, yn, zn], dim=0)

    coo = _mesh_methods.sphere_mesh(r)

    geo = Sphere()
    geo.cp_vertices = r * radius
    geo.cp_elements = coo

    geo.initialize()
    return geo


def circle(seed_size: float, radius: float = 1.0):

    r = torch.meshgrid(torch.arange(-radius, radius, seed_size),
                       torch.arange(-radius, radius, seed_size))
    r = torch.stack(
        [r[0].flatten(), r[1].flatten(),
         torch.zeros_like(r[0].flatten())],
        dim=0)

    rnorm = torch.norm(r, dim=0)
    r = r[:, rnorm <= radius]
    
    r = r / 1.4
    
    r_new = torch.zeros([3, round(2 * np.pi * radius / seed_size)])
    r_new[0] = radius * torch.cos(torch.linspace(0, 2 * np.pi, r_new.shape[1]))
    r_new[1] = radius * torch.sin(torch.linspace(0, 2 * np.pi, r_new.shape[1]))
    
    r = torch.cat([r, r_new], dim=1)

    r[2] = 1e-10

    coo = _mesh_methods.build_triangular_mesh(r[:2])
    
    for i in range(5):
        r[:2] = _mesh_methods.edge_length_regularization_surf2D(coo, weight_boundary=100.) * radius
        coo = _mesh_methods.refine_triangular_mesh(r.T, coo)
    
    normal = torch.cross(r[:, coo[:, 1]] - r[:, coo[:, 0]], r[:, coo[:, 2]] - r[:, coo[:, 0]])
    index_flip = normal[2] < 0
    coo[index_flip, 1], coo[index_flip, 2] = coo[index_flip, 2], coo[index_flip, 1]

    geo = Plane()
    geo.cp_vertices = r
    geo.cp_elements = coo
    geo.initialize()
    return geo


def cylinder(seed_size: float, radius: float = 1.0, height: float = 1.0):

    num_circle = round(2 * np.pi * radius / seed_size)
    num_height = round(height / seed_size)

    theta = torch.linspace(0, 2 * np.pi, num_circle + 1)[:-1]
    z = torch.linspace(0, height, num_height)

    Theta, Z = torch.meshgrid(theta, z, indexing='ij')
    r = torch.stack([radius * torch.cos(Theta), radius * torch.sin(Theta), Z],
                    dim=0).reshape(3, -1)

    index = torch.arange(r.shape[1]).reshape(num_circle, num_height)

    coo1 = torch.cat([
        index[:, :-1].flatten(), index[:, :-1].roll(-1, dims=0).flatten(),
        index[:, 1:].flatten()
    ],
                     dim=0).reshape(3, -1)
    coo2 = torch.cat([
        index[:, 1:].flatten(), index[:, :-1].roll(-1, dims=0).flatten(),
        index[:, 1:].roll(-1, dims=0).flatten()
    ],
                     dim=0).reshape(3, -1)
    coo = torch.cat([coo1, coo2], dim=1).T

    geo = Cylinder()
    geo.cp_vertices = r
    geo.cp_elements = coo
    geo.initialize()
    return geo

def import_from_stl(file_path: str):
    geo = _Surface_base()
    geo.import_from_stl(file_path)
    return geo