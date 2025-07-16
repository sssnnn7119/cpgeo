import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mayavi import mlab
import numpy as np
import CPGEO
from CPGEO.utils import mesh, mlab_visualization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')


r0 = 10.
length = 74.
seed_size = 1.2
    
num_points = round(
    (2 * math.pi * r0 * length + 2 * math.pi * r0**2) /
    (1.732 * seed_size**2) * 2)
num_points = max(100, num_points)

threshold = (length / 1.2) / (length + r0 * 2)

n = torch.arange(num_points) + 1
phi = (math.sqrt(5) - 1) / 2
zn = (2 * n - 1) / num_points - 1
xn = torch.sqrt(1 - zn**2) * torch.cos(2 * torch.pi * n * phi)
yn = torch.sqrt(1 - zn**2) * torch.sin(2 * torch.pi * n * phi)

knots = torch.stack([xn, yn, zn], dim=0).T

phi0 = (torch.acos(knots[:, 2]) - math.pi / 2) / (math.pi / 2)

index_lateral = (phi0.abs() < threshold)
index_head = ~index_lateral & (phi0 >= 0)
index_bottom = ~index_lateral & (phi0 <= 0)

theta = torch.atan2(knots[index_lateral, 1], knots[index_lateral,
                                                    0])
phi = phi0[index_lateral]
phi = phi / phi.abs().max()

control_points = torch.zeros([3, num_points])
control_points[0, index_lateral] = r0 * torch.cos(theta)
control_points[1, index_lateral] = r0 * torch.sin(theta)
control_points[2, index_lateral] = length * phi / 2

control_points[0, index_head] = knots[index_head, 0] / math.cos(
    threshold * math.pi / 2) * r0
control_points[1, index_head] = knots[index_head, 1] / math.cos(
    threshold * math.pi / 2) * r0
control_points[2, index_head] = length / 2

control_points[0,
                index_bottom] = knots[index_bottom, 0] / math.cos(
                    threshold * math.pi / 2) * r0
control_points[1,
                index_bottom] = knots[index_bottom, 1] / math.cos(
                    threshold * math.pi / 2) * r0
control_points[2, index_bottom] = -length / 2

control_points[0] *= -1

cpgeo = CPGEO.surface.Sphere()
cpgeo.cp_vertices = control_points
cpgeo.cp_elements = CPGEO.surface._mesh_methods.sphere_mesh(knots.T)

normal = torch.cross(knots[cpgeo.cp_elements[:, 1]] - knots[cpgeo.cp_elements[:, 0]],
                    knots[cpgeo.cp_elements[:, 2]] - knots[cpgeo.cp_elements[:, 0]],
                    dim=1).T
normal_ = torch.cross(cpgeo.cp_vertices.T[cpgeo.cp_elements[:, 1]] - cpgeo.cp_vertices.T[cpgeo.cp_elements[:, 0]],
                    cpgeo.cp_vertices.T[cpgeo.cp_elements[:, 2]] - cpgeo.cp_vertices.T[cpgeo.cp_elements[:, 0]],
                    dim=1).T
triangularcenter = (knots[cpgeo.cp_elements[:, 0]] + knots[cpgeo.cp_elements[:, 1]] +
                    knots[cpgeo.cp_elements[:, 2]]).T / 3
triangularcenter_ = (cpgeo.cp_vertices.T[cpgeo.cp_elements[:, 0]] +
                    cpgeo.cp_vertices.T[cpgeo.cp_elements[:, 1]] +
                    cpgeo.cp_vertices.T[cpgeo.cp_elements[:, 2]]).T / 3
cpgeo.k_neighbors=10
cpgeo.initialize()
cpgeo.pre_load(1)

cpgeo.reconstruction(seed_size=seed_size)

raise Exception('stop')