import numpy as np
import torch
from..utils import mesh as _mesh_methods
from .base import _Surface_base



class Plane(_Surface_base):

    def __init__(self):
        super().__init__()
        self.boundary_points: torch.Tensor = None

    def initialize(self):

        boundary_points_index = _mesh_methods.get_boundary_edges(
            self.cp_elements)[0]
        # make the index of the boundary points in the first

        new_index = torch.arange(self.cp_vertices.shape[1])
        new_index[boundary_points_index] = torch.arange(
            len(boundary_points_index))
        other_index = torch.tensor(
            list(
                set(range(self.cp_vertices.shape[1])) -
                set(boundary_points_index.tolist()))).sort()[0]
        new_index[other_index] = torch.arange(
            len(boundary_points_index),
            len(boundary_points_index) + len(other_index))

        new_index_inverse = torch.zeros_like(new_index)
        new_index_inverse[new_index] = torch.arange(new_index.shape[0])

        self.cp_vertices = self.cp_vertices[:, new_index_inverse]
        self.cp_elements = new_index[self.cp_elements]

        super().initialize()
        self.boundary_points = self.cp_vertices[:,
                                                self.boundary_points_index[0]]

    def reconstruction(self, seed_size: float, hold_boundary: bool = False):
        super().reconstruction(seed_size)

        if hold_boundary:
            self.cp_vertices[:, :self.boundary_points.
                             shape[1]] = self.boundary_points

    def _refine_node_position(self,
                              point_now: torch.Tensor,
                              elements: torch.Tensor,
                              order=3):

        def step(point_now: torch.Tensor):
            design_variables = self.reference_to_curvilinear(point_now)
            index_boundary = self.boundary_points_index[0].tolist()
            index_remain = torch.tensor(
                list(set(range(point_now.shape[1])) - set(index_boundary)))
            design_variables = _mesh_methods.edge_length_regularization_surf3D(
                design_variables,
                elements,
                index=index_remain,
                mapping23=self.map_c,
                order=order)
            point_now = self.curvilinear_to_reference(design_variables)[0]
            return point_now

        point_now = step(point_now)
        point_now = self._reposition_knots(point_now)

        return point_now

    # @staticmethod
    # def curvilinear_to_reference(curvilinear_points: torch.Tensor = None,
    #                              derivative: int = 0):
    #     """
    #     return the partial derivatives to u, v
    #         x, xdu, xdu2
    #     """
    #     num_points = curvilinear_points.shape[1]
    #     # curvilinear_points.requires_grad_(True)
    #     x = torch.zeros([3, num_points])
    #     x[0] = curvilinear_points[1] * torch.cos(curvilinear_points[0])
    #     x[1] = curvilinear_points[1] * torch.sin(curvilinear_points[0])

    #     output = [x]

    #     if derivative >= 1:
    #         xdu = torch.zeros([3, 2, num_points])
    #         xdu[0,
    #             0] = -curvilinear_points[1] * torch.sin(curvilinear_points[0])
    #         xdu[0, 1] = torch.cos(curvilinear_points[0])
    #         xdu[1,
    #             0] = curvilinear_points[1] * torch.cos(curvilinear_points[0])
    #         xdu[1, 1] = torch.sin(curvilinear_points[0])
    #         output.append(xdu)
    #     if derivative >= 2:
    #         xdu2 = torch.zeros([3, 2, 2, num_points])
    #         xdu2[0, 0,
    #              0] = -curvilinear_points[1] * torch.cos(curvilinear_points[0])
    #         xdu2[0, 0, 1] = -torch.sin(curvilinear_points[0])
    #         xdu2[0, 1, 0] = -torch.sin(curvilinear_points[0])

    #         xdu2[1, 0,
    #              0] = -curvilinear_points[1] * torch.sin(curvilinear_points[0])
    #         xdu2[1, 0, 1] = torch.cos(curvilinear_points[0])
    #         xdu2[1, 1, 0] = torch.cos(curvilinear_points[0])

    #         output.append(xdu2)

    #     return output

    # @staticmethod
    # def reference_to_curvilinear(ref_points: torch.Tensor = None):
    #     """
    #     convert the reference points to the curvilinear space (polar coordinates)
        
    #     Args:
    #         ref_points (torch.Tensor): [3, num_points], the points on the reference space
            
    #     Returns:
    #         torch.Tensor: [2, num_points], the points on the curvilinear space
    #     """

    #     num_points = ref_points.shape[1]

    #     x = torch.zeros([2, num_points])
    #     x[0] = torch.atan2(ref_points[1], ref_points[0])
    #     x[1] = (ref_points[0]**2 + ref_points[1]**2).sqrt()

    #     return x

    # @staticmethod
    # def _reposition_knots(knots):
    #     """
    #     reposition the knots to the manifold surface
    #     """
    #     knots = knots.clone()
    #     knots[2] = 0
    #     ind_norm = knots.norm(dim=0) > 1
    #     knots[:, ind_norm] = knots[:, ind_norm] / knots[:, ind_norm].norm(dim=0)
    #     return knots

    def _initialize_from_connection(self,
                                    connection: torch.Tensor):
        num_points = connection.unique().shape[0]
        boundary_points_index = _mesh_methods.get_boundary_edges(connection)[0]

        knots = _mesh_methods.edge_length_regularization_surf2D(connection, weight_boundary=0.5)
        knots = torch.cat([knots, torch.zeros([1, num_points])], dim=0)
        self.knots = knots
        self.boundary_points_index = [boundary_points_index]
        

    def mesh_knots(self, knots: torch.Tensor):
        knots_ref = knots.clone()
        knots_ref[:, self.boundary_points_index[0]] *= 1.5
        return _mesh_methods.build_triangular_mesh(knots_ref[:2])
