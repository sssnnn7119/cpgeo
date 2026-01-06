import numpy as np
import torch
from..utils import mesh as _mesh_methods
from ..utils import mlab_visualization as vis
from .base import _Surface_base

class Sphere_Symmetry(_Surface_base):

    def __init__(self, symmetric = [0]):
        super().__init__()
        
        self._curvilinear_flip = False
        """
        if True, the curvilinear coordinates are flipped to the other side of the sphere.
        """

        self.symmetric = symmetric
        """
        - [0] means the surface is not symmetric
        - [1, []] means the surface is axially symmetric\n
            - 0: x-axis, 1: y-axis, 2: z-axis
        """

    def initialize(self):

        if self.symmetric[0] == 1:
            index_remain = self

        super().initialize()

        

    @property
    def knots(self):
        result = self._knots.clone()
        if self.symmetric[0] == 1:
            result = result.reshape([3, 1, -1]).repeat([1, 2, 1])
            result[2, 1, :] *= -1
            result = result.reshape([3, -1])
        return result
    
    @knots.setter
    def knots(self, value):
        delta_knots = value - self._knots
        if self.symmetric[0] == 1:
            delta_knots = delta_knots.reshape([3, 1, -1]).repeat([1, 2, 1])
            delta_knots[2, 1, :] *= -1
            delta_knots = delta_knots.reshape([3, -1])
        self._knots = self.knots + delta_knots

    @property
    def cp_vertices(self):
        result = self._cp_vertices.clone()
        if self.symmetric[0] == 1:
            result = result.reshape([3, 1, -1]).repeat([1, 2, 1])
            result[self.symmetric[1][0], 1, :] *= -1
            result = result.reshape([3, -1])
        return result
    @cp_vertices.setter
    def cp_vertices(self, value):
        delta_cp_vertices = value - self._cp_vertices
        if self.symmetric[0] == 1:
            delta_cp_vertices = delta_cp_vertices.reshape([3, 1, -1]).repeat([1, 2, 1])
            delta_cp_vertices[self.symmetric[1][0], 1, :] *= -1
            delta_cp_vertices = delta_cp_vertices.reshape([3, -1])
        self._cp_vertices = self.cp_vertices + delta_cp_vertices

    @property
    def cp_elements(self):
        result = self._cp_elements.clone()
        if self.symmetric[0] == 1:
            boundary_points_index = _mesh_methods.get_boundary_edges(
                self._cp_elements)[0]
            
            result2 = result + self._cp_vertices.shape[1]
            result_coo1 = torch.stack([boundary_points_index, boundary_points_index.roll(1), boundary_points_index+self._cp_vertices.shape[1]], dim=1)
            result_coo2 = torch.stack([boundary_points_index.roll(1), boundary_points_index+self._cp_vertices.shape[1], boundary_points_index.roll(-1)+self._cp_vertices.shape[1]], dim=1)

            knots = self.knots
            normal_coo1 = torch.cross(knots[:, result_coo1[:, 1]] - knots[:, result_coo1[:, 0]], knots[:, result_coo1[:, 2]] - knots[:, result_coo1[:, 0]])
            index_flip = (normal_coo1*knots[:, result_coo1].mean(dim=2)).sum(dim=0) < 0
            result_coo1[index_flip, 1], result_coo1[index_flip, 2] = result_coo1[index_flip, 2], result_coo1[index_flip, 1]

            normal_coo2 = torch.cross(knots[:, result_coo2[:, 1]] - knots[:, result_coo2[:, 0]], knots[:, result_coo2[:, 2]] - knots[:, result_coo2[:, 0]])
            index_flip = (normal_coo2*knots[:, result_coo2].mean(dim=2)).sum(dim=0) < 0 
            result_coo2[index_flip, 1], result_coo2[index_flip, 2] = result_coo2[index_flip, 2], result_coo2[index_flip, 1]

            result = torch.cat([result, result2, result_coo1, result_coo2], dim=0)
        
        return result

    @cp_elements.setter
    def cp_elements(self, value):
        if self.symmetric[0] == 1:
            ind_remain = ((value < self._cp_vertices.shape[1]).sum(dim=1) == 3)
            value = value[ind_remain]
        self._cp_elements = value

    def _refine_node_position(self,
                              point_now: torch.Tensor,
                              elements: torch.Tensor,
                              order=3):

        def step(point_now):
            design_variables = self.reference_to_curvilinear(point_now)
            index_boundary = torch.where(design_variables.norm(
                dim=0) > 5)[0].tolist() + sum([
                    self.boundary_points_index[i].tolist()
                    for i in range(len(self.boundary_points_index))
                ], [])
            index_remain = torch.tensor(
                list(set(range(point_now.shape[1])) - set(index_boundary)))
            design_variables = _mesh_methods.edge_length_regularization_surf3D(
                design_variables,
                elements,
                index=index_remain,
                mapping23=self.map_c,
                order=order)
            t = 1 / (4 + design_variables[0]**2 + design_variables[1]**2)
            point_now = self.curvilinear_to_reference(design_variables)[0]
            return point_now

        point_now = step(point_now)
        self._curvilinear_flip = True
        point_now = step(point_now)
        self._curvilinear_flip = False

        return point_now

    def curvilinear_to_reference(self, curvilinear_points: torch.Tensor = None,
                                 derivative: int = 0):
        """
        return the partial derivtive to p_0, p_1
            x, xdu, xdu2
        """
        flag = -1 if self._curvilinear_flip else 1

        num_points = curvilinear_points.shape[1]

        x = torch.zeros([3, num_points])
        t = 1 / (4 + curvilinear_points[0]**2 + curvilinear_points[1]**2)
        x[0] = 4 * curvilinear_points[0] * t
        x[1] = 4 * curvilinear_points[1] * t
        x[2] = (1 - 8 * t) * flag

        output = [x]
        if derivative >= 1:

            tdu = torch.zeros([2, num_points])
            tdu[0] = -2 * curvilinear_points[0] * t**2
            tdu[1] = -2 * curvilinear_points[1] * t**2
            xdu = torch.zeros([3, 2, num_points])
            xdu[0, 0] = 4 * t + 4 * curvilinear_points[0] * tdu[0]
            xdu[0, 1] = 4 * curvilinear_points[0] * tdu[1]
            xdu[1, 0] = 4 * curvilinear_points[1] * tdu[0]
            xdu[1, 1] = 4 * t + 4 * curvilinear_points[1] * tdu[1]
            xdu[2, 0] = -8 * tdu[0] * flag
            xdu[2, 1] = -8 * tdu[1] * flag

            output.append(xdu)
        if derivative >= 2:
            tdu2 = torch.zeros([2, 2, num_points])
            tdu2[0, 0] = -2 * t**2 - 4 * curvilinear_points[0] * t * tdu[0]
            tdu2[1, 1] = -2 * t**2 - 4 * curvilinear_points[1] * t * tdu[1]
            tdu2[0, 1] = -4 * curvilinear_points[0] * t * tdu[1]
            tdu2[1, 0] = tdu2[0, 1]

            xdu2 = torch.zeros([3, 2, 2, num_points])
            xdu2[0, 0, 0] = 4 * tdu[0] + 4 * curvilinear_points[0] * tdu2[
                0, 0] + 4 * tdu[0]
            xdu2[0, 0, 1] = 4 * tdu[1] + 4 * curvilinear_points[0] * tdu2[0, 1]
            xdu2[0, 1, 0] = xdu2[0, 0, 1]
            xdu2[0, 1, 1] = 4 * curvilinear_points[0] * tdu2[1, 1]
            xdu2[1, 0, 0] = 4 * curvilinear_points[1] * tdu2[0, 0]
            xdu2[1, 0, 1] = 4 * tdu[0] + 4 * curvilinear_points[1] * tdu2[0, 1]
            xdu2[1, 1, 0] = xdu2[1, 0, 1]
            xdu2[1, 1, 1] = 4 * tdu[1] + 4 * curvilinear_points[1] * tdu2[
                1, 1] + 4 * tdu[1]
            xdu2[2] = -8 * tdu2 * flag

            output.append(xdu2)

        return output

    def reference_to_curvilinear(self, ref_points: torch.Tensor = None):

        flag = -1 if self._curvilinear_flip else 1

        num_points = ref_points.shape[1]

        x = torch.zeros([2, num_points])
        x[0] = 2 * ref_points[0] / (1 - ref_points[2] * flag)
        x[1] = 2 * ref_points[1] / (1 - ref_points[2] * flag)

        ind_inf = torch.where(ref_points[2] == 1)[0]
        x[0, ind_inf] = 0
        x[1, ind_inf] = 0

        return x

    @staticmethod
    def _reposition_knots(knots):
        """
        reposition the knots to the manifold surface
        """
        knots = knots / knots.norm(dim=0)
        return knots

    def _initialize_from_connection(self,
                                    connection: torch.Tensor):

        num_points = connection.unique().shape[0]

        knots = torch.randn([2, num_points])

        import networkx as nx

        # using dijkstra algorithm to split the connection into two parts
        edges = _mesh_methods.get_edges(connection)
        G = nx.Graph()
        G.add_edges_from(edges.cpu().numpy())
        shortest_path = nx.single_source_dijkstra(G, 50)
        keys = torch.tensor(list(shortest_path[0].keys()))
        values = torch.tensor(list(shortest_path[0].values()))
        mid_values = values[round(len(values) / 2)]

        index_half1 = torch.tensor(keys[values <= mid_values])

        connection1_index = (torch.isin(connection,
                                        index_half1).sum(dim=1) == 3)
        connection1 = connection[connection1_index]
        connection2 = connection[~connection1_index]
        edges = _mesh_methods.get_edges(connection)

        boundary_points_index = _mesh_methods.get_boundary_edges(connection1)

        index1_1, index1_2 = _mesh_methods.divide_mesh_by_line(
            connection, boundary_points_index[0])
        index_half1 = index1_1 if index1_1.shape[0] > index1_2.shape[
            0] else index1_2
        boundary_points_index = boundary_points_index[0]

        index_half1_ = torch.tensor(
            list(
                set(index_half1.tolist()) -
                set(boundary_points_index.tolist())))

        phi0 = np.arccos(1 - 2 * index_half1_.shape[0] / num_points)
        r0 = 2 * np.sin(phi0) / (1 - np.cos(phi0))

        theta = torch.arange(0, 2 * np.pi,
                             2 * np.pi / boundary_points_index.shape[0])
        knots[:, boundary_points_index] = r0 * torch.stack(
            [torch.cos(theta), torch.sin(theta)], dim=0)

        knots = _mesh_methods.edge_length_regularization_surf2D_part(
            knots, connection, index_half1_)

        index_extra = torch.where(knots.norm(dim=0) > 0.7 * r0)[0]

        index_half2 = torch.tensor(
            list((set(range(num_points)) - set(index_half1_.tolist())).union(
                set(index_extra.tolist()))))

        knots3 = self.curvilinear_to_reference(knots)[0]
        knots3[2] *= -1
        knots = self.reference_to_curvilinear(knots3)

        # refine the other semi-sphere

        knots = _mesh_methods.edge_length_regularization_surf2D_part(
            knots, connection, index_half2)

        knots3 = self.curvilinear_to_reference(knots)[0]

        self.knots = knots3
        self.boundary_points_index = torch.zeros([0], dtype=torch.int64)

    def mesh_knots(self, knots: torch.Tensor):
        return _mesh_methods.sphere_mesh(knots)
