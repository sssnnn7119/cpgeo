import numpy as np
import torch
from..utils import mesh as _mesh_methods
from ..utils import mlab_visualization as vis
from .base import _Surface_base

class Sphere(_Surface_base):

    def __init__(self):
        super().__init__()
        
        self._curvilinear_flip = False
        """
        if True, the curvilinear coordinates are flipped to the other side of the sphere.
        """

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
        thre = mid_values
        while True:
            index_half1 = torch.tensor(keys[values<thre])

            connection1_index = (torch.isin(connection,
                                            index_half1).sum(dim=1) == 3)
            connection1 = connection[connection1_index]
            connection2 = connection[~connection1_index]
            edges = _mesh_methods.get_edges(connection)

            boundary_points_index = _mesh_methods.get_boundary_edges(connection1)
            if len(boundary_points_index) == 1:
                break
            thre -= 1

        if index_half1.shape[0] < num_points / 2:
            index_half1 = torch.tensor(
                list((set(range(num_points)) - set(index_half1.tolist()))))
        
        connection1_index = (torch.isin(connection,
                                        index_half1).sum(dim=1) == 3)
        connection1 = connection[connection1_index]
        connection2 = connection[~connection1_index]
        
        boundary_points_index = boundary_points_index[0]

        index_half1_ = torch.tensor(
            list(
                set(index_half1.tolist()) -
                set(boundary_points_index.tolist())))

        phi0 = np.arccos(1 - 2 * index_half1_.shape[0] / num_points)
        r0 = 2 * np.sin(phi0) / (1 - np.cos(phi0))

        theta = (torch.arange(0, boundary_points_index.shape[0]) / boundary_points_index.shape[0]) * 2 * np.pi
        knots[:, boundary_points_index] = r0 * torch.stack(
            [torch.cos(theta), torch.sin(theta)], dim=0)

        # map from the z+
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

        # refine the x+
        knots3 = self.curvilinear_to_reference(knots)[0]
        knots3 = knots3[[1,2,0]]
        knots = self.reference_to_curvilinear(knots3)
        index_now = torch.where(knots.norm(dim=0) < knots.norm(dim=0).sort().values[int(knots.shape[1]*2/3)])[0]
        knots = _mesh_methods.edge_length_regularization_surf2D_part(
            knots, connection, index_now)
        
        # refine the x-
        knots3 = self.curvilinear_to_reference(knots)[0]
        knots3[2] *= -1
        knots = self.reference_to_curvilinear(knots3)
        index_now = torch.where(knots.norm(dim=0) < knots.norm(dim=0).sort().values[int(knots.shape[1]*2/3)])[0]
        knots = _mesh_methods.edge_length_regularization_surf2D_part(
            knots, connection, index_now)
        
        # refine the y+
        knots3 = self.curvilinear_to_reference(knots)[0]
        knots3[2] *= -1
        knots3 = knots3[[1,2,0]]
        knots = self.reference_to_curvilinear(knots3)
        index_now = torch.where(knots.norm(dim=0) < knots.norm(dim=0).sort().values[int(knots.shape[1]*2/3)])[0]
        knots = _mesh_methods.edge_length_regularization_surf2D_part(
            knots, connection, index_now)
        
        # refine the y-
        knots3 = self.curvilinear_to_reference(knots)[0]
        knots3[2] *= -1
        knots = self.reference_to_curvilinear(knots3)
        index_now = torch.where(knots.norm(dim=0) < knots.norm(dim=0).sort().values[int(knots.shape[1]*2/3)])[0]
        knots = _mesh_methods.edge_length_regularization_surf2D_part(
            knots, connection, index_now)
        
        # finally get the knots
        knots3 = self.curvilinear_to_reference(knots)[0]
        knots3 = knots3[[1,2,0]]
        
        self.knots = knots3
        self.boundary_points_index = torch.zeros([0], dtype=torch.int64)
        

    def mesh_knots(self, knots: torch.Tensor):
        return _mesh_methods.sphere_mesh(knots)
