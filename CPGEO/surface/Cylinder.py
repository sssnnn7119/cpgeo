import numpy as np
import torch
from..utils import mesh as _mesh_methods
from..utils import sparse as _sparse_methods
from .base import _Surface_base
import networkx as _nx


class Cylinder(_Surface_base):
    """
    A class to represent a cylinder surface.
    curvilinear coordinates: theta, z
    
    """
    def _initialize_from_connection(self,
                                    connection: torch.Tensor):

        def equilateral_triangle_mesh(faces: torch.Tensor,
                                      cut_line: torch.Tensor) -> torch.Tensor:
            """
            generate a triangular mesh with equilateral triangles
            
            Args:
                vertices (torch.Tensor): The vertices of the mesh (3 * n).
                faces (torch.Tensor): The faces of the mesh (m x 3).
            
            Returns:
                torch.Tensor: The 2D parameterization of the vertices (2 * n).
            """

            unique_index = torch.arange(connection.max() + 1)

            n = unique_index.shape[0]
            m = faces.shape[0]

            design_variables = torch.randn([2 * unique_index.shape[0] - 2])

            index_remain_list = torch.tensor(
                list(
                    set(unique_index.tolist()) -
                    set(faces[0][:1].tolist())))

            index_remain_list = index_remain_list.sort().values

            def get_uv(design_variables):
                u = torch.zeros([n * 2])
                v = torch.zeros([n * 2])
                u[faces[0][0]] = 0
                v[faces[0][0]] = 0

                u[index_remain_list] = design_variables.reshape([2, -1])[0]
                v[index_remain_list] = design_variables.reshape([2, -1])[1]

                u[index_remain_list + n] = u[index_remain_list] + 2 * np.pi
                v[index_remain_list + n] = v[index_remain_list]

                return u, v

            def closure(design_variables: torch.Tensor, derivative=0):
                # design_variables = design_variables.detach().clone().requires_grad_(True)
                u, v = get_uv(design_variables)

                uf = u[faces]
                vf = v[faces]

                A = (vf[:, 0] - vf[:, 1] + 1 / np.sqrt(3) *
                     (2 * uf[:, 2] - uf[:, 0] - uf[:, 1]))
                B = (vf[:, 1] - vf[:, 2] + 1 / np.sqrt(3) *
                     (2 * uf[:, 0] - uf[:, 1] - uf[:, 2]))
                C = (vf[:, 2] - vf[:, 0] + 1 / np.sqrt(3) *
                     (2 * uf[:, 1] - uf[:, 2] - uf[:, 0]))
                D = (uf[:, 1] - uf[:, 0] + 1 / np.sqrt(3) *
                     (2 * vf[:, 2] - vf[:, 0] - vf[:, 1]))
                E = (uf[:, 2] - uf[:, 1] + 1 / np.sqrt(3) *
                     (2 * vf[:, 0] - vf[:, 1] - vf[:, 2]))
                F = (uf[:, 0] - uf[:, 2] + 1 / np.sqrt(3) *
                     (2 * vf[:, 1] - vf[:, 2] - vf[:, 0]))

                loss1 = A**2 + B**2 + C**2 + D**2 + E**2 + F**2

                if derivative == 0:
                    return loss1.sum()

                lduf = torch.zeros_like(uf)
                lduf[:, 0] = \
                    2*A * (-1 / np.sqrt(3)) + \
                    2*B * (2 / np.sqrt(3)) + \
                    2*C * (-1 / np.sqrt(3)) + \
                    2*D * (-1) + \
                    2*F * (1)

                lduf[:, 1] = \
                    2 * A * (-1/np.sqrt(3)) +\
                    2 * B * (-1/np.sqrt(3)) +\
                    2 * C * (2/np.sqrt(3)) +\
                    2 * D * (1) +\
                    2 * E * (-1)
                lduf[:, 2] = (2 * A * (2 / np.sqrt(3)) + 2 * B *
                              (-1 / np.sqrt(3)) + 2 * C * (-1 / np.sqrt(3)) +
                              2 * E * (1) + 2 * F * (-1))

                ldvf = torch.zeros_like(vf)
                ldvf[:, 0] = (2 * A * (1) + 2 * C * (-1) + 2 * D *
                              (-1 / np.sqrt(3)) + 2 * E * (2 / np.sqrt(3)) +
                              2 * F * (-1 / np.sqrt(3)))
                ldvf[:, 1] = (2 * A * (-1) + 2 * B * (1) + 2 * D *
                              (-1 / np.sqrt(3)) + 2 * E * (-1 / np.sqrt(3)) +
                              2 * F * (2 / np.sqrt(3)))
                ldvf[:, 2] = (2 * B * (-1) + 2 * C * (1) + 2 * D *
                              (2 / np.sqrt(3)) + 2 * E * (-1 / np.sqrt(3)) +
                              2 * F * (-1 / np.sqrt(3)))

                ldu = torch.zeros_like(u)
                ldv = torch.zeros_like(v)

                ldu.scatter_add_(0, faces.flatten(), lduf.flatten())
                ldv.scatter_add_(0, faces.flatten(), ldvf.flatten())

                lduv = torch.stack([ldu, ldv], dim=0)

                if derivative == 1:
                    return loss1.sum(), lduv

                lduf_2 = torch.zeros([faces.shape[0], 3, 3],
                                     dtype=torch.float64)
                lduf_2[:, 0, 0] = 8
                lduf_2[:, 0, 1] = -4
                lduf_2[:, 0, 2] = -4
                lduf_2[:, 1, 0] = -4
                lduf_2[:, 1, 1] = 8
                lduf_2[:, 1, 2] = -4
                lduf_2[:, 2, 0] = -4
                lduf_2[:, 2, 1] = -4
                lduf_2[:, 2, 2] = 8

                ldufdvf = torch.zeros([faces.shape[0], 3, 3],
                                      dtype=torch.float64)
                ldufdvf[:, 0, 1] = 12 / np.sqrt(3)
                ldufdvf[:, 0, 2] = -12 / np.sqrt(3)
                ldufdvf[:, 1, 2] = 12 / np.sqrt(3)
                ldufdvf[:, 1, 0] = -12 / np.sqrt(3)
                ldufdvf[:, 2, 0] = 12 / np.sqrt(3)
                ldufdvf[:, 2, 1] = -12 / np.sqrt(3)

                ldvf_2 = torch.zeros([faces.shape[0], 3, 3],
                                     dtype=torch.float64)
                ldvf_2[:, 0, 0] = 8
                ldvf_2[:, 0, 1] = -4
                ldvf_2[:, 0, 2] = -4
                ldvf_2[:, 1, 0] = -4
                ldvf_2[:, 1, 1] = 8
                ldvf_2[:, 1, 2] = -4
                ldvf_2[:, 2, 0] = -4
                ldvf_2[:, 2, 1] = -4
                ldvf_2[:, 2, 2] = 8

                lduv_2_indices1 = torch.stack([
                    torch.zeros([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 3, 1]).repeat([1, 1, 3]),
                    torch.zeros([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 1, 3]).repeat([1, 3, 1]),
                ])

                lduv_2_indices2 = torch.stack([
                    torch.zeros([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 3, 1]).repeat([1, 1, 3]),
                    torch.ones([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 1, 3]).repeat([1, 3, 1]),
                ])

                lduv_2_indices3 = torch.stack([
                    torch.ones([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 3, 1]).repeat([1, 1, 3]),
                    torch.zeros([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 1, 3]).repeat([1, 3, 1]),
                ])

                lduv_2_indices4 = torch.stack([
                    torch.ones([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 3, 1]).repeat([1, 1, 3]),
                    torch.ones([m, 3, 3], dtype=torch.long),
                    faces.reshape([-1, 1, 3]).repeat([1, 3, 1]),
                ])

                lduv_2_indices = torch.cat([
                    lduv_2_indices1.reshape([4, -1]),
                    lduv_2_indices2.reshape([4, -1]),
                    lduv_2_indices3.reshape([4, -1]),
                    lduv_2_indices4.reshape([4, -1])
                ],
                                           dim=1)
                lduv_2_values = torch.cat([
                    lduf_2.flatten(),
                    ldufdvf.flatten(), -ldufdvf.flatten(),
                    ldvf_2.flatten()
                ],
                                          dim=0)

                lduv_2 = torch.sparse_coo_tensor(lduv_2_indices, lduv_2_values, size=[2, 2*n]*2)

                return loss1.sum(), lduv, lduv_2.coalesce()

            for _ in range(2):
                # design_variables.requires_grad_(True)
                loss, lduv, lduv_2 = closure(design_variables, derivative=2)

                lduv = lduv[:, :n] + lduv[:, n:]
                lduv = lduv[:, index_remain_list].flatten()

                lduv_2_indices = lduv_2.indices()
                lduv_2_values = lduv_2.values()
                
                lduv_2_indices[1] %= n
                lduv_2_indices[3] %= n
                
                lduv_2 = torch.sparse_coo_tensor(lduv_2_indices, lduv_2_values)

                lduv_2 = lduv_2.index_select(1,
                                             index_remain_list).index_select(
                                                 3, index_remain_list)
                lduv_2 = _sparse_methods._sparse_reshape(
                    lduv_2, 2 * [2 * lduv_2.shape[1]]).coalesce()

                duv = _sparse_methods._conjugate_gradient(lduv_2.indices(),
                                                          lduv_2.values(),
                                                          -lduv.flatten(),
                                                          tol=1e-10)

                # print(loss)
                if duv.norm() < 1e-7:
                    break

                design_variables += duv
            # print(loss)
            result = torch.stack(get_uv(design_variables), dim=0)
            return result[:, :n]
        

        num_points = connection.unique().shape[0]
        self.boundary_points_index = _mesh_methods.get_boundary_edges(
            connection)

        edges = _mesh_methods.get_edges(connection)
        G = _nx.Graph()
        G.add_edges_from(edges.cpu().numpy())
        shortest_path = _nx.single_source_dijkstra(
            G, self.boundary_points_index[0][0].item())
        keys = torch.tensor(list(shortest_path[0].keys()))
        ind_other_boundary = torch.where(
            torch.isin(keys, self.boundary_points_index[1]))[0]
        ind_other_boundary = keys[ind_other_boundary[0]].item()
        route = torch.tensor(shortest_path[1][ind_other_boundary])

        # cut the cylinder to the rectangle
        index = torch.zeros([connection.shape[0]], dtype=torch.bool)
        for i in range(len(route) - 1):
            index = index | \
                    ((connection[:, 0] == route[i]) & (connection[:, 1] == route[i + 1])) | \
                    ((connection[:, 1] == route[i]) & (connection[:, 2] == route[i + 1])) | \
                    ((connection[:, 2] == route[i]) & (connection[:, 0] == route[i + 1]))

        faces_connection = connection[index]
        pt_index = torch.unique(faces_connection.flatten())
        ind1 = torch.isin(connection, pt_index).sum(1) == 3
        ind2 = (torch.isin(connection, pt_index).sum(1) == 2) & (torch.isin(
            connection, route).sum(1) == 1)

        boundary_elements = connection[ind1 | ind2]
        other_elements = connection[~(ind1 | ind2)]
        boundary_elements[torch.isin(boundary_elements, route)] += num_points

        self.cut_line = route
        _cp_elements = torch.cat([boundary_elements, other_elements],
                                      dim=0)
        self.cp_elements = _cp_elements % num_points

        knots = equilateral_triangle_mesh(_cp_elements,
                                               self.cut_line)
        knots = torch.cat([knots, torch.zeros([1, num_points])],
                                 dim=0)
        
        self.knots = self.curvilinear_to_reference(knots)[0]

        # print(route)

    @staticmethod
    def _reposition_knots(knots):
        knots_ = knots.clone()
        radius = knots_[:2].norm(dim=0)
        knots_[0] = knots_[0] / radius
        knots_[1] = knots_[1] / radius
        return knots_
    

    def curvilinear_to_reference(self, curvilinear_points = None, derivative = 0):
        
        ref_points = torch.zeros([3, curvilinear_points.shape[1]])
        theta = curvilinear_points[0] % (2 * np.pi)
        ref_points[0] = torch.cos(theta)
        ref_points[1] = torch.sin(theta)
        ref_points[2] = curvilinear_points[1]
        
        output = [ref_points]

        if derivative >= 1:
            xdu = torch.zeros([3, 2, curvilinear_points.shape[1]])
            xdu[0, 0] = -torch.sin(theta)
            xdu[1, 0] = torch.cos(theta)
            xdu[2, 1] = 1
            output.append(xdu)
        if derivative >= 2:
            xdu2 = torch.zeros([3, 2, 2, curvilinear_points.shape[1]])
            xdu2[0, 0, 0] = -torch.cos(theta)
            xdu2[1, 0, 0] = -torch.sin(theta)
            output.append(xdu2)
            
        return output
 
    def reference_to_curvilinear(self, reference_points = None):
        
        cur_points = torch.zeros([2, reference_points.shape[1]])
        cur_points[0] = torch.atan2(reference_points[1], reference_points[0])
        cur_points[1] = reference_points[2]
        
        return cur_points