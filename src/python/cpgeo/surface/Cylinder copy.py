import numpy as np
import torch
from..utils import mesh as _mesh_methods
from..utils import sparse as _sparse_methods
from .base import _Surface_base
import networkx as _nx


class Cylinder(_Surface_base):

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

            while True:
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

                print(loss)
                if duv.norm() < 1e-7:
                    break

                design_variables += duv
            print(loss)
            result = torch.stack(get_uv(design_variables), dim=0)
            return result[:, :n]
        
        
        
        def equilateral_triangle_mesh__(faces: torch.Tensor,
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
                u = torch.zeros([n + cut_line.shape[0]])
                v = torch.zeros([n + cut_line.shape[0]])
                u[faces[0][0]] = 0
                v[faces[0][0]] = 0

                u[index_remain_list] = design_variables.reshape([2, -1])[0]
                v[index_remain_list] = design_variables.reshape([2, -1])[1]

                u[-cut_line.shape[0]:] = u[cut_line] + 2 * np.pi
                v[-cut_line.shape[0]:] = v[cut_line]

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

                lduv_2 = torch.sparse_coo_tensor(lduv_2_indices, lduv_2_values)

                return loss1.sum(), lduv, lduv_2.coalesce()

            while True:
                # design_variables.requires_grad_(True)
                loss, lduv, lduv_2 = closure(design_variables, derivative=2)

                lduv[:, cut_line] += lduv[:, -cut_line.shape[0]:]

                lduv = lduv[:, index_remain_list].flatten()

                lduv_2_indices = lduv_2.indices()
                lduv_2_values = lduv_2.values()
                
                for i in range(cut_line.shape[0]):
                    lduv_2_indices[1, lduv_2_indices[1] == n + i] = cut_line[i]
                    lduv_2_indices[3, lduv_2_indices[3] == n + i] = cut_line[i]
                
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

                print(loss)
                if duv.norm() < 1e-7:
                    break

                design_variables += duv
            print(loss)
            result = torch.stack(get_uv(design_variables), dim=0)
            return result[:, :-cut_line.shape[0]]

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
        self._cp_elements = torch.cat([boundary_elements, other_elements],
                                      dim=0)
        self.cp_elements = self._cp_elements % num_points

        self.knots = equilateral_triangle_mesh(self._cp_elements,
                                               self.cut_line)
        self.knots = torch.cat([self.knots, torch.zeros([1, num_points])],
                                 dim=0)
        
        self._reload_threshold()

        print(route)

    @staticmethod
    def _distance_measure(knots: torch.Tensor, ref_points: torch.Tensor, indices: torch.Tensor = None):
        
        knots_l = knots.clone()
        knots_l[0] -= 2 * np.pi
        knots_r = knots.clone()
        knots_r[0] += 2 * np.pi
        


        if indices is not None:
            delta, D0 = _Surface_base._distance_measure(knots, ref_points, indices)
            deltal, Dl = _Surface_base._distance_measure(knots_l, ref_points, indices)
            deltar, Dr = _Surface_base._distance_measure(knots_r, ref_points, indices)
            
            
        else:
            delta, D0 = _Surface_base._distance_measure(knots, ref_points)
            deltal, Dl = _Surface_base._distance_measure(knots_l, ref_points)
            deltar, Dr = _Surface_base._distance_measure(knots_r, ref_points)
            
            
        ind = torch.argmin(torch.stack([D0, Dl, Dr], dim=0), dim=0)
        
        delta_result = torch.zeros_like(delta)
        for i in range(3):
            delta_result[i][ind == 0] = delta[i][ind == 0]
            delta_result[i][ind == 1] = deltal[i][ind == 1]
            delta_result[i][ind == 2] = deltar[i][ind == 2]
        
        DD_result = torch.zeros_like(D0)
        DD_result[ind == 0] = D0[ind == 0]
        DD_result[ind == 1] = Dl[ind == 1]
        DD_result[ind == 2] = Dr[ind == 2]
        
        # return delta_result, DD_result
        return delta_result, DD_result
        
    def get_mesh(self, num_tessellation=0, **kwargs):
        r0_extra = self.knots.clone()
        r0_extra[0] += 2 * np.pi
        r0 = torch.cat([self.knots, r0_extra], dim=1)
        
        coo0 = self._cp_elements.clone()

        for i in range(num_tessellation):
            rmid = r0[:, coo0].mean(dim=2)
            rmid = self._reposition_knots(rmid)

            coo0 = torch.cat([
                torch.stack([
                    torch.arange(coo0.shape[0]) + r0.shape[1], coo0[:, 0],
                    coo0[:, 1]
                ],
                            dim=1),
                torch.stack([
                    torch.arange(coo0.shape[0]) + r0.shape[1], coo0[:, 1],
                    coo0[:, 2]
                ],
                            dim=1),
                torch.stack([
                    torch.arange(coo0.shape[0]) + r0.shape[1], coo0[:, 2],
                    coo0[:, 0]
                ],
                            dim=1)
            ],
                             dim=0)

            r0 = torch.cat([r0, rmid], dim=1)

        coo0[(coo0>=self.knots.shape[1]) & (coo0<self.knots.shape[1]*2)] -= self.knots.shape[1]
            
            
        index_remain = torch.tensor(
            list(set(range(r0.shape[1])) - set(range(self.knots.shape[1], self.knots.shape[1] *2))))
        r0 = r0[:, index_remain]
        coo0 = _mesh_methods.adjust_faces(coo0, index_remain)
        
        return r0, coo0

    def _build_trees(self):
        
        self.knots = self.knots.reshape([3, 1, -1]).repeat([1, 3, 1])
        self.knots[:, 1][0] -= 2 * np.pi
        self.knots[:, 2][0] += 2 * np.pi
        self.knots = self.knots.reshape([3, -1])
        
        self.threshold = self.threshold.reshape([1, -1]).repeat([3, 1]).flatten()
        
        super()._build_trees()
        
        self.knots = self.knots[:, :self.knots.shape[1] // 3]
        self.threshold = self.threshold[:self.threshold.shape[0] // 3]
        
    def get_indices(self, points):
        indices = super().get_indices(points)
        indices[0] = indices[0] % self.knots.shape[1]
        return indices
    
    def _reload_threshold(self):
        
        self.knots = self.knots.reshape([3, 1, -1]).repeat([1, 3, 1])
        self.knots[:, 1][0] -= 2 * np.pi
        self.knots[:, 2][0] += 2 * np.pi
        self.knots = self.knots.reshape([3, -1])

        super()._reload_threshold()
        
        self.knots = self.knots[:, :self.knots.shape[1] // 3]
        self.threshold = self.threshold[:self.knots.shape[1]]