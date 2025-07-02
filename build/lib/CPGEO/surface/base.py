from math import cos
import numpy as np
import torch
from ..utils import mesh as _mesh_methods
from ..utils import sparse as _sparse_methods
from ..utils import mlab_visualization as vis
from ..base import _CPGEO_base
from mayavi import mlab

import vtk

class _Surface_base(_CPGEO_base):

    def __init__(self):
        super().__init__()

    def show(self, opacity: float = 1., color: tuple = (40.0 / 255, 120.0 / 255, 181.0 / 255)):

        self._show(opacity, color)
        
        points, coo = self.get_mesh(1)
        r = self.map(points)
        coo = coo.tolist()
        # coo = _mesh_methods.refine_triangular_mesh(r.T, coo).tolist()
        r = r.tolist()
        surface = mlab.pipeline.surface(
            mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo),
            color=(1.0 / 255, 1.0 / 255, 1.0 / 255),
            opacity=opacity)
        surface.actor.property.representation = 'wireframe'
        control_points = self.cp_vertices.cpu().numpy()
        mlab.points3d(control_points[0],
                      control_points[1],
                      control_points[2],
                      scale_factor=0.1,
                      color=(40.0 / 255, 120.0 / 255, 181.0 / 255))

        mlab.show()
    
    def export_stl(self, path: str):
        """
        Export the surface to a stl file
        
        Args:
            path (str): the path of the stl file
        """
        
        points, coo = self.get_mesh(1)
        r = self.map(points)
        coo = _mesh_methods.refine_triangular_mesh(r.T, coo).tolist()
        r = r.tolist()
        
        surface = mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo)
        surface_vtk = surface.outputs[0]._vtk_obj
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetFileName(path + '.stl')
        stlWriter.SetInputConnection(surface_vtk.GetOutputPort())
        stlWriter.Write()
        mlab.close()

    def _show(self, opacity: float = 1., color: tuple = (1.0 / 255, 1.0 / 255, 1.0 / 255)):
        points, coo = self.get_mesh(1)
        r = self.map(points)
        # coo = _mesh_methods.refine_triangular_mesh(r.T, coo).tolist()
        coo = coo.tolist()
        r = r.tolist()
        mlab.triangular_mesh(r[0], r[1], r[2], coo, opacity=opacity, color=color)

    def show_knots(self):
        r = self.knots.tolist()
        coo = self.cp_elements.tolist()
        mlab.triangular_mesh(r[0], r[1], r[2], coo, opacity=1)
        surface = mlab.pipeline.surface(
            mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo),
            color=(1.0 / 255, 1.0 / 255, 1.0 / 255),
            opacity=1)
        surface.actor.property.representation = 'wireframe'

        mlab.show()

    def show_P0(self):
        r = self.cp_vertices.tolist()
        coo = self.cp_elements.tolist()
        mlab.triangular_mesh(r[0], r[1], r[2], coo, opacity=1)
        surface = mlab.pipeline.surface(
            mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo),
            color=(1.0 / 255, 1.0 / 255, 1.0 / 255),
            opacity=1)
        surface.actor.property.representation = 'wireframe'

        mlab.show()

    def geometric_info(self, points: torch.Tensor = None, require: str = 'all'):
        """
        Get the geometric information of the points
        
        Args:
            points (torch.Tensor): [v, num_points] the points coordinates in the curvilinear space
            require (str): the information required\n
                            'all' for all information\n
                            'r' for the points\n
                            'n' for the normal\n
                            'H' for the mean curvature\n
                            'K' for the Gaussian curvature\n
                            'C' for the curvature
            
        Returns:
            tuple[torch.Tensor]: the points, the normal, the first fundamental form, the second fundamental form, the Gaussian curvature \n
            points (torch.Tensor): [3, num_points] the points coordinates in the configuration space \n
            Normal (torch.Tensor): [3, num_points] the normal vector of the surface \n
            I (torch.Tensor): [2, 2, num_points] the first fundamental form \n
            II (torch.Tensor): [2, 2, num_points] the second fundamental form \n
            H (torch.Tensor): [num_points] the mean curvature \n
            K (torch.Tensor): [num_points] the Gaussian curvature \n
            C (torch.Tensor): [num_points] the curvature
        """
        r, rdu, rdu2 = self.map_c(points, derivative=2)
        Normal0 = torch.cross(rdu[:, 1], rdu[:, 0], dim=0)
        Normal = Normal0 / torch.sqrt(torch.sum(Normal0**2, dim=0))

        I = torch.einsum('imp, inp->mnp', rdu, rdu)
        invI = I.permute([2, 0, 1]).inverse().permute([1, 2, 0])
        II = torch.einsum('imnp, ip->mnp', rdu2, Normal)

        detI = I[0, 0] * I[1, 1] - I[0, 1] * I[1, 0]
        detII = II[0, 0] * II[1, 1] - II[0, 1] * II[1, 0]

        H = 0.5 * (invI * II).sum([0, 1])
        K = detII / detI

        C = 4 * H**2 - 2 * K

        if require == 'all':
            return r, Normal0, I, II, H, K, C
        if require == 'r':
            return r
        if require == 'n':
            return Normal0
        if require == 'C':
            return C
        if require == 'H':
            return H
        if require == 'K':
            return K

    
    
    def _refine_node_position(self,
                              point_now: torch.Tensor,
                              elements: torch.Tensor,
                              order=3) -> torch.Tensor:
        raise NotImplementedError

    def initialize(self):
        # self.cp_elements = _mesh_methods.refine_triangular_mesh(
        #     self.cp_vertices.T, self.cp_elements)
        super().initialize()
        
        index_boundary = torch.tensor(
            sum([
                self.boundary_points_index[i].tolist()
                for i in range(len(self.boundary_points_index))
            ], [])).to(torch.int64)
        self.knots_weight[index_boundary] = 3.

    def uniformly_mesh(self, seed_size: float) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        """
        Uniformly mesh the geometry
        
        Args:
            seed_size (float): the size of the seed

        Returns:
            tuple[torch.Tensor,torch.Tensor]: the vertices and faces of the mesh
                contains:
                    - r
                    - connection
        """

        def __I1_init_surface_grids(N):

            grids = self.knots
            elems = self.cp_elements
            while True:

                grids_now = grids

                if abs(grids_now.shape[1] - N) <= 10:
                    break

                r = self.map(grids_now)
                elems = self.mesh_knots(grids_now)
                elems = _mesh_methods.refine_triangular_mesh(r.T, elems)

                index_remain = (elems < grids.shape[1]).all(dim=1)
                elems_now = elems[index_remain]

                area = torch.cross(
                    r[:, elems_now[:, 1]] - r[:, elems_now[:, 0]],
                    r[:, elems_now[:, 2]] - r[:, elems_now[:, 0]],
                    dim=0).norm(dim=0) / 2
                grids_mid = (grids[:, elems_now[:, 0]] +
                             grids[:, elems_now[:, 1]] +
                             grids[:, elems_now[:, 2]]) / 3
                grids_mid = self._reposition_knots(grids_mid)

                if grids_now.shape[1] < N - 10:
                    grids = torch.cat([
                        grids,
                        grids_mid[:,
                                  area.argsort()[round((grids_now.shape[1] -
                                                        N)):]]
                    ],
                                      dim=1)
                elif grids_now.shape[1] > N + 10:
                    edges = _mesh_methods.get_edges(elems_now)
                    ind_delete = sum([
                        self.boundary_points_index[i].tolist()
                        for i in range(len(self.boundary_points_index))
                    ], [])
                    ind_delete = torch.tensor(ind_delete)
                    edges = edges[torch.where(
                        torch.isin(edges, ind_delete).sum(dim=1) == 0)]

                    length = (r[:, edges[:, 0]] -
                              r[:, edges[:, 1]]).norm(dim=0)
                    ind = length.argsort()[:round((grids_now.shape[1] - N))]

                    edges_now = edges[ind]
                    index = 0
                    while True:
                        if index == edges_now.shape[0]:
                            break

                        index_remain = (
                            (edges_now == edges_now[index, 0]) |
                            (edges_now == edges_now[index, 1])).sum(dim=1) == 0
                        index_remain[:index + 1] = True
                        edges_now = edges_now[index_remain]

                        index += 1

                    grids_add = (grids[:, edges_now[:, 0]] +
                                 grids[:, edges_now[:, 1]]) / 2
                    grids_add = grids_add / grids_add.norm(dim=0)

                    index_remain = torch.ones([grids.shape[1]],
                                              dtype=torch.bool)
                    index_remain[edges_now.flatten()] = False

                    grids = torch.cat([grids[:, index_remain], grids_add],
                                      dim=1)

            # if a point's neighbor is all in the fixed points, then this point is deleted

            return grids

        def __I2_refine_mesh_post_process(point_now: torch.Tensor):

            def __area_insert(point_now, r, elements):
                """insert a point in the middle of a triangle if the area is too large"""
                amin = seed_size**2 * 0.4 * 0.5
                amax = seed_size**2 * 0.4 * 3.0

                re0 = r[:, elements[:, 0]]
                re1 = r[:, elements[:, 1]]
                re2 = r[:, elements[:, 2]]

                normal = (re1 - re0).cross(re2 - re0, dim=0)

                area = normal.norm(dim=0) / 2

                ind1 = torch.where(((area > amax)))[0]
                if len(ind1) > 0:

                    point_mid = (point_now[:, elements[ind1, 0]] +
                                 point_now[:, elements[ind1, 1]] +
                                 point_now[:, elements[ind1, 2]]) / 3
                    point_mid = self._reposition_knots(point_mid)

                    point_now = torch.cat([point_now, point_mid], dim=1)

                    return point_now, True
                return point_now, False

            def __area_delete(point_now, r, elements):
                """merge a point in the middle of a triangle if the area is too small"""

                amin = seed_size**2 * 0.4 * 0.5
                amax = seed_size**2 * 0.4 * 3.0

                re0 = r[:, elements[:, 0]]
                re1 = r[:, elements[:, 1]]
                re2 = r[:, elements[:, 2]]

                normal = (re1 - re0).cross(re2 - re0, dim=0)

                area = normal.norm(dim=0) / 2

                ind2 = torch.where((area < amin))[0]
                if len(ind2) > 0:
                    ind2 = ind2[area[ind2].argsort()]
                    elems = elements[ind2]
                    index = 0
                    while True:
                        if index == elems.shape[0]:
                            break

                        index_remain = ((elems == elems[index, 0]) |
                                        (elems == elems[index, 1]) |
                                        (elems == elems[index, 2])).sum(
                                            dim=1) == 0
                        index_remain[:index + 1] = True
                        elems = elems[index_remain]

                        index += 1

                    point_mid = (point_now[:, elems[:, 0]] +
                                 point_now[:, elems[:, 1]] +
                                 point_now[:, elems[:, 2]]) / 3
                    point_mid = self._reposition_knots(point_mid)

                    index_remain = torch.ones([point_now.shape[1]],
                                              dtype=torch.bool)
                    index_remain[elems.flatten()] = False

                    point_now = torch.cat(
                        [point_now[:, index_remain], point_mid], dim=1)

                    return point_now, True
                return point_now, False

            def __angle_insert(point_now: torch.Tensor, r: torch.Tensor, elements):
                """insert a point at the edge when the angle is too large"""
                re0 = r[:, elements[:, 0]]
                re1 = r[:, elements[:, 1]]
                re2 = r[:, elements[:, 2]]

                vec01 = re1 - re0
                vec02 = re2 - re0
                vec12 = re2 - re1

                vec01 = vec01 / vec01.norm(dim=0)
                vec02 = vec02 / vec02.norm(dim=0)
                vec12 = vec12 / vec12.norm(dim=0)

                cosangle01 = (vec02 * vec12).sum(dim=0)
                cosangle12 = (vec01 * vec02).sum(dim=0)
                cosangle02 = -(vec01 * vec12).sum(dim=0)

                ind_insert01 = torch.where((cosangle01 < -0.5))[0]
                ind_insert12 = torch.where((cosangle12 < -0.5))[0]
                ind_insert02 = torch.where((cosangle02 < -0.5))[0]


                if ind_insert01.shape[0] > 0 or ind_insert12.shape[0] > 0 or ind_insert02.shape[0] > 0:
                    # Collect all edges where we need to insert points
                    new_points_list = []

                    if ind_insert01.shape[0] > 0:
                        # Points for edge between vertices 0 and 1
                        points01 = (point_now[:, elements[ind_insert01, 0]] + 
                                    point_now[:, elements[ind_insert01, 1]]) / 2
                        new_points_list.append(points01)

                    if ind_insert12.shape[0] > 0:
                        # Points for edge between vertices 1 and 2
                        points12 = (point_now[:, elements[ind_insert12, 1]] + 
                                    point_now[:, elements[ind_insert12, 2]]) / 2
                        new_points_list.append(points12)

                    if ind_insert02.shape[0] > 0:
                        # Points for edge between vertices 0 and 2
                        points02 = (point_now[:, elements[ind_insert02, 0]] + 
                                    point_now[:, elements[ind_insert02, 2]]) / 2
                        new_points_list.append(points02)

                    # Combine all new points to be inserted
                    if new_points_list:
                        new_points = torch.cat(new_points_list, dim=1)
                    else:
                        # No edges to insert points, return empty tensor as placeholder
                        new_points = torch.zeros([3, 0])

                    # check if new points are not unique
                    distance = (new_points[:, :, None] - point_now[:, None, :]).norm(dim=0)
                    unique_mask = (distance > 1e-6).all(dim=0)


                    new_points = self._reposition_knots(new_points)

                    point_now = torch.cat([point_now, new_points], dim=1)

                    return point_now, True
                return point_now, False

            def __length_delete(point_now: torch.Tensor, r: torch.Tensor, elements: torch.Tensor):
                """delete a point if the length is too small"""

                ratio = 0.7

                re0 = r[:, elements[:, 0]]
                re1 = r[:, elements[:, 1]]
                re2 = r[:, elements[:, 2]]

                vec01 = re1 - re0
                vec02 = re2 - re0
                vec12 = re2 - re1

                len01 = vec01.norm(dim=0)
                len02 = vec02.norm(dim=0)
                len12 = vec12.norm(dim=0)

                ind_delete01 = torch.where((len01 < seed_size * ratio))[0]
                ind_delete12 = torch.where((len12 < seed_size * ratio))[0]
                ind_delete02 = torch.where((len02 < seed_size * ratio))[0]

                if ind_delete01.shape[0] > 0 or ind_delete12.shape[0] > 0 or ind_delete02.shape[0] > 0:
                    
                    delete_length = torch.cat([elements[ind_delete01][:, [0,1]],
                                               elements[ind_delete12][:, [1,2]],
                                               elements[ind_delete02][:, [0,2]]], dim=0)
                    
                    index_flip = delete_length[:, 0] > delete_length[:, 1]
                    delete_length[index_flip, 0], delete_length[index_flip, 1] = \
                        delete_length[index_flip, 1], delete_length[index_flip, 0]
                    
                    delete_length = delete_length.unique(dim=0)

                    new_points = (point_now[:, delete_length[:, 0]] +
                                  point_now[:, delete_length[:, 1]]) / 2
                    new_points = self._reposition_knots(new_points)


                    index_remain = torch.ones([point_now.shape[1]],
                                            dtype=torch.bool)
                    index_remain[delete_length.flatten()] = False
                    point_now = torch.cat([point_now[:, index_remain], new_points],
                                          dim=1)

                    return point_now, True
                return point_now, False



            iteration = 0
            while True:

                iteration += 1


                elements = self.mesh_knots(point_now)

                r = self.map(point_now)

                elements = _mesh_methods.refine_triangular_mesh(nodes=r.T,
                                                                elems=elements)
                
                if iteration > 30:
                    print('Warning: mesh refinement iteration exceeds 30, '
                          'may not converge')
                    break


                index_remain = (elements < point_now.shape[1]).sum(dim=1) == 3
                elements_now = elements[index_remain]

                # point_now, cond = __broken_triangle(point_now, r, elements_now)
                # if cond:
                #     continue

                point_now, cond = __area_insert(point_now, r, elements_now)
                if cond:
                    continue
                point_now, cond = __area_delete(point_now, r, elements_now)
                if cond:
                    continue
                point_now, cond = __angle_insert(point_now, r, elements_now)
                if cond:
                    continue
                point_now, cond = __length_delete(point_now, r, elements_now)
                if cond:
                    continue

                break
            r = self.map(point_now)
            elements = _mesh_methods.refine_triangular_mesh(nodes=r.T,
                                                                elems=elements)
            return point_now, elements

        # surf.pre_load()
        knots = self.knots

        r = self.map(knots)

        rr = torch.zeros([
            self.cp_elements.shape[0],
            3,
            3,
        ])
        rr[:, 0] = r[:, self.cp_elements[:, 0]].T
        rr[:, 1] = r[:, self.cp_elements[:, 1]].T
        rr[:, 2] = r[:, self.cp_elements[:, 2]].T

        area = (torch.cross(rr[:, 1] - rr[:, 0], rr[:, 2] - rr[:, 0],
                            dim=1).norm(dim=1) / 2)

        num_points = round(area.sum().item() / (1.732 * seed_size**2 / 2))
        num_points = max(num_points, 400)

        knots_new = __I1_init_surface_grids(num_points)

        knots_new = knots_new.reshape([3, -1])
        # knots_element_new = surf.knots_element
        knots_element_new = self.mesh_knots(knots_new)
        knots_new_2 = knots_new.clone()
        a = knots_element_new
        # t0 = time.time()
        for i in range(10):
            r = self.map(knots_new_2)
            # a = _mesh_methods.refine_triangular_mesh(nodes=r.T, elems=a)

            knots_new_2 = self._refine_node_position(knots_new_2, a, order=4)

            knots_new, a = __I2_refine_mesh_post_process(knots_new_2)
            
            if knots_new.shape[1] == knots_new_2.shape[1]:
                if (knots_new - knots_new_2).norm() < 1e-10:
                    break
            knots_new_2 = knots_new.clone()

        return knots_new_2, a

    def reconstruction(self, seed_size: float, hold_boundary=False):
        initial_knots, elements = self.uniformly_mesh(seed_size)
        initial_points = self.map(initial_knots)
        self.cp_vertices = initial_points.detach().clone()
        self.cp_elements = elements
        self.initialize()
        num_points = initial_points.shape[1]

        # post process

        indices, rdot = self.get_weights(self.knots_r(), derivative=0)
        r = self.map(self.knots)

        alpha = 1.0
        loss0 = ((r - initial_points)**2).sum()
        loss1 = ((self.cp_vertices - initial_points)**2).sum()
        loss = loss0 + loss1 * alpha
        print()
        print('before refine P0: loss = %e' % (loss))

        ldr = 2 * (r - initial_points)
        ldr_2_values = 2 * torch.ones([3, num_points]).flatten()
        ldr_2_indices = torch.stack([
            torch.arange(0, 3).reshape([3, 1]).repeat(1, num_points),
            torch.arange(0, num_points).reshape([1, num_points]).repeat(3, 1),
            torch.arange(0, 3).reshape([3, 1]).repeat(1, num_points),
            torch.arange(0, num_points).reshape([1, num_points]).repeat(3, 1),
        ],
                                    dim=0).reshape([4, -1])
        ldr_2 = torch.sparse_coo_tensor(ldr_2_indices,
                                        ldr_2_values,
                                        size=[3, num_points, 3, num_points])

        l0dot = _sparse_methods._from_Adr_to_Adot(indices=indices,
                                                  Adr=ldr,
                                                  rdot=rdot,
                                                  numel_output=num_points)

        l1dot = 2 * (self.cp_vertices - initial_points)

        index_boundary = torch.tensor(sum([
            self.boundary_points_index[i].tolist()
            for i in range(len(self.boundary_points_index))
        ], []))
        index_boundary2 = index_boundary.clone()
        
        for i in range(5):
            index_boundary2 = self.cp_elements[torch.where(torch.isin(self.cp_elements, index_boundary2).sum(dim=1) > 0)[0]].unique()

        index_no_boundary = torch.tensor(
            list(set(range(num_points)) - set(index_boundary2.tolist())))

        ldot = l0dot + l1dot * alpha

        ldot = ldot[:, index_no_boundary]

        l0dot_2 = _sparse_methods._from_Sdr_to_Sdot_2(indices=indices,
                                                      Sdr_2=ldr_2,
                                                      rdot=rdot)

        l1dot_2 = ldr_2

        ldot_2 = l0dot_2 + l1dot_2 * alpha

        ldot_2 = ldot_2.index_select(1, index_no_boundary).index_select(
            3, index_no_boundary)

        ldot_2 = _sparse_methods._sparse_reshape(
            ldot_2, 2 * [3 * ldot_2.shape[1]]).coalesce()

        dP = _sparse_methods._conjugate_gradient(ldot_2.indices(),
                                                 ldot_2.values(),
                                                 -ldot.flatten(),
                                                 tol=1e-7,
                                                 max_iter=30000)
        dP.view(-1)[dP.view(-1).isnan()] = 0
        dP = dP.reshape([3, -1])

        self.cp_vertices[:,
                         index_no_boundary] = self.cp_vertices[:,
                                                               index_no_boundary] + dP

        r1 = self.map(self.knots)

        epsilon = torch.zeros([3, 3, 3])
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

        loss0_new = ((r1 - initial_points)**2).sum()
        loss1_new = ((self.cp_vertices - initial_points)**2).sum() * alpha

        print('after refine P0: loss = %e' % (loss0_new + loss1_new * alpha))

        self.cp_elements = _mesh_methods.refine_triangular_mesh(
            self.cp_vertices.T, self.cp_elements)


    def reconstruction2(self, seed_size: float):
        initial_knots, elements = self.uniformly_mesh(seed_size)
        initial_points = self.map(initial_knots)
        self.cp_vertices = initial_points.detach().clone()
        self.cp_elements = elements
        self._initialize_from_connection(elements)
        num_points = initial_points.shape[1]

        # post process

        r0, rdu = self.map(self.knots, derivative=1)
        n0 = torch.cross(rdu[:, 1], rdu[:, 0], dim=0).detach()
        n0 = n0 / torch.sqrt(torch.sum(n0**2, dim=0))

        # self.cp_vertices = torch.zeros_like(self.cp_vertices)
        # self.cp_vertices.requires_grad_(True)

        indices, rdot = self.get_weights(self.knots_r())
        r = self.map(self.knots)

        epsilon = torch.zeros([3, 3, 3])
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

        alpha = 10
        loss0 = ((r - initial_points)**2).sum()
        loss1 = (torch.einsum('ijk, jp, kp->ip', epsilon, r - r0, n0)**2).sum()
        loss = loss0 + loss1 * alpha
        print()
        print('before refine P0: loss = %e' % (loss))

        ldr = 2 * (r - initial_points) + 2 * alpha * torch.einsum(
            'iln, np, ijk, jp, kp->lp', epsilon, n0, epsilon, r - r0, n0)
        ldot = _sparse_methods._from_Adr_to_Adot(indices=indices,
                                                 Adr=ldr,
                                                 rdot=rdot,
                                                 numel_output=num_points)

        ldr_2_values = 2 * torch.eye(3).reshape([3, 3, 1]).repeat(1, 1, num_points) + \
            2*alpha*torch.einsum('iln, np, ijk, jo, kp->lop', epsilon, n0, epsilon, torch.eye(3), n0)

        ldr_2_indices = torch.stack([
            torch.arange(3).reshape([3, 1, 1]).repeat(1, 3, num_points),
            torch.arange(num_points).reshape([1, 1, num_points]).repeat(
                3, 3, 1),
            torch.arange(3).reshape([1, 3, 1]).repeat(3, 1, num_points),
            torch.arange(num_points).reshape([1, 1, num_points]).repeat(
                3, 3, 1),
        ],
                                    dim=0).reshape([4, -1])
        ldr_2 = torch.sparse_coo_tensor(ldr_2_indices,
                                        ldr_2_values.flatten(),
                                        size=[3, num_points, 3, num_points])

        # l1dot = 2 * (self.cp_vertices - initial_points)

        index_no_boundary = torch.tensor(
            list(
                set(range(num_points)) - set(
                    sum([
                        self.boundary_points_index[i].tolist()
                        for i in range(len(self.boundary_points_index))
                    ], []))))

        l0dot_2 = _sparse_methods._from_Sdr_to_Sdot_2(indices=indices,
                                                      Sdr_2=ldr_2,
                                                      rdot=rdot)

        ldot_2 = l0dot_2

        ldot_2 = ldot_2.index_select(1, index_no_boundary).index_select(
            3, index_no_boundary)

        ldot_2 = _sparse_methods._sparse_reshape(
            ldot_2, 2 * [3 * ldot_2.shape[1]]).coalesce()

        ldot = ldot[:, index_no_boundary]

        dP = _sparse_methods._conjugate_gradient(ldot_2.indices(),
                                                 ldot_2.values(),
                                                 -ldot.flatten(),
                                                 tol=1e-7,
                                                 max_iter=30000)

        dP.view(-1)[dP.view(-1).isnan()] = 0
        dP = dP.reshape([3, -1])

        self.cp_vertices[:,
                         index_no_boundary] = self.cp_vertices[:,
                                                               index_no_boundary] + dP

        r1 = self.map(self.knots)

        epsilon = torch.zeros([3, 3, 3])
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

        loss0_new = ((r1 - initial_points)**2).sum()
        loss1_new = (torch.einsum('ijk, jp, kp->ip', epsilon, r1 - r0,
                                  n0)**2).sum()
        loss_new = loss0_new + loss1_new * alpha

        print('after refine P0: loss = %e' % (loss_new))

    def get_mesh(self, num_tessellation=0, **kwargs):
        """
        get the surface mesh of the geometry
        
        Args:
            num_tessellation (int): the number of tessellation
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: the vertices and faces of the mesh
            vertices (torch.Tensor): [3, num_vertices] the vertices coordinates
            faces (torch.Tensor): [num_faces, 3] the faces
        """

        r0 = self.knots.clone()
        coo0 = self.cp_elements.clone()

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

        return r0, coo0

    def curvilinear_to_reference(self, curvilinear_points: torch.Tensor = None,
                                 derivative: int = 0):
        """
        return the partial derivatives to u, v
            x, xdu, xdu2
        """
        num_points = curvilinear_points.shape[1]
        # curvilinear_points.requires_grad_(True)
        x = torch.zeros([3, num_points])
        x[0] = curvilinear_points[0]
        x[1] = curvilinear_points[1]
        x[2] = torch.zeros([num_points])

        output = [x]

        if derivative >= 1:
            xdu = torch.zeros([3, 2, num_points])
            xdu[0, 0] = 1
            xdu[1, 1] = 1
            output.append(xdu)
        if derivative >= 2:
            xdu2 = torch.zeros([3, 2, 2, num_points])
            output.append(xdu2)

        return output
 
    def reference_to_curvilinear(self, ref_points: torch.Tensor = None):
        """
        no need to implement
        
        Args:
            ref_points (torch.Tensor): [3, num_points], the points on the reference space
        
        Returns:
            torch.Tensor: [2, num_points], the points on the curvilinear space
        """

        num_points = ref_points.shape[1]

        x = torch.zeros([2, num_points])
        x[0] = ref_points[0]
        x[1] = ref_points[1]

        return x

    def get_guassion_points(self, num_subdivisions: int, num_gauss_points: int):
        
        u, faces = self.get_mesh(num_subdivisions)
        
        ue = u[:, faces]
        area = torch.cross(u[:, faces[:, 1]] - u[:, faces[:, 0]],
                        u[:, faces[:, 2]] - u[:, faces[:, 0]],
                        dim=0).norm(dim=0) / 2
        
        #-----------------4
        if num_gauss_points == 4:
            gauss_points_tri = torch.tensor([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2],
                                            [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
            gauss_weights_tri = torch.tensor([-27 / 48, 25 / 48, 25 / 48, 25 / 48])

        #-----------------3
        if num_gauss_points == 3:
            gauss_points_tri = torch.tensor([[1 / 6, 1 / 6, 2 / 3],
                                            [1 / 6, 2 / 3, 1 / 6],
                                            [2 / 3, 1 / 6, 1 / 6]])
            gauss_weights_tri = torch.tensor([1 / 3, 1 / 3, 1 / 3])

        #-----------------1
        if num_gauss_points == 1:
            gauss_points_tri = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
            gauss_weights_tri = torch.tensor([1.])


        gauss_weights = torch.einsum('g, e->eg', gauss_weights_tri, area)

        u_gauss = torch.einsum('iem, gm -> ieg', ue,
                            gauss_points_tri)
        
        u_gauss = self._reposition_knots(u_gauss.view([3, -1])).reshape([3, -1])
        
        return u_gauss, gauss_weights.flatten()
