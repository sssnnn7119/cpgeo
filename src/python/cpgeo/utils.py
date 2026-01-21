import numpy as np
from . import capi

def _closure_edge_length(r:np.ndarray, edges:np.ndarray, order=2, derivatives=0):



    edge_length2 = np.sum((r[edges[:, 0]] - r[edges[:, 1]])**2,
                             axis=1)**0.5
    num_dim = r.shape[1]
    loss = (edge_length2**order).sum()

    if derivatives == 0:
        return [loss]

    Ldl = order * edge_length2**(order - 1)

    ldre = (r[:, edges[:, 0]] - r[:, edges[:, 1]]) / edge_length2
    values_1 = Ldl * ldre

    Ldr = np.zeros([num_dim, r.shape[1]])
    Ldr.scatter_add_(1, edges[:, 0].unsqueeze(0).repeat(num_dim, 1), values_1)
    Ldr.scatter_add_(1, edges[:, 1].unsqueeze(0).repeat(num_dim, 1), -values_1)

    if derivatives == 1:
        return loss, Ldr

    Ldl_2 = order * (order - 1) * edge_length2**(order - 2)
    ldre_2 = np.eye(num_dim).reshape(num_dim, num_dim, 1) / edge_length2 + \
            -np.einsum('ip, jp->ijp', ldre, ldre) / edge_length2

    values_2 = Ldl_2 * np.einsum('ip,jp->ijp', ldre, ldre) + Ldl * ldre_2
    # diag
    Ldr2_values = [values_2.flatten(), values_2.flatten()]

    # cross item
    Ldr2_values += [-values_2.flatten(), -values_2.flatten()]

    Ldr2_values = np.cat(Ldr2_values, dim=-1)

    # region : define the indices
    # diag
    Ldr2_indices0 = [
        np.stack([
            np.arange(num_dim).reshape([num_dim, 1, 1]).repeat([
                1, num_dim, edges.shape[0]
            ]), edges[:, 0].reshape([1, 1, -1]).repeat([num_dim, num_dim, 1]),
            np.arange(num_dim).reshape([1, num_dim, 1]).repeat([
                num_dim, 1, edges.shape[0]
            ]), edges[:, 0].reshape([1, 1, -1]).repeat([num_dim, num_dim, 1])
        ],
                    dim=0).reshape([4, -1]),
        np.stack([
            np.arange(num_dim).reshape([num_dim, 1, 1]).repeat([
                1, num_dim, edges.shape[0]
            ]), edges[:, 1].reshape([1, 1, -1]).repeat([num_dim, num_dim, 1]),
            np.arange(num_dim).reshape([1, num_dim, 1]).repeat([
                num_dim, 1, edges.shape[0]
            ]), edges[:, 1].reshape([1, 1, -1]).repeat([num_dim, num_dim, 1])
        ],
                    dim=0).reshape([4, -1]),
    ]

    # cross item
    Ldr2_indices0 += [
        np.stack([
            np.arange(num_dim).reshape([num_dim, 1, 1]).repeat(
                [1, num_dim, edges.shape[0]]), edges[:, 0].reshape(
                    [1, 1, edges.shape[0]]).repeat([num_dim, num_dim, 1]),
            np.arange(num_dim).reshape([1, num_dim, 1]).repeat(
                [num_dim, 1, edges.shape[0]]), edges[:, 1].reshape(
                    [1, 1, edges.shape[0]]).repeat([num_dim, num_dim, 1])
        ],
                    dim=0).reshape([4, -1]),
        np.stack([
            np.arange(num_dim).reshape([num_dim, 1, 1]).repeat(
                [1, num_dim, edges.shape[0]]), edges[:, 1].reshape(
                    [1, 1, edges.shape[0]]).repeat([num_dim, num_dim, 1]),
            np.arange(num_dim).reshape([1, num_dim, 1]).repeat(
                [num_dim, 1, edges.shape[0]]), edges[:, 0].reshape(
                    [1, 1, edges.shape[0]]).repeat([num_dim, num_dim, 1])
        ],
                    dim=0).reshape([4, -1]),
    ]
    Ldr2_indices = np.cat(Ldr2_indices0, dim=-1)

    # endregion

    Ldr2 = np.sparse_coo_tensor(Ldr2_indices, Ldr2_values,
                                   [num_dim, r.shape[1], num_dim, r.shape[1]])

    return loss, Ldr, Ldr2

def mesh_regulation_2D(nodes: np.ndarray,
                                      elements: np.ndarray,
                                      index_nodes: np.ndarray,
                                      order=2):
    """
    Torch implementation of edge length regularization

    """
    edges = get_edges(elements)

    while True:
        loss, Ldp, Ldp2 = _closure_edge_length(nodes,
                                               edges,
                                               derivatives=2,
                                               order=order)

        Ldp = Ldp[:, index_nodes]
        Ldp2 = Ldp2.index_select(1, index_nodes).index_select(3, index_nodes)
        Ldp2 = __sparse_methods._sparse_reshape(
            Ldp2, 2 * [2 * Ldp2.shape[1]]).coalesce()

        dP = __sparse_methods._conjugate_gradient(Ldp2.indices(),
                                                  Ldp2.values(),
                                                  -Ldp.flatten(),
                                                  tol=1e-10)
        dP = dP.reshape([2, -1])
        dP.view(-1)[dP.view(-1).isnan()] = 0

        if dP.norm() < 1e-6:
            break

        nodes[:, index_nodes] += dP
    
    r1 = nodes[:, elements[:, 0]]
    r2 = nodes[:, elements[:, 1]]
    r3 = nodes[:, elements[:, 2]]
    
    e1 = r2 - r1
    e2 = r3 - r1
    normal = e1[1] * e2[0] - e1[0] * e2[1]
    
    if normal[0] < 0:
        nodes[0] *= -1

    return nodes
