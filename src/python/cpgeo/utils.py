import numpy as np
import scipy
from . import capi
import scipy.sparse as sp
import pypardiso
def mesh_regulation_2D(nodes: np.ndarray,
                                      elements: np.ndarray,
                                      index_nodes: np.ndarray,
                                      order=2):
    """
    Torch implementation of edge length regularization

    """
    nodes = nodes.copy()

    edges = capi.get_mesh_edges(elements)[:, :2]
    num_nodes = nodes.shape[0]
    dim_nodes = nodes.shape[1]

    loss, Ldp, Ldp2_indices, Ldp2_values = capi.get_mesh_closure_edge_length_derivative2(nodes,
                                            edges,
                                            order=order)
    
    Ldp2_indices_flatten = np.stack([Ldp2_indices[:, 0] * dim_nodes + Ldp2_indices[:, 1],
                                    Ldp2_indices[:, 2] * dim_nodes + Ldp2_indices[:, 3]], axis=1)
                                    

    Ldp2 = sp.coo_matrix((Ldp2_values, (Ldp2_indices_flatten[:, 0], Ldp2_indices_flatten[:, 1])),
                         shape=(num_nodes * dim_nodes, num_nodes * dim_nodes)).tocsr()

    Ldp = Ldp[index_nodes].flatten()

    
    
    index_remain = np.stack([index_nodes * dim_nodes + d for d in range(dim_nodes)], axis=1).flatten()
    Ldp2 = Ldp2[index_remain, :][:, index_remain]

    dP = pypardiso.spsolve(Ldp2, -Ldp)

    nodes[index_nodes] += dP.reshape((len(index_nodes), dim_nodes))
    
    r1 = nodes[elements[:, 0]]
    r2 = nodes[elements[:, 1]]
    r3 = nodes[elements[:, 2]]
    
    e1 = r2 - r1
    e2 = r3 - r1
    normal = e1[:, 1] * e2[:, 0] - e1[:, 0] * e2[:, 1]
    
    if normal[0] < 0:
        nodes[:, 0] *= -1

    return nodes
