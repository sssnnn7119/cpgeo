import torch
import numpy as np
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from sparse import _sparse_reshape, _sparse_permute, _sparse_unsqueeze_repeat

def _from_AdI_to_Adru(AdI: torch.Tensor = None,
                        AdII: torch.Tensor = None,
                        Idru: torch.Tensor = None,
                        IIdru: torch.Tensor = None,
                        IIdru2: torch.Tensor = None):
    if AdI is None:
        AdI = torch.zeros([1, 1, 1])
    if AdII is None:
        AdII = torch.zeros([1, 1, 1])
    if Idru is None:
        Idru = torch.zeros([1, 1, 1, 1, 1])
    if IIdru is None:
        IIdru = torch.zeros([1, 1, 1, 1, 1])

    Adru = torch.einsum('mnp, mnirp->irp', AdI, Idru) + \
                torch.einsum('mnp, mnirp->irp', AdII, IIdru)
    Adru2 = torch.einsum('mnp, mnirsp->irsp', AdII, IIdru2)
    return Adru, Adru2

def _from_AdI_to_Adru_2(AdI: torch.Tensor = None,
                        AdII: torch.Tensor = None,
                        AdI_2: torch.Tensor = None,
                        AdIdII: torch.Tensor = None,
                        AdII_2: torch.Tensor = None,
                        Idru: torch.Tensor = None,
                        IIdru: torch.Tensor = None,
                        IIdru2: torch.Tensor = None,
                        Idru_2: torch.Tensor = None,
                        IIdru_2: torch.Tensor = None,
                        IIdrudru2: torch.Tensor = None):
    """
    return: Adru_2, Adrudru2, Adru2_2
    """

    if AdI is None:
        AdI = torch.zeros([1, 1, 1])
    if AdII is None:
        AdII = torch.zeros([1, 1, 1])
    if AdI_2 is None:
        AdI_2 = torch.zeros([1, 1, 1, 1, 1])
    if AdIdII is None:
        AdIdII = torch.zeros([1, 1, 1, 1, 1])
    if AdII_2 is None:
        AdII_2 = torch.zeros([1, 1, 1, 1, 1])

    if Idru is None:
        Idru = torch.zeros([1, 1, 1, 1, 1])
    if IIdru is None:
        IIdru = torch.zeros([1, 1, 1, 1, 1])
    if IIdru2 is None:
        IIdru2 = torch.zeros([1, 1, 1, 1, 1, 1])
    if Idru_2 is None:
        Idru_2 = torch.zeros([1, 1, 1, 1, 1, 1, 1])
    if IIdru_2 is None:
        IIdru_2 = torch.zeros([1, 1, 1, 1, 1, 1, 1])
    if IIdrudru2 is None:
        IIdrudru2 = torch.zeros([1, 1, 1, 1, 1, 1, 1, 1])

    temp = torch.einsum('MNmnp, MNirp, mnjsp->irjsp', AdIdII, Idru, IIdru)
    Adru_2 = torch.einsum('mnp, mnirjsp->irjsp', AdI, Idru_2) + \
                torch.einsum('MNmnp, MNirp, mnjsp->irjsp', AdI_2, Idru, Idru) + \
                torch.einsum('mnp, mnirjsp->irjsp', AdII, IIdru_2) + \
                torch.einsum('MNmnp, MNirp, mnjsp->irjsp', AdII_2, IIdru, IIdru) + \
                temp + \
                temp.permute([2, 3, 0, 1, 4])

    Adrudru2 = torch.einsum('MNmnp, MNirp, mnjsqp->irjsqp', AdIdII, Idru, IIdru2) + \
                torch.einsum('MNmnp, MNirp, mnjsqp->irjsqp', AdII_2, IIdru, IIdru2) + \
                torch.einsum('MNp, MNirjsqp->irjsqp', AdII, IIdrudru2)

    Adru2_2 = torch.einsum('MNmnp, MNiosp, mnjqrp->iosjqrp', AdII_2,
                            IIdru2, IIdru2)
    return Adru_2, Adrudru2, Adru2_2

def _from_Adr_to_Adot(indices: torch.Tensor,
                        Adr: torch.Tensor = None,
                        Adru: torch.Tensor = None,
                        Adru2: torch.Tensor = None,
                        rdot: torch.Tensor = None,
                        rdudot: torch.Tensor = None,
                        rdu2dot: torch.Tensor = None,
                        numel_output: int = None):
    if Adr is None:
        Adr = torch.zeros([1, 1])
    else:
        Adr = Adr.index_select(-1, indices[1])

    if Adru is None:
        Adru = torch.zeros([1, 1, 1])
    else:
        Adru = Adru.index_select(-1, indices[1])

    if Adru2 is None:
        Adru2 = torch.zeros([1, 1, 1, 1])
    else:
        Adru2 = Adru2.index_select(-1, indices[1])

    if rdot is None:
        rdot = torch.zeros([1])

    if rdudot is None:
        rdudot = torch.zeros([1, 1])

    if rdu2dot is None:
        rdu2dot = torch.zeros([1, 1, 1])

    Adot = torch.einsum('ip, p->ip', Adr, rdot) + \
            torch.einsum('irp, rp->ip', Adru, rdudot) + \
            torch.einsum('irsp, rsp->ip', Adru2, rdu2dot)

    if numel_output is None:
        numel_output = int(indices[0].max().item() + 1)
    Adot = Adot.reshape([-1, int(Adot.shape[-1])])
    result = torch.zeros([int(Adot.shape[0]), numel_output],
                            device=Adot.device)
    for i in range(Adot.shape[0]):
        result[i].scatter_add_(0, indices[0], Adot[i])
    return result

# @torch.jit.script
def _from_Adr_to_Adot_2(indices: torch.Tensor,
                        Adr_2: torch.Tensor = torch.zeros([0]),
                        Adru_2: torch.Tensor = torch.zeros([0]),
                        Adrudru2: torch.Tensor = torch.zeros([0]),
                        Adru2_2: torch.Tensor = torch.zeros([0]),
                        rdot: torch.Tensor = torch.zeros([0]),
                        rdudot: torch.Tensor = torch.zeros([0]),
                        rdu2dot: torch.Tensor = torch.zeros([0])):

    indices_new = indices[1] * (indices[0].max() + 10) + indices[0]
    argsort = indices_new.argsort()
    # reorder the data
    indices0 = indices[:, argsort]

    num_points = int(indices[1].max().item() + 1)

    if Adr_2.numel() == 0:
        Adr_2 = torch.zeros([1, 1, num_points], device=indices.device)
    if Adru_2.numel() == 0:
        Adru_2 = torch.zeros([1, 1, 1, 1, num_points],
                                device=indices.device)
    if Adrudru2.numel() == 0:
        Adrudru2 = torch.zeros([1, 1, 1, 1, 1, num_points],
                                device=indices.device)
    if Adru2_2.numel() == 0:
        Adru2_2 = torch.zeros([1, 1, 1, 1, 1, 1, num_points],
                                device=indices.device)

    if rdot.numel() == 0:
        rdot0 = torch.zeros([int(indices.shape[1])], device=indices.device)
    else:
        rdot0 = rdot[argsort]
    if rdudot.numel() == 0:
        rdudot0 = torch.zeros([1, int(indices.shape[1])],
                                device=indices.device)
    else:
        rdudot0 = rdudot[:, argsort]
    if rdu2dot.numel() == 0:
        rdu2dot0 = torch.zeros([1, 1, int(indices.shape[1])],
                                device=indices.device)
    else:
        rdu2dot0 = rdu2dot[:, :, argsort]

    indices_num = torch.bincount(indices0[1])

    indices_index_cusum = torch.cat([
        torch.tensor([0], dtype=indices_num.dtype, device=Adru_2.device),
        indices_num.cumsum(0)
    ])

    indices_num_unique = torch.unique(indices_num)

    numel_control_points = int(indices[0].max().item() + 1)
    numel_points = int(indices[1].max().item() + 1)

    Adot_2 = torch.sparse_coo_tensor(
        indices=torch.zeros([4, 0],
                            dtype=indices0.dtype,
                            device=indices0.device),
        values=torch.zeros([0], device=Adru_2.device),
        size=[3, numel_control_points, 3, numel_control_points],
        dtype=Adru_2.dtype,
        device=Adru_2.device).coalesce()

    for i in range(len(indices_num_unique)):
        ind = indices_num_unique[i]
        index_Adru = torch.where(indices_num == ind)[0]
        index_now = (indices_index_cusum[index_Adru] + torch.arange(
            ind, dtype=indices_num.dtype,
            device=indices_num.device).unsqueeze(-1))
        index_indices = indices0[0][index_now.flatten()].reshape_as(
            index_now)

        Adr_2_now = Adr_2.index_select(-1, index_Adru)
        Adru_2_now = Adru_2.index_select(-1, index_Adru)
        Adrudru2_now = Adrudru2.index_select(-1, index_Adru)
        Adru2_2_now = Adru2_2.index_select(-1, index_Adru)

        rdot_now = rdot0[index_now.flatten()].reshape(
            [int(ind.item()), -1])
        rdudot_now = rdudot0[:, index_now.flatten()].reshape(
            [int(rdudot0.shape[0]),
                int(ind.item()), -1])
        rdu2dot_now = rdu2dot0[:, :, index_now.flatten()].reshape([
            int(rdu2dot0.shape[0]),
            int(rdu2dot0.shape[0]),
            int(ind.item()), -1
        ])

        v0 = torch.einsum('ijp, rp, sp->irjsp', Adr_2_now, rdot_now,
                            rdot_now)
        v1 = torch.einsum('imjnp, mrp, nsp->irjsp', Adru_2_now, rdudot_now,
                            rdudot_now)
        v2 = torch.einsum('imjnbp, mrp, nbsp->irjsp', Adrudru2_now,
                            rdudot_now, rdu2dot_now)
        v2 = v2 + v2.permute([2, 3, 0, 1, 4])
        v3 = torch.einsum('imajnbp, marp, nbsp->irjsp', Adru2_2_now,
                            rdu2dot_now, rdu2dot_now)

        result_values = (v0 + v1 + v2 + v3).flatten()
        result_indices1 = torch.tensor([0, 1, 2],
                                        dtype=indices0.dtype,
                                        device=indices0.device).reshape(
                                            [3, 1, 1, 1, 1]).repeat([
                                                1,
                                                int(ind), 3,
                                                int(ind),
                                                int(index_Adru.shape[0])
                                            ])
        result_indices2 = index_indices.reshape(
            [1, int(ind), 1, 1,
                int(index_Adru.shape[0])]).repeat([3, 1, 3,
                                                int(ind), 1])
        result_indices3 = torch.tensor([0, 1, 2],
                                        dtype=indices0.dtype,
                                        device=indices0.device).reshape(
                                            [1, 1, 3, 1, 1]).repeat([
                                                3,
                                                int(ind), 1,
                                                int(ind),
                                                int(index_Adru.shape[0])
                                            ])
        result_indices4 = index_indices.reshape(
            [1, 1, 1, int(ind),
                int(index_Adru.shape[0])]).repeat([3, int(ind), 3, 1, 1])

        result_indices = torch.stack([
            result_indices1, result_indices2, result_indices3,
            result_indices4
        ],
                                        dim=0).reshape([4, -1])
        index = torch.where(result_values != 0)[0]
        Adot_2 = Adot_2 + (torch.sparse_coo_tensor(
            indices=result_indices[:, index],
            values=result_values[index],
            size=[3, numel_control_points, 3, numel_control_points],
            dtype=Adru_2.dtype,
            device=Adru_2.device).coalesce())
    Adot_2 = Adot_2.coalesce()
    return Adot_2

def _from_Sdr_to_Sdot_2(indices: torch.Tensor, Sdr_2: torch.Tensor,
                        rdot: torch.Tensor):
    """
    Sdr: [3, num_points, 3, num_points]
    rdot: [num_sparse]
    """
    num_points = Sdr_2.shape[1]
    num_control_points = int(indices[0].max().item() + 1)

    rdot = torch.sparse_coo_tensor(indices=indices.flip(0),
                                    values=rdot,
                                    size=[num_points, num_control_points],
                                    dtype=rdot.dtype,
                                    device=rdot.device).coalesce()

    Sdr_2 = _sparse_reshape(Sdr_2,
                                            [3 * 3 * num_points, num_points])
    Sdot = Sdr_2 @ rdot
    Sdot = _sparse_reshape(
        Sdot, [3, num_points, 3, num_control_points])
    Sdot = _sparse_permute(Sdot, [2, 3, 0, 1])
    Sdot = _sparse_reshape(
        Sdot, [3 * num_control_points * 3, num_points])
    Sdot = Sdot @ rdot
    Sdot = _sparse_reshape(
        Sdot, [3, num_control_points, 3, num_control_points])
    return Sdot
