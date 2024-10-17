import numpy as np
import torch


def calc_neighbor(label_1, label_2):
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(np.float32)
    return Sim


def calc_neighbor_unknown(label_1, label_2):
    Sim = np.dot(label_1, label_2.transpose()).astype(np.float32)
    Sim[(Sim > 0) & (Sim < 0.01)] = -1  # unknown removal
    Sim[Sim >= 1] = 1
    return Sim


def calc_neighbor_an(label_1, label_2):
    label_1[(label_1 <= 0.01)] = 0
    label_2[(label_2 <= 0.01)] = 0
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(np.float32)
    return Sim


def calc_neighbor_unknown_pytorch(label_1, label_2):
    Sim = torch.mm(label_1, label_2.t()).float()
    Sim[(Sim > 0) & (Sim < 0.01)] = -1  # unknown removal
    Sim[Sim >= 1] = 1
    #Sim = torch.tanh(Sim)
    return Sim


def calc_neighbor_an_pytorch(label_1, label_2):
    label_1[(label_1 <= 0.01)] = 0
    label_2[(label_2 <= 0.01)] = 0
    Sim = (torch.mm(label_1, label_2.t()) > 0).float()
    return Sim


def calc_neighbor_adaptive_pytorch(label_1, label_2, thres=0.01):
    Sim = calc_neighbor_unknown_pytorch(label_1, label_2)
    r = (Sim == 0.0).sum() / (Sim > 0.01).sum()
    # print(r.data)
    if r < thres:
        Sim[Sim == -1] = 0
    return Sim, r.data


def calc_neighbor_adaptive_negative_fillup_pytorch(label_1, label_2, thres=0.01):
    Sim = calc_neighbor_unknown_pytorch(label_1, label_2)
    n_pos = (Sim > 0.01).sum()
    n_neg = (Sim == 0.0).sum()
    r = n_neg / n_pos
    # print(r.data)
    if r < thres:
        n_fill = (n_pos * thres).int() - n_neg
        idx = torch.where(Sim == -1)
        i_sel = torch.randperm(len(idx[0]))[:n_fill]
        new_idx = (idx[0][i_sel], idx[1][i_sel])
        Sim[new_idx] = 0
    return Sim, r.data


def calc_neighbor_augmented_pytorch(label_1, label_2):
    label_1[label_1 == 0.0001] = 0.1
    label_2[label_2 == 0.0001] = 0.1
    Sim = torch.mm(label_1, label_2.t()).float()
    Sim = torch.tanh(Sim)
    return Sim


def calc_neighbor_continuous_label_pytorch(label_1, label_2):
    Sim = 1 - (1 - label_1.unsqueeze(1) * label_2.unsqueeze(0)).prod(dim=-1)
    return Sim


def calc_neighbor_adaptive_negative_fillup_isolated_pytorch(label_1, label_2, fill_thres=0.001, fill_ratio=1.0):
    Sim = calc_neighbor_unknown_pytorch(label_1, label_2)
    n_pos = (Sim == 1.0).sum()
    n_neg = (Sim == 0.0).sum()
    r = n_neg / n_pos
    # print(r.data)
    if r < fill_thres:
        n_fill = (n_pos * fill_ratio).int() - n_neg
        idx = torch.where(Sim == -1)
        i_sel = torch.randperm(len(idx[0]))[:n_fill]
        new_idx = (idx[0][i_sel], idx[1][i_sel])
        Sim[new_idx] = 0
    return Sim, r.data


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
