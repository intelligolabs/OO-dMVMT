import torch
import numpy as np

def norm_scale(x):
    return (x - torch.mean(x)) / torch.mean(x)

def get_CG(x: torch.Tensor, in_n=16, joints_n=26):
    """
    x : shape (B, in_n, joints_n, 3)
    """
    p = x
    M = []
    # upper triangle index with offset 1, which means upper triangle without diagonal
    iu = torch.triu_indices(joints_n, joints_n, 1).to(p.device)
    d_m = torch.cdist(p[:], p[:], p=2)
    d_m = torch.nan_to_num(d_m, nan=0)
    M = d_m[:, iu[0], iu[1]]
    M = norm_scale(M)  # normalize
    M = torch.nan_to_num(M, nan=0)
    return M
