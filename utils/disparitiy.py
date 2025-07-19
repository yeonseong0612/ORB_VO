import torch

def compute_disparities(kpts_left: torch.Tensor, kpts_right: torch.Tensor):
    x_left = kpts_left[:, 0]
    x_right = kpts_right[:, 0]
    disparity = x_left - x_right
    return disparity