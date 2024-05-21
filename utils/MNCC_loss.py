from __future__ import annotations
import torch
from einops import rearrange
# from diffdrr.visualization import plot_drr
import matplotlib.pyplot as plt
import torch.nn as nn
from torchmetrics import Metric


class MaskedNormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images."""

    def __init__(self, patch_size=None, eps=1e-5):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

    def forward(self, x1, x2, mask2):
        if self.patch_size is not None:
            x1 = to_patches(x1, self.patch_size)
            x2 = to_patches(x2, self.patch_size)
        assert x1.shape == x2.shape, "Input images must be the same size"
        _, c, h, w = x1.shape
        x2 = x2 * mask2
        x1, x2 = self.norm(x1), self.norm(x2)
        # plot_drr(x1)
        # plot_drr(x2)
        # plt.show()
        score = torch.einsum("b...,b...->b", x1, x2)
        score /= c * h * w
        return score

    def norm(self, x):
        mu = x.mean(dim=[-1, -2], keepdim=True)
        var = x.var(dim=[-1, -2], keepdim=True, correction=0) + self.eps
        std = var.sqrt()
        return (x - mu) / std


def to_patches(x, patch_size):
    x = x.unfold(2, patch_size, step=1).unfold(3, patch_size, step=1).contiguous()
    return rearrange(x, "b c p1 p2 h w -> b (c p1 p2) h w")


def cal_ncc(I, J, eps):
    # compute local sums via convolution
    cross = (I - torch.mean(I)) * (J - torch.mean(J))
    I_var = (I - torch.mean(I)) * (I - torch.mean(I))
    J_var = (J - torch.mean(J)) * (J - torch.mean(J))

    cc = torch.sum(cross) / torch.sum(torch.sqrt(I_var * J_var + eps))

    test = torch.mean(cc)
    return test


class CustomMetric(Metric):
    is_differentiable: True

    def __init__(self, LossClass, **kwargs):
        super().__init__()
        self.lossfn = LossClass(**kwargs)
        self.add_state("loss", default=torch.tensor(0.0).cuda(), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0).cuda(), dist_reduce_fx="sum")

    def update(self, preds, target, mask):
        self.loss += self.lossfn(preds, target, mask).sum()
        self.count += len(preds)

    def compute(self):
        return self.loss.float() / self.count


class MaskedNCC(CustomMetric):
    """`torchmetric` wrapper for NCC."""

    higher_is_better: True

    def __init__(self, patch_size=None):
        super().__init__(MaskedNormalizedCrossCorrelation2d, patch_size=patch_size)


# Gradient-NCC Loss
def GradNCC(I, J,  target_mask, device='cuda', win=None, eps=1e-10):
    # compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y
        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)
    Jx = Jx * target_mask
    Jy = Jy * target_mask
    # Ix = torch.permute(Ix, (0, 1, 3, 2))
    # Jx = torch.permute(Jx, (0, 1, 3, 2))
    # Iy = torch.permute(Iy, (0, 1, 3, 2))
    # Jy = torch.permute(Jy, (0, 1, 3, 2))
    plt.subplot(1, 4, 1)
    plt.imshow(Ix[0, :].squeeze().detach().cpu().numpy())
    plt.subplot(1, 4, 2)
    plt.imshow(Jx[0, :].squeeze().detach().cpu().numpy())
    plt.subplot(1, 4, 3)
    plt.imshow(Iy[0, :].squeeze().detach().cpu().numpy())
    plt.subplot(1, 4, 4)
    plt.imshow(Jy[0, :].squeeze().detach().cpu().numpy())
    plt.show()

    return 0.5 * cal_ncc(Ix, Jx, eps) + 0.5 * cal_ncc(Iy, Jy, eps)


