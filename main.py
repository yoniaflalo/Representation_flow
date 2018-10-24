import numpy as np
import torch.nn as nn
import torch


class TVL1OF(nn.Module):
    def __init__(self, size_in, num_iter, lambda_, tau, theta, is_w_trainable=True):
        self.ch_in = 1
        super(TVL1OF, self).__init__()
        self.size_y = size_in[0]
        self.size_x = size_in[1]
        self.num_iter = num_iter
        self.grad_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], requires_grad=False)
        self.grad_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], requires_grad=False)
        self.grad_ = torch.stack([self.grad_x, self.grad_y])
        if is_w_trainable:
            self.wx = nn.Parameter(torch.tensor([[1.0, 1.0]]))
            self.wy = nn.Parameter((torch.tensor([[1.0], [1.0]])))
        else:
            self.wx = torch.tensor([[1.0, 1.0]], requires_grad=False)
            self.wy = torch.tensor([[1.0], [1.0]], requires_grad=False)
        self.lambda_ = nn.Parameter(torch.tensor([lambda_]))
        self.tau = nn.Parameter(torch.tensor([tau]))
        self.theta = nn.Parameter(torch.tensor([theta]))
        self.grad = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=[3, 3], padding=1, bias=False)
        self.grad.weight.data = self.grad_.unsqueeze(1)
        self.grad.weight.requires_grad = False
        self.div_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 2], padding=[0, 1], bias=False)
        self.div_x.weight.data = self.wx.unsqueeze(0).unsqueeze(0)
        self.div_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2, 1], padding=[1, 0], bias=False)
        self.div_y.weight.data = self.wy.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        num_batch = x.shape[0]
        p1 = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        p2 = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        u = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        v = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        rho_c = x[:, 1:, :, :] - x[:, 0:-1, :, :]
        grad_im = self.grad(x[:, 1:, :, :])
        norm_grad = torch.sum(grad_im * grad_im, dim=1).unsqueeze(1)
        for i in range(self.num_iter):
            rho = rho_c + torch.sum(grad_im * u, dim=1).unsqueeze(1)
            th = self.theta * self.lambda_ * norm_grad
            v = u - (torch.abs(rho) < th).float() * rho * grad_im / norm_grad - (
                    torch.abs(rho) > th).float() * self.theta * self.lambda_ * grad_im * torch.sign(rho)
            u[:, 0, :, :] = v[:, 0, :, :] + self.theta * (
                    self.div_x(p1[:, 0, :, :].unsqueeze(1))[:, :, :, 0:-1] + self.div_y(
                p2[:, 0, :, :].unsqueeze(1))[:, :, 0:-1, :]).squeeze()
            u[:, 1, :, :] = v[:, 1, :, :] + self.theta * (
                    self.div_x(p1[:, 1, :, :].unsqueeze(1))[:, :, :, 0:-1] + self.div_y(
                p2[:, 1, :, :].unsqueeze(1))[:, :, 0:-1, :]).squeeze()
            gradu1 = self.grad(u[:, 0, :, :].unsqueeze(1))
            gradu2 = self.grad(u[:, 1, :, :].unsqueeze(1))
            p1 = (p1 + self.tau / self.theta * gradu1) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu1)))
            p2 = (p2 + self.tau / self.theta * gradu2) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu2)))
        return u


t = TVL1OF([256, 256], 20, 10.0, 0.1, 1.0)
x = torch.randn([3, 2, 256, 256])
a = t(x)
