import numpy as np
import torch.nn as nn
import torch


class TVL1OF(nn.Module):
    def __init__(self, size_in, ch_in, num_iter, lambda_, tau, theta, is_w_trainable=True):
        super(TVL1OF, self).__init__()
        self.size_y = size_in[0]
        self.size_x = size_in[1]
        self.num_iter = num_iter
        self.p = torch.zeros([2, 2, self.size_y, self.size_x])
        self.u = torch.zeros([1, 2*ch_in, self.size_y, self.size_x])
        self.v = torch.zeros([1, 2*ch_in, self.size_y, self.size_x])
        self.grad_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], requires_grad=False)
        self.grad_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], requires_grad=False)
        self.grad_ = torch.stack([self.grad_x, self.grad_y])
        if is_w_trainable:
            self.wx = nn.Parameter(torch.tensor([[1.0, 1.0]]))
            self.wy = nn.Parameter((torch.tensor([[1.0], [1.0]])))
        else:
            self.wx = torch.tensor([[1.0, 1.0]])
            self.wy = torch.tensor([[1.0], [1.0]])
        self.lambda_ = nn.Parameter(torch.tensor([lambda_]))
        self.tau = nn.Parameter(torch.tensor([tau]))
        self.theta = nn.Parameter(torch.tensor([theta]))
        self.grad = nn.Conv2d(in_channels=ch_in, out_channels=2*ch_in, kernel_size=[3, 3], padding=1, groups=ch_in, bias=False)
        self.grad.weight.data = self.grad_.unsqueeze(1).repeat([ch_in,1,1,1])
        self.grad.weight.requires_grad = False
        self.div_x = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=[1, 2], padding=1, groups=ch_in,  bias=False)
        self.div_x.weight.data = self.wx.unsqueeze(1).repeat([ch_in,1,1,1])
        self.div_y = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=[2, 1], padding=1, groups=ch_in,  bias=False)
        self.div_y.weight.data = self.wy.unsqueeze(1).repeat([ch_in,1,1,1])

    def forward(self, x):
        rho_c = x[:, 1:, :, :] - x[:, 0:-1, :, :]
        grad_im = self.grad(x[:, 1:, :, :])
        norm_grad = torch.sum(grad_im * grad_im, dim=-3).unsqueeze(1)
        for i in range(self.num_iter):
            rho = rho_c + torch.sum(grad_im * self.u, dim=-3).unsqueeze(1)
            th = self.theta * self.lambda_ * norm_grad
            self.v = self.u - (torch.abs(rho) < th).float() * rho * grad_im / norm_grad - (
                        torch.abs(rho) > th).float() * self.theta * self.lambda_ * grad_im * torch.sign(rho)
            px = self.p[0, :, :, :]
            py = self.p[1, :, :, :]
            self.u = self.v + self.theta * (self.div_x(px) + self.div_y(py))
            gradu = self.grad(self.u)
            self.p = (self.p + self.tau / self.theta * gradu) / (
                        1 + self.tau / self.theta * torch.sum(torch.abs(gradu)))
        return self.u

t=TVL1OF([100,30],5,20,10.0,0.1,1.0)
x=torch.randn([3,6,100,30])
t(x)