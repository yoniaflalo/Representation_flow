import torch.nn as nn
import torch


class TVL1OF(nn.Module):
    def __init__(self, num_iter=20, lambda_=0.02, tau=0.25, theta=0.1, is_w_trainable=True, verbose = False):
        self.ch_in = 1
        super(TVL1OF, self).__init__()
        self.num_iter = num_iter
        self.verbose =verbose
        self.lambda_ = nn.Parameter(torch.tensor([lambda_]))
        self.tau = nn.Parameter(torch.tensor([tau]))
        self.theta = nn.Parameter(torch.tensor([theta]))
        self.grad_x = 1 / 6.0 * torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                                             requires_grad=False)
        self.grad_y = 1 / 6.0 * torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                                             requires_grad=False)
        if is_w_trainable:
            self.wx = nn.Parameter(torch.tensor([[-1.0, 1.0, 0.0]]))
            self.wy = nn.Parameter((torch.tensor([[-1.0], [1.0], [0.0]])))
        else:
            self.wx = 0.5 * torch.tensor([[-1.0, 1.0, 0.0]], requires_grad=False)
            self.wy = 0.5 * torch.tensor([[-1.0], [1.0], [0.0]], requires_grad=False)
        self.grad_x_u = torch.tensor([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
                                     requires_grad=False)
        self.grad_y_u = torch.tensor([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                                     requires_grad=False)
        self.grad_ = torch.stack([self.grad_x, self.grad_y])
        self.grad_u_ = torch.stack([self.grad_x_u, self.grad_y_u])
        self.grad = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=[3, 3], padding=1, bias=False)
        self.grad_u = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=[3, 3], padding=1, bias=False)
        self.div_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 3], padding=[0, 1], bias=False)
        self.div_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, 1], padding=[1, 0], bias=False)
        self.div_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 3], padding=[0, 1], bias=False)
        self.div_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, 1], padding=[1, 0], bias=False)
        self.div_x.weight.data = self.wx.unsqueeze(0).unsqueeze(0)
        self.div_y.weight.data = self.wy.unsqueeze(0).unsqueeze(0)
        self.grad.weight.data = self.grad_.unsqueeze(1)
        self.grad.weight.requires_grad = False
        self.grad_u.weight.data = self.grad_u_.unsqueeze(1)
        self.grad_u.weight.requires_grad = False

    def forward(self, x1, x2):
        if not x1.shape == x2.shape:
            raise NotImplementedError
        shape = x1.shape
        batch_size = shape[0]
        num_channels = shape[1]
        size_y = shape[2]
        size_x = shape[3]
        xx1 = x1
        xx2 = x2
        xx1 = xx1.reshape([batch_size * num_channels, size_y, size_x])
        xx2 = xx2.reshape([batch_size * num_channels, size_y, size_x])
        xx = torch.stack([xx1, xx2], dim=1)
        epsilon = 1e-8
        num_batch = xx.shape[0]
        p1 = torch.zeros([num_batch, 2, size_y, size_x])
        p2 = torch.zeros([num_batch, 2, size_y, size_x])
        u = torch.zeros([num_batch, 2, size_y, size_x])
        v = torch.zeros([num_batch, 2, size_y, size_x])
        rho_c = (xx[:, 1, :, :] - xx[:, 0, :, :]).unsqueeze(1)
        grad_im = self.grad(xx[:, 1:, :, :])
        norm_grad = torch.sum(grad_im * grad_im, dim=1).unsqueeze(1) + epsilon
        for i in range(self.num_iter):
            rho = rho_c + torch.sum(grad_im * u, dim=1).unsqueeze(1)
            th = self.theta * self.lambda_ * norm_grad
            v = u - (torch.abs(rho) < th).float() * rho * grad_im / norm_grad - (
                    torch.abs(rho) >= th).float() * self.theta * self.lambda_ * grad_im * torch.sign(rho)
            u[:, 0, :, :] = v[:, 0, :, :] + self.theta * (self.div_x(p1[:, 0, :, :].unsqueeze(1)) + self.div_y(
                p1[:, 1, :, :].unsqueeze(1))).squeeze()
            u[:, 1, :, :] = v[:, 1, :, :] + self.theta * (self.div_x(p2[:, 0, :, :].unsqueeze(1)) + self.div_y(
                p2[:, 1, :, :].unsqueeze(1))).squeeze()
            gradu1 = self.grad_u(u[:, 0, :, :].unsqueeze(1))
            gradu2 = self.grad_u(u[:, 1, :, :].unsqueeze(1))
            p1 = (p1 + self.tau / self.theta * gradu1) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu1), dim=1).unsqueeze(1))
            p2 = (p2 + self.tau / self.theta * gradu2) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu2), dim=1).unsqueeze(1))
            rho = rho_c + torch.sum(grad_im * u, dim=1).unsqueeze(1)
            err = torch.sum(torch.abs(self.lambda_ * rho) + torch.abs(gradu1) + torch.abs(gradu2))
            if self.verbose:
                print(f"Error : {err.data.cpu().numpy()}")
        u = u.reshape(batch_size, num_channels, 2, size_y, size_x)
        # u = torch.nn.AvgPool2d(3, stride=1, padding=1)(u)

        return u
