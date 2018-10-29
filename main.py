import numpy as np
import torch.nn as nn
import torch
from PIL import Image
import cv2
import os


class TVL1OF(nn.Module):
    def __init__(self, size_in, num_iter, lambda_, tau, theta, is_w_trainable=True):
        self.ch_in = 1
        super(TVL1OF, self).__init__()
        self.size_y = size_in[0]
        self.size_x = size_in[1]
        self.num_iter = num_iter
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

    def forward(self, x):
        epsilon = 1e-8
        num_batch = x.shape[0]
        p1 = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        p2 = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        u = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        v = torch.zeros([num_batch, 2, self.size_y, self.size_x])
        rho_c = (x[:, 1, :, :] - x[:, 0, :, :]).unsqueeze(1)
        grad_im = self.grad(x[:, 1:, :, :])
        norm_grad = torch.sum(grad_im * grad_im, dim=1).unsqueeze(1) + epsilon
        for i in range(self.num_iter):
            # u = torch.nn.AvgPool2d(3, stride=1, padding=1)(u)
            rho = rho_c + torch.sum(grad_im * u, dim=1).unsqueeze(1)
            th = self.theta * self.lambda_ * norm_grad
            v = u - (torch.abs(rho) < th).float() * rho * grad_im / norm_grad - (
                    torch.abs(rho) >= th).float() * self.theta * self.lambda_ * grad_im * torch.sign(rho)
            u[:, 0, :, :] = v[:, 0, :, :] + self.theta * (self.div_x(p1[:, 0, :, :].unsqueeze(1)) + self.div_y(
                p1[:, 1, :, :].unsqueeze(1))).squeeze()
            u[:, 1, :, :] = v[:, 1, :, :] + self.theta * (self.div_x(p2[:, 0, :, :].unsqueeze(1)) + self.div_y(
                p2[:, 1, :, :].unsqueeze(1))).squeeze()
            gradu2 = self.grad_u(u[:, 1, :, :].unsqueeze(1))
            gradu1 = self.grad_u(u[:, 0, :, :].unsqueeze(1))
            p1 = (p1 + self.tau / self.theta * gradu1) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu1), dim=1).unsqueeze(1))
            p2 = (p2 + self.tau / self.theta * gradu2) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu2), dim=1).unsqueeze(1))
            rho = rho_c + torch.sum(grad_im * u, dim=1).unsqueeze(1)
            err = torch.sum(torch.abs(self.lambda_ * rho) + torch.abs(gradu1) + torch.abs(gradu2))
            print(err.data.cpu().numpy())
        return torch.nn.AvgPool2d(3, stride=1, padding=1)(u)


ratio = 0.5

path = "eval-data/Mequon/"
list_files = os.listdir(path)
list_files = [os.path.join(path, file) for file in list_files if file.endswith(".png")]
list_files.sort()
list_files = list_files[0:2]
images = [Image.open(f).convert('L') for f in list_files]
images = [np.array(im.resize([int(im.size[0] * ratio), int(im.size[1] * ratio)])) for im in images]
x = torch.stack([torch.tensor(im) for im in images]).float()  # / 255.0

x = torch.stack([torch.stack([x[i, :, :], x[i + 1, :, :]]) for i in range(x.shape[0] - 1)])
x=torch.nn.AvgPool2d(3, stride=1, padding=1)(x)
t = TVL1OF(size_in=images[0].shape, num_iter=100, lambda_=0.03, tau=0.25, theta=0.3)
a = t(x)
tv = a.data.cpu().numpy()
for i in range(tv.shape[0]):
    tv1 = tv[i, 0, :, :].squeeze()
    tv2 = tv[i, 1, :, :].squeeze()
    hsv = np.zeros([tv1.shape[0] - 2, tv1.shape[1] - 2, 3])
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(tv1, tv2)
    hsv[..., 0] = ang[1:-1, 1:-1, ...] * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag[1:-1, 1:-1, ...], None, 0, 255, cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    Image.fromarray(bgr).show()

    im1 = cv2.imread(list_files[i])
    im2 = cv2.imread(list_files[i + 1])

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros([im1.shape[0], im1.shape[1], 3])
    hsv[..., 1] = 255
    # flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dtvl1 = cv2.createOptFlow_DualTVL1()
    flow = dtvl1.calc(im1, im2, None)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    Image.fromarray(bgr).show()
