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
        self.grad_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], requires_grad=False)
        self.grad_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], requires_grad=False)
        self.grad_ = torch.stack([self.grad_x, self.grad_y])
        if is_w_trainable:
            self.wx = nn.Parameter(torch.tensor([[-1.0, 0.0, 1.0]]))
            self.wy = nn.Parameter((torch.tensor([[-1.0], [0.0], [1.0]])))
        else:
            self.wx = torch.tensor([[-1.0, 0.0, 1.0]], requires_grad=False)
            self.wy = torch.tensor([[-1.0], [0.0], [1.0]], requires_grad=False)
        self.lambda_ = nn.Parameter(torch.tensor([lambda_]))
        self.tau = nn.Parameter(torch.tensor([tau]))
        self.theta = nn.Parameter(torch.tensor([theta]))
        self.grad = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=[3, 3], padding=1, bias=False)
        self.grad.weight.data = self.grad_.unsqueeze(1)
        self.grad.weight.requires_grad = False

        self.gradx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, 3], padding=1, bias=False)
        self.gradx.weight.data = self.grad_x.unsqueeze(0).unsqueeze(0)
        self.gradx.weight.requires_grad = False

        self.grady = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, 3], padding=1, bias=False)
        self.grady.weight.data = self.grad_y.unsqueeze(0).unsqueeze(0)
        self.grady.weight.requires_grad = False

        self.div_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 3], padding=[0, 1], bias=False)
        self.div_x.weight.data = self.wx.unsqueeze(0).unsqueeze(0)
        self.div_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, 1], padding=[1, 0], bias=False)
        self.div_y.weight.data = self.wy.unsqueeze(0).unsqueeze(0)

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
            rho = rho_c + torch.sum(grad_im * u, dim=1).unsqueeze(1)
            th = self.theta * self.lambda_ * norm_grad
            v = u - (torch.abs(rho) < th).float() * rho * grad_im / norm_grad - (
                    torch.abs(rho) >= th).float() * self.theta * self.lambda_ * grad_im * torch.sign(rho)

            if i == self.num_iter - 2:
                tmp = 1

            u[:, 0, :, :] = v[:, 0, :, :] + self.theta * (self.gradx(p1[:, 0, :, :].unsqueeze(1)) + self.grady(
                p2[:, 0, :, :].unsqueeze(1))).squeeze()
            u[:, 1, :, :] = v[:, 1, :, :] + self.theta * (self.gradx(p1[:, 1, :, :].unsqueeze(1)) + self.grady(
                p2[:, 1, :, :].unsqueeze(1))).squeeze()
            gradu1 = self.grad(u[:, 0, :, :].unsqueeze(1))
            gradu2 = self.grad(u[:, 1, :, :].unsqueeze(1))
            p1 = (p1 + self.tau / self.theta * gradu1) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu1), dim=1).unsqueeze(1))
            p2 = (p2 + self.tau / self.theta * gradu2) / (
                    1 + self.tau / self.theta * torch.sum(torch.abs(gradu2), dim=1).unsqueeze(1))
            rho = rho_c + torch.sum(grad_im * u, dim=1).unsqueeze(1)
            err = torch.sum(torch.abs(self.lambda_ * rho) + torch.abs(gradu1) + torch.abs(gradu2))
            print(err.data.cpu().numpy())
        return u


ratio = 1

path = "eval-data/Dumptruck/"
list_files = os.listdir(path)
list_files = [os.path.join(path, file) for file in list_files if file.endswith(".png")]
list_files.sort()
# list_files = list_files[0:2]
images = [Image.open(f).convert('L') for f in list_files]
images = [np.array(im.resize([int(im.size[0] * ratio), int(im.size[1] * ratio)])) for im in images]
x = torch.stack([torch.tensor(im) for im in images]).float() / 255.0

x = torch.stack([torch.stack([x[i, :, :], x[i + 1, :, :]]) for i in range(x.shape[0] - 1)])
t = TVL1OF(size_in=images[0].shape, num_iter=50, lambda_=50.0, tau=0.25, theta=0.0001)
a = t(x)
tv = a.data.cpu().numpy()
for i in range(tv.shape[0]):
    tv1 = tv[i, 0, :, :].squeeze()
    tv2 = tv[i, 1, :, :].squeeze()
    hsv = np.zeros([tv1.shape[0], tv1.shape[1], 3])
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(tv1, tv2)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    Image.fromarray(bgr).show()

    #
    im1 = cv2.imread(list_files[i])
    im2 = cv2.imread(list_files[i + 1])

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros([im1.shape[0], im1.shape[1], 3])
    hsv[..., 1] = 255
    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    Image.fromarray(bgr).show()
