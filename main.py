import numpy as np
from PIL import Image
import cv2
import os
import torch

from TVL1OF import *

ratio = 0.5

path = "eval-data/Mequon/"
list_files = os.listdir(path)
list_files = [os.path.join(path, file) for file in list_files if file.endswith(".png")]
list_files.sort()
list_files = list_files[0:5]
images = [Image.open(f).convert('L') for f in list_files]
images = [np.array(im.resize([int(im.size[0] * ratio), int(im.size[1] * ratio)])) for im in images]
x = torch.stack([torch.tensor(im) for im in images]).float()  # / 255.0
x1 = torch.zeros([2, int((x.shape[0] - 1) / 2), x.shape[1], x.shape[2]])
x2 = torch.zeros([2, int((x.shape[0] - 1) / 2), x.shape[1], x.shape[2]])
x1[0, ...] = x[0:2, :, :]
x1[1, ...] = x[2:4, :, :]
x2[0, ...] = x[1:3, :, :]
x2[1, ...] = x[3:, :, :]
t = TVL1OF(num_iter=100)
a = t(x1, x2)
tv = a.data.cpu().numpy()
tv = tv.reshape(tv.shape[0] * tv.shape[1], tv.shape[2], tv.shape[3], tv.shape[4])
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
