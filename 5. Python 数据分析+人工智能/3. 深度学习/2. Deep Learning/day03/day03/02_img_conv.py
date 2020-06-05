# img_conv.py
# 图像卷积示例
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sn

im = misc.imread("../data/zebra.png", flatten=True)
# im = sn.imread("../data/zebra.png", flatten=True)
flt = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
flt2 = np.array([[1, 2, 1],
                 [0, 0, 0],
                 [-1, -2, -1]])
conv_img1 = signal.convolve2d(im, flt,
                      boundary="symm", #边界处理方式
                      mode="same").astype("int32")#同维卷积
conv_img2 = signal.convolve2d(im, flt2,
                      boundary="symm", #边界处理方式
                      mode="same").astype("int32")#同维卷积
plt.figure("Conv2D")
plt.subplot(131)
plt.imshow(im, cmap="gray") #原始图像
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.imshow(conv_img1, cmap="gray") #卷积核1卷积后的图像
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.imshow(conv_img2, cmap="gray") #卷积核2卷积后的图像
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()