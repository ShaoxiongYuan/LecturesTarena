# 04_equalize_hist.py
# 直方图均衡化示例
# 直方图: 图像中每个像素灰度值的统计直方图
#        反映图像灰度值分布规律
# 直方图均衡化:调整像素的灰度分布规律,实现色彩或亮度调整
import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread("../data/sunrise.jpg", 0)
cv2.imshow("orig", im)

# 直方图均衡化处理
im_equ = cv2.equalizeHist(im)
cv2.imshow("im_equ", im_equ)

# 绘制图像灰度直方图
plt.subplot(2, 1, 1)
plt.hist(im.ravel(), # 返回一个扁平化统计的数组
         256, # 箱子数量(方条的数量)
         [0,256], label="orig")
plt.legend() # 图例


plt.subplot(2, 1, 2)
plt.hist(im_equ.ravel(), # 返回一个扁平化统计的数组
         256, # 箱子数量(方条的数量)
         [0,256], label="equalize")
plt.legend() # 图例

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()