# 03_BGR_Channel.py
# 对BGR彩色图像某个通道进行操作
import cv2
import numpy as np

# 读取图像
im = cv2.imread("../data/opencv2.png")
print(im.shape)
cv2.imshow("im", im)

# 通过数组切片,取出蓝色通道,显示为单通道图像
b = im[:, :, 0] # 索引为0的通道:B
print(b.shape)
cv2.imshow("b", b)

# 去掉蓝色通道
im[:, :, 0] = 0 # 将蓝色通道的值全置为0
cv2.imshow("im-b0", im)

# 摸去绿色通道
im[:, :, 1] = 0 # 将绿色通道的值全置为0
cv2.imshow("im-g0", im)

cv2.waitKey()
cv2.destroyAllWindows()