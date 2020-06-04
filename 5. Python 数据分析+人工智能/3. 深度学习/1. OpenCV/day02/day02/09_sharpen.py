# 09_sharpen.py
# 图像锐化处理: 增大像素之间的差异
import cv2
import numpy as np

im = cv2.imread("../data/lena.jpg", 0)
cv2.imshow("im", im)

# 定义锐化算子,进行锐化操作
sharpen_1 = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
im_sharpen1 = cv2.filter2D(im,  # 原始图像
                           -1,  # 通道
                           sharpen_1)  # 锐化算子
cv2.imshow("im_sharpen1", im_sharpen1)

# 锐化算子2
# 锐化算子2
sharpen_2 = np.array([[0, -1, 0],
                      [-1, 8, -1],
                      [0, 1, 0]]) / 4.0
im_sharpen2 = cv2.filter2D(im,  # 原始图像
                           -1,  # 通道
                           sharpen_2)  # 锐化算子
cv2.imshow("im_sharpen2", im_sharpen2)

cv2.waitKey()
cv2.destroyAllWindows()
