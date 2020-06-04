# 07_morphology_gradient.py
# 形态学梯度示例: 用膨胀的图像减去腐蚀的图像
import cv2
import numpy as np

im = cv2.imread("../data/6.png")
k = np.ones((3, 3), np.uint8)
r = cv2.morphologyEx(im,  # 原始图像
                     cv2.MORPH_GRADIENT,  # 形态学梯度
                     k)  # 运算核
cv2.imshow("im", im)
cv2.imshow("r", r)

cv2.waitKey()
cv2.destroyAllWindows()
