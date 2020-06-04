# 06_morphology_close.py
# 闭运算: 先膨胀, 再腐蚀
import cv2
import numpy as np

# 读取原始图像
im1 = cv2.imread("../data/9.png")
im2 = cv2.imread("../data/10.png")
# 执行闭运算
k = np.ones((8,8), np.uint8)
r1 = cv2.morphologyEx(im1, # 原始图像
                      cv2.MORPH_CLOSE, # 做闭运算
                      k, iterations=2) # 运算核
r2 = cv2.morphologyEx(im2, # 原始图像
                      cv2.MORPH_CLOSE, # 做闭运算
                      k, iterations=2) # 运算核
cv2.imshow("im1", im1)
cv2.imshow("im2", im2)
cv2.imshow("r1", r1)
cv2.imshow("r2", r2)

cv2.waitKey()
cv2.destroyAllWindows()
