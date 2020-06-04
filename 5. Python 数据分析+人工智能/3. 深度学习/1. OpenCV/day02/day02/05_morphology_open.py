# 05_morphology_open.py
# 开运算: 先腐蚀, 再膨胀
import cv2
import numpy as np

# 读取原始图像
im1 = cv2.imread("../data/7.png")
im2 = cv2.imread("../data/8.png")
# 执行开运算
k = np.ones((8,8), np.uint8)
r1 = cv2.morphologyEx(im1, # 原始图像
                      cv2.MORPH_OPEN, # 做开运算
                      k) # 运算核
r2 = cv2.morphologyEx(im2, # 原始图像
                      cv2.MORPH_OPEN, # 做开运算
                      k) # 运算核
cv2.imshow("im1", im1)
cv2.imshow("im2", im2)
cv2.imshow("r1", r1)
cv2.imshow("r2", r2)

cv2.waitKey()
cv2.destroyAllWindows()
