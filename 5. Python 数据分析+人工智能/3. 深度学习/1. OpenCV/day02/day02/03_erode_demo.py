# 03_erode_demo.py
# 图像腐蚀
import cv2
import numpy as np

im = cv2.imread("../data/5.png")
cv2.imshow("im", im)

# 腐蚀
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(im, # 原始图像
                    kernel, # 腐蚀核
                    iterations=3) # 迭代次数
cv2.imshow("erosion", erosion)

cv2.waitKey()
cv2.destroyAllWindows()