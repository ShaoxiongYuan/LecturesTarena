# 04_dilate_demo.py
# 图像膨胀示例
import cv2
import numpy as np

im = cv2.imread("../data/6.png")
cv2.imshow("im", im)

# 膨胀
kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(im,
                      kernel,
                      iterations=12)
cv2.imshow("dilation", dilation)

cv2.waitKey()
cv2.destroyAllWindows()