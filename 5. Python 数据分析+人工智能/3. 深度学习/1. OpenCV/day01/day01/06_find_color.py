# 06_find_color.py
# 从图像中提取特定色彩区域
# 原理: 使用inRange函数找到满足条件的部分,再做一个掩模运算
import cv2
import numpy as np

im = cv2.imread("../data/opencv2.png")
cv2.imshow("im", im)
# BGR ==> HSV
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

###### 取蓝色 ######
# 制定蓝色颜色范围, 由三个值构成:色相,饱和度,亮度
minBlue = np.array([110, 50, 50])
maxBlue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, minBlue, maxBlue) # 选取出满足条件区域
# cv2.imshow("mask", mask)
blue = cv2.bitwise_and(im, im, mask=mask) # 执行掩模运算
cv2.imshow("blue", blue)

###### 取红色 ######
minRed = np.array([0, 50, 50])
maxRed = np.array([30, 255, 255])
mask = cv2.inRange(hsv, minRed, maxRed) # 选取出满足条件区域
red = cv2.bitwise_and(im, im, mask=mask) # 执行掩模运算
cv2.imshow("red", red)

cv2.waitKey()
cv2.destroyAllWindows()
