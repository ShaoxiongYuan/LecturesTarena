# 13_draw_circle.py
# 绘制最小外接圆
import cv2
import numpy as np

im = cv2.imread("../data/cloud.png", 0)
cv2.imshow("im", im)

ret, binary = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
img, contours, hierarchy = cv2.findContours(binary,
                                            cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)
# 计算轮廓的最小外接圆形
# (x, y), radius = cv2.minEnclosingCircle(contours[0])
# center = (int(x), int(y)) # 圆心转换为整型
# radius = int(radius)
# cv2.circle(im, center, radius, # 原始图像,圆心,半径
#            (255,255,255), 2) # 颜色,线条粗细

# 计算并绘制最优拟合椭圆
ellipse = cv2.fitEllipse(contours[0])  # 拟合最优椭圆
print("ellipse:", ellipse)
cv2.ellipse(im, ellipse, (0, 0, 255), 2)  # 绘制椭圆

cv2.imshow("result", im)

cv2.waitKey()
cv2.destroyAllWindows()

