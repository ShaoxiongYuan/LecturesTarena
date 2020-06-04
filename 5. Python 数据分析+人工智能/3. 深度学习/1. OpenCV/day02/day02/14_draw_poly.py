# 14_draw_poly.py
# 绘制逼近多边形
import cv2
import numpy as np

im = cv2.imread("../data/cloud.png")
cv2.imshow("im", im)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 灰度
ret, binary = cv2.threshold(gray, 127, 225, cv2.THRESH_BINARY)
img, contours, hierarchy = cv2.findContours(binary,
                                            cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)
# 精度一
adp = im.copy()
epsilon = 0.005 * cv2.arcLength(contours[0], True)  # 计算周长和精度
approx = cv2.approxPolyDP(contours[0],  # 要计算多边形的轮廓
                          epsilon,  # 精度
                          True)  # 是否封闭
cv2.drawContours(adp, [approx], 0, (0, 0, 255), 2)  # 绘制多边形
cv2.imshow("epsilon0.005", adp)

# 精度二
adp2 = im.copy()
epsilon = 0.012 * cv2.arcLength(contours[0], True)  # 计算周长和精度
approx = cv2.approxPolyDP(contours[0],  # 要计算多边形的轮廓
                          epsilon,  # 精度
                          True)  # 是否封闭
cv2.drawContours(adp2, [approx], 0, (0, 0, 255), 2)  # 绘制多边形
cv2.imshow("epsilon0.01", adp2)

cv2.waitKey()
cv2.destroyAllWindows()
