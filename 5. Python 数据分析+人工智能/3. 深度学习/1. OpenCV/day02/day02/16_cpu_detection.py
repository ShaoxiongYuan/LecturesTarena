# 16_cpu_detection.py
# 芯片表面瑕疵检测
import cv2
import numpy as np
import math

im = cv2.imread("../data/CPU3.png")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # 灰度化
cv2.imshow("gray", gray)

# 二值化处理
ret, im_bin = cv2.threshold(gray, 162, 255, cv2.THRESH_BINARY)
cv2.imshow("im_bin", im_bin)

# 查找轮廓, 进行实心填充
img, contours, hie = cv2.findContours(im_bin,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
mask = np.zeros(im_bin.shape, np.uint8) #产生值全为0的数组
mask = cv2.drawContours(mask, contours,
                        -1, (255,0,0), -1)
cv2.imshow("mask", mask)

# 二值化图像和实心填充图像相减
im_sub = cv2.subtract(mask, im_bin)
cv2.imshow("im_sub", im_sub)

# 图像做闭运算(先膨胀, 再腐蚀), 去除瑕疵内部空洞
k = np.ones((10,10), np.uint8)
im_close = cv2.morphologyEx(im_sub,
                            cv2.MORPH_CLOSE,
                            k, iterations=3)
cv2.imshow("im_close", im_close)

# 提取瑕疵区域的轮廓, 绘制最小外接圆形
img, contours, hie = cv2.findContours(im_close,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
(x, y), radius = cv2.minEnclosingCircle(contours[1])
center = (int(x), int(y))
radius = int(radius)
print("center:", center, " radius:", radius)
cv2.circle(im_close, center, radius, (255,0,0), 2)#绘制
cv2.imshow("min_circle", im_close)

# 根据计算的圆心,半径在原始图像上绘制出瑕疵区域
cv2.circle(im, center, radius, (0,0,255), 2)#绘制红色圆圈
cv2.imshow("im_circle", im)

# 计算最小包围圆的面积
area = math.pi * radius * radius
print("area:", area)
if area > 12:
    print("度盘表面有瑕疵, 瑕疵区域大小:", area)

cv2.waitKey()
cv2.destroyAllWindows()