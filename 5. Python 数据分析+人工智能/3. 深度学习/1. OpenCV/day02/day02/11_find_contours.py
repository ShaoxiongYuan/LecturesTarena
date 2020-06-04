# 11_find_contours.py
# 查找和绘制轮廓
import cv2
import numpy as np

im = cv2.imread("../data/3.png")
cv2.imshow("im", im)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 转灰度
ret, binary = cv2.threshold(gray, 127, 255,
                            cv2.THRESH_BINARY)  # 二值化处理
# 查找轮廓
# img: 返回原图像
# contours: 列表,每个元素存储了一系列轮廓点
# hierachy: 轮廓层次关系
img, contours, hierachy = cv2.findContours(binary,  # 图像
                                           cv2.RETR_EXTERNAL,  # 外部轮廓
                                           cv2.CHAIN_APPROX_NONE)  # 存储所有轮廓点
# 列表转换为数组,并打印形状
# arr_cnt = np.array(contours)
# for cnt in arr_cnt:
#     print(cnt.shape)
#     for point in cnt:
#         print(point)

# 绘制轮廓
im_cnt = cv2.drawContours(im,  # 在哪个图像上绘制
                          contours,  # 通过findContours返回的轮廓列表
                          1,  # 要绘制的轮廓的索引,-1全部绘制
                          (0, 0, 255),  # 轮廓颜色(红色)
                          2)  # 轮廓粗细
cv2.imshow("im_cnt", im_cnt)

cv2.waitKey()
cv2.destroyAllWindows()
