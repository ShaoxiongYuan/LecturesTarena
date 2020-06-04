# 12_draw_rect.py
# 绘制矩形示例
import cv2
import numpy as np

im = cv2.imread("../data/cloud.png", 0)
cv2.imshow("im", im)
ret, binary = cv2.threshold(im, 127, 255,
                            cv2.THRESH_BINARY)  # 二值化处理
# 查找轮廓
img, contours, hierarchy = cv2.findContours(binary,
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_NONE)
# 根据轮廓计算包围矩形的坐标
x, y, w, h = cv2.boundingRect(contours[0])
print(x,y,w,h)
# 绘制矩形
brcnt = np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]])
cv2.drawContours(im, [brcnt], -1, [255,255,255], 2)
cv2.imshow("result", im)

cv2.waitKey()
cv2.destroyAllWindows()