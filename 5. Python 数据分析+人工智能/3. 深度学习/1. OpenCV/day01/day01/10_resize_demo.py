# 10_resize_demo.py
# 图像缩放示例
import cv2
import numpy as np

im = cv2.imread("../data/Linus.png")
cv2.imshow("im", im)

h, w = im.shape[:2] # 取出高度,宽度

# 计算缩小尺寸
dst_size = (int(w/2), int(h/2))
resized = cv2.resize(im, dst_size) # 执行缩放
cv2.imshow("reduce", resized)

# 放大: 最邻近插值法
dst_size = (200, 300)
method = cv2.INTER_NEAREST # 最邻近插值法
resized = cv2.resize(im, dst_size, interpolation=method)
cv2.imshow("NEAREST", resized)

# 放大: 双线性插值法
method = cv2.INTER_LINEAR # 双线性插值法
resized = cv2.resize(im, dst_size, interpolation=method)
cv2.imshow("LINEAR", resized)


cv2.waitKey()
cv2.destroyAllWindows()