# 05_color_equalize_hist.py
# 彩色图像直方图均衡化处理
"""
imread, imshow默认对BGR色彩空间图像操作
图像读取出来后, 需要转换为带亮度通道的色彩空间
将亮度通道取出进行均衡化处理, 处理完成后再赋值给原亮度通道
"""
import cv2

# 读取的是BGR色彩空间
im = cv2.imread("../data/sunrise.jpg")
cv2.imshow("im", im)

# 将BGR ==> YUV色彩空间
yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
# 取出亮度通道,进行均衡化处理,并重新覆盖原来的亮度通道
yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
equal_hist = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR) #YUV ==> BGR
cv2.imshow("equal_hist", equal_hist)

cv2.waitKey()
cv2.destroyAllWindows()