# 08_blur.py
# 图像模糊化处理
import cv2
import numpy as np

im = cv2.imread("../data/lena.jpg", 0)
cv2.imshow("im", im)

## 中值滤波
im_median_blur = cv2.medianBlur(im, 5) #5为滤波模板大小
cv2.imshow("median_blur", im_median_blur)
## 均值滤波
im_mean_blur = cv2.blur(im, (3,3)) # 第二个参数kernel大小
cv2.imshow("mean_blur", im_mean_blur)
## 高斯滤波
# im_gaussian_blur = cv2.GaussianBlur(im, (3,3),
#                                     3) # x方向上的标准差
# cv2.imshow("gaussian_blur", im_gaussian_blur)

# 使用高斯算子和filter2D自定义滤波操作
gaussan_blur = np.array([
                [1, 4, 7, 4, 1],
                [4, 16, 26, 16, 4],
                [7, 26, 41, 26, 7],
                [4, 16, 26, 16, 4],
                [1, 4, 7, 4, 1]], np.float32) / 273
# 使用filter2D, 第二个参数为目标图像的所需深度, -1表示和原图像相同
im_gaussian_blur2 = cv2.filter2D(im, -1, gaussan_blur)
cv2.imshow("gaussian_blur2", im_gaussian_blur2)

cv2.waitKey()
cv2.destroyAllWindows()