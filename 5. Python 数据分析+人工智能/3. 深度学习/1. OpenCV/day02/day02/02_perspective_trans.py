# 02_perspective_trans.py
# 图像透视变换, 主要用于图像形状矫正
import cv2
import numpy as np

im = cv2.imread("../data/pers.png")
rows, cols = im.shape[:2]
print(rows, cols)

# 定义定点间的映射关系
## 原始图像的4个顶点
pts1 = np.float32([[58,2], [167,9], [8,196], [126,196]])
## 映射到新图像的4个顶点
pts2 = np.float32([[16,2], [167,8], [8,196], [169,196]])
# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换
dst = cv2.warpPerspective(im, # 原始图像
                          M, # 透视变换矩阵
                          (cols, rows)) # 输出图像大小
# 逆变换
M = cv2.getPerspectiveTransform(pts2, pts1)
dst2 = cv2.warpPerspective(dst, # 原始图像
                          M, # 透视变换矩阵
                          (cols, rows)) # 输出图像大小

cv2.imshow("im", im)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)

cv2.waitKey()
cv2.destroyAllWindows()
