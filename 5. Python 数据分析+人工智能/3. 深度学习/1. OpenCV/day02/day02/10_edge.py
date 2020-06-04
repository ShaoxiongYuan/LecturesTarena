# 10_edge.py
# 边沿提取: Canny(提取外边沿)
#          Sobel(轮廓粗糙)
#          Laplacian(轮廓细腻)
import cv2

im = cv2.imread("../data/lily.png", 0)
cv2.imshow("im", im)

# Sobel
sobel = cv2.Sobel(im,  # 原始图像
                  cv2.CV_64F,  # 输出图像通道
                  1, 1,  # x,y方向上边沿提取
                  ksize=5)  # kernel大小
cv2.imshow("sobel", sobel)

# Laplacian
laplacian = cv2.Laplacian(im, cv2.CV_64F)
cv2.imshow("laplacian", laplacian)

# canny
canny = cv2.Canny(im,  # 原始图像
                  50,  # 滞后阈值
                  240)  # 模糊度
cv2.imshow("canny", canny)

cv2.waitKey()
cv2.destroyAllWindows()
