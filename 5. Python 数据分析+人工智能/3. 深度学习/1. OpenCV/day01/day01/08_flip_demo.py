# 08_flip_demo.py
# 图像翻转示例
import cv2

im = cv2.imread("../data/Linus.png")
cv2.imshow("im", im)

# 垂直镜像
im_flip0 = cv2.flip(im, 0)
cv2.imshow("im_flip0", im_flip0)

# 水平镜像
im_flip1 = cv2.flip(im, 1)
cv2.imshow("im_flip1", im_flip1)

cv2.waitKey()
cv2.destroyAllWindows()

