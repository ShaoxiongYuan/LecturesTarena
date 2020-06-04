# 01_img_add.py
# 图像相加示例
import cv2

# 读取原始图像
a = cv2.imread("../data/lena.jpg", 0)
b = cv2.imread("../data/lily_square.png", 0)
# 简单相加
dst1 = cv2.add(a, b)
# 按照权重相加
dst2 = cv2.addWeighted(a, 0.6, # 图像1及权重
                       b, 0.4, # 图像2及权重
                       0) # 亮度调节量

cv2.imshow("lena", a)
cv2.imshow("lily", b)
cv2.imshow("add", dst1)
cv2.imshow("addWeighted", dst2)

cv2.waitKey()
cv2.destroyAllWindows()