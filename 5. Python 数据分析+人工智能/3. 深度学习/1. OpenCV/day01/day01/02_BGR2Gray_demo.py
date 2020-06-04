# 02_BGR2Gray_demo.py
# BGR色彩空间图像转换灰度图像
import cv2

im = cv2.imread("../data/Linus.png", 1) # 读取彩色图像
cv2.imshow("BGR", im) # 显示彩色图像

# 使用cvtColor函数进行色彩空间转换
img_gray = cv2.cvtColor(im, # 原始图像数据
                        cv2.COLOR_BGR2GRAY) # BGR ==> Gray
cv2.imshow("img_gray", img_gray) # 显示转换后的图像

cv2.waitKey()
cv2.destroyAllWindows()

# 课堂练习(编写该案例):3分钟