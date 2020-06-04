# 01_imread_demo.py
# 图像读取,显示,保存示例
import cv2

# 读取图像
im = cv2.imread("../data/Linus.png", # 图像路径
                0) # 1:彩色图像  0-灰度图像
cv2.imshow("test", im) # 参数分别为:窗体名称, 图像数据

# 打印图像数据的类型,形状
print(type(im)) # 类型
print(im.shape) # 形状

# 保存图像
cv2.imwrite("../data/Linus_new.png", im)

cv2.waitKey() # 等待用户敲击按键(阻塞函数,程序在此阻塞)
cv2.destroyAllWindows() # 销毁所有创建的窗体