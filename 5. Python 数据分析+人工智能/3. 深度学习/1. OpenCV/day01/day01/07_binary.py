# 07_binary.py
# 图像而值化和反二值化示例
import cv2

im = cv2.imread("../data/lena.jpg", 0)
cv2.imshow("im", im) 

# 二值化处理
t, rst = cv2.threshold(im,  # 原始图像
                            127, # 阈值
                            255, # 超过阈值全部设置为255
                            cv2.THRESH_BINARY) # 二值化处理 
cv2.imshow("rst", rst)

# 反二值化
t, rst2 = cv2.threshold(im,127,255, cv2.THRESH_BINARY_INV)  
cv2.imshow("rst2", rst2)

cv2.waitKey()
cv2.destroyAllWindows()