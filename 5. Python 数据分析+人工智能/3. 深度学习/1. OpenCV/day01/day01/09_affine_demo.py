# 09_affine_demo.py
# 图像平移, 旋转(仿射变换)
import cv2
import numpy as np

def translate(img, x, y):
    """
    图像平移变换
    :param img 原始图像
    :param x 平移x坐标
    :param y 平移y坐标
    :return 返回经过平移后的图像数据
    """
    h, w = img.shape[:2] # 取图像高度, 宽度
    # 构建平移矩阵
    M = np.float32([[1, 0, x],[0, 1, y]])
    # 调用warpAffine进行仿射变换
    # 参数分别为: 原始图像, 变换矩阵, 输出图像大小
    shifted = cv2.warpAffine(img, M, (w, h)) 
    return shifted

def rotate(img, angle, center=None, scale=1.0):
    """
    图像旋转变换
    :param img 原始图像
    :param angle 旋转角度
    :param center 旋转中心,如果为None表示以原图中心为中心
    :param scale 缩放比率
    :return 返回经过旋转后的图像数据
    """   
    h, w = img.shape[:2] 
    # 计算旋转中心
    if center is None:
        center = (w/2, h/2)
    # 构建旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    shifted = cv2.warpAffine(img, M, (w, h)) 
    return shifted


if __name__ == "__main__":
    im = cv2.imread("../data/Linus.png") # 读取图像
    cv2.imshow("im", im)

    # 平移
    shifted1 = translate(im, -40, 50) #向下平移50像素
    cv2.imshow("shifted1", shifted1)

    # 旋转
    rotated = rotate(im, -30)
    cv2.imshow("rotated", rotated)

    cv2.waitKey()
    cv2.destroyAllWindows()
