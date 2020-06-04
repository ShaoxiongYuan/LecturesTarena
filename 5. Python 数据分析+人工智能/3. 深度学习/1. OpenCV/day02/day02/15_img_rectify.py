# 15_img_rectify.py
# 利用透视变换,对纸张进行形状矫正
import cv2
import numpy as np

im = cv2.imread("../data/paper.jpg")
cv2.imshow("im", im)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# 模糊化, 膨胀处理, 处理掉细小的边沿
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
# cv2.imshow("dilate", dilate)

# 检测边沿
edged = cv2.Canny(dilate,
                  30, 120,  # 滞后阈值, 模糊度
                  3)  # 孔径大小
# cv2.imshow("edged", edged)

# 检测轮廓
cnts = cv2.findContours(edged.copy(),  # 在提取边沿上检测轮廓
                        cv2.RETR_EXTERNAL,  # 只检测外轮廓
                        cv2.CHAIN_APPROX_SIMPLE)  # 保留终点坐标
cnts = cnts[1]
im_cnt = cv2.drawContours(im,  # 绘制图像
                          cnts,  # 检测到的轮廓
                          -1,  # 绘制所有轮廓
                          (0, 0, 255), 2)  # 颜色和宽度
cv2.imshow("im_cnt", im_cnt)

# 计算面积, 排序, 并使用多边形进行拟合
docCnt = None  # 纸张轮廓
if len(cnts) > 0:
    cnts = sorted(cnts,  # 可迭代对象
                  key=cv2.contourArea,  # 计算面积作为排序依据
                  reverse=True)  # 逆序排列
    for c in cnts:
        peri = cv2.arcLength(c, True)  # 计算轮廓周长
        # 多边形拟合
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # 拟合出来的为4边形
            docCnt = approx  # 找到纸张轮廓
            break

# 用圆形标记角点
# 用圆圈标记处角点
points = []  # 角点
for peak in docCnt:
    peak = peak[0]
    # 绘制圆
    cv2.circle(im, tuple(peak), 5, (0, 0, 255), 2)
    points.append(peak)  # 添加到列表
# cv2.imshow("im_point", im)
print(points)

# 使用透视变换进行矫正
src = np.float32([points[0], points[1], points[2], points[3]])
dst = np.float32([[0, 0], [0, 488], [337, 488], [337, 0]])
m = cv2.getPerspectiveTransform(src, dst)  # 计算变换矩阵
result = cv2.warpPerspective(gray.copy(),
                             m,
                             (337, 488))  # 透视变换
cv2.imshow("result", result)

cv2.waitKey()
cv2.destroyAllWindows()
