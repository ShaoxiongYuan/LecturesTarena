"""
功能：胶囊缺陷识别
时间：2020/06/11
作者：Daniel.Wang
地点：北京.东城.桥湾
"""
import cv2 as cv
import numpy as np
import os
from scipy import signal
from scipy import misc
import scipy.ndimage as sn

def bub_check(img_path, img_file, im, im_gray):
    # 二值化处理
    ret, im_bin = cv.threshold(im_gray, 180, 255, cv.THRESH_BINARY)
    cv.imshow("im_bin", im_bin)

    # 提取轮廓
    img, contours, hierarchy = cv.findContours(im_bin,  # 二值化处理后的图像
                                               cv.RETR_LIST,  # 检测所有轮廓
                                               cv.CHAIN_APPROX_NONE)  # 存储所有的轮廓点

    # 计算轮廓面积，过滤掉面积过大的轮廓
    cnts = np.array(contours)
    all_area = []
    new_cnts = []

    if len(contours) > 0:
        for cnt in cnts:
            area = cv.contourArea(cnt)
            # all_area.append(area)
            if area < 10000:
                new_cnts.append(cnt)

    # 绘制轮廓
    im_cnt = cv.drawContours(im,  # 绘制图像
                             new_cnts,  # 轮廓点列表
                             -1,  # 绘制全部轮廓
                             (0, 0, 255),  # 轮廓颜色：红色
                             2)  # 轮廓粗细
    cv.imshow("im_cnt", im_cnt)

    # 将图片移动到子目录
    # if len(new_cnts) > 0:
    #     print(img_path, ":", "气泡瑕疵")
    #     new_path = os.path.join("capsules/bub", img_file)
    #     os.rename(img_path, new_path)
    #     print("文件移动成功:%s ==> %s" % (img_path, new_path))

def balance_check(img_path, img_file, im, im_gray):
    # 边沿提取，比较几种边沿提取算法，Canny效果较好，保留
    # sobel = cv.Sobel(im_gray, cv.CV_64F, 1, 1, ksize=5)
    # cv.imshow('Sobel', sobel)

    # # Laplacian滤波：对细节反映更明显
    # laplacian = cv.Laplacian(im_gray, cv.CV_64F)
    # cv.imshow('Laplacian', laplacian)

    # 二值化后效果并不好
    # ret, im_bin = cv.threshold(im_gray, 200, 255, cv.THRESH_BINARY)
    # cv.imshow("im_bin", im_bin)

    blurred = cv.GaussianBlur(im_gray, (5, 5), 0)

    # 膨胀
    dilate = cv.dilate(blurred,
                        cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))  # 根据函数返回kernel
    cv.imshow("dilate", dilate)

    # Canny边沿提取
    canny = cv.Canny(dilate,
                     60,  # 滞后阈值
                     200)  # 模糊度
    cv.imshow('Canny', canny)

    # 提取轮廓
    img, contours, hierarchy = cv.findContours(canny,  # 二值化处理后的图像
                                               cv.RETR_LIST,  # 检测所有轮廓
                                               cv.CHAIN_APPROX_NONE)  # 存储所有的轮廓点
    # 计算轮廓面积，过滤掉面积过大的轮廓
    cnts = np.array(contours)
    new_cnts = []

    if len(contours) > 0:
        for cnt in cnts:
            circle_len = cv.arcLength(cnt, True) # 计算周长
            if circle_len >= 1000: # 周长太小的过滤掉
                # print("circle_len:", circle_len)
                new_cnts.append(cnt)

    new_cnts = sorted(new_cnts,  # 数据
                  key=cv.contourArea,  # 排序依据，根据contourArea函数结果排序
                  reverse=True)

    new_cnts = new_cnts[1:2] # 取出面积第二的轮廓
    print("new_cnts.shape:", np.array(new_cnts).shape)

    # 绘制轮廓
    im_cnt = cv.drawContours(im,  # 绘制图像
                             new_cnts,  # 轮廓点列表
                             -1,  # 绘制全部轮廓
                             (0, 0, 255),  # 轮廓颜色：红色
                             2)  # 轮廓粗细
    cv.imshow("im_cnt", im_cnt)

    # 取出最大最小x,y
    max_x, max_y = new_cnts[0][0][0][0], new_cnts[0][0][0][1]
    min_x, min_y = max_x, max_y

    for cnt in new_cnts[0]:
        # print(cnt[0][0], cnt[0][1])
        if cnt[0][0] >= max_x:
            max_x = cnt[0][0]
        if cnt[0][0] <= min_x:
            min_x = cnt[0][0]
        if cnt[0][1] >= max_y:
            max_y = cnt[0][1]
        if cnt[0][1] <= min_y:
            min_y = cnt[0][1]
    print(" min_x:", min_x, " min_y:", min_y, "max_x:", max_x, " max_y:", max_y)

    # 绘制直线
    # cv.line(im, (min_x, min_y), (max_x, min_y), (0, 0, 255), 2)
    # cv.line(im, (min_x, min_y), (min_x, max_y), (0, 0, 255), 2)
    # cv.line(im, (max_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    # cv.line(im, (min_x, max_y), (max_x, max_y), (0, 0, 255), 2)
    mid_y = int((min_y+max_y)/2)
    mid_up = int((mid_y + min_y)/2) # 上半部分中线
    mid_down = int((mid_y + max_y)/2) # 下半部分中线
    cv.line(im, (min_x, mid_y), (max_x, mid_y), (0, 0, 255), 2)
    cv.line(im, (min_x, mid_up), (max_x, mid_up), (0, 0, 255), 2)
    cv.line(im, (min_x, mid_down), (max_x, mid_down), (0, 0, 255), 2)
    cv.imshow("im_line", im)

    # 求交点
    # cross_point_up = []
    # cross_point_down = []
    cross_point_up = set()
    cross_point_down = set()

    for cnt in new_cnts[0]:
        x, y =cnt[0][0], cnt[0][1]
        if y == mid_up:
            cross_point_up.add((x, y))

        if y == mid_down:
            cross_point_down.add((x, y))

    print("cross_point_up:", cross_point_up)
    print("cross_point_down:", cross_point_down)

    # 集合转列表
    cross_point_up = list(cross_point_up)
    cross_point_down = list(cross_point_down)

    # 绘制交点
    for p in cross_point_up:
        cv.circle(im,  # 绘制图像
                  (p[0], p[1]), 8,  # 圆心、半径
                   (0, 0, 255), 2)  # 颜色、粗细
    for p in cross_point_down:
        cv.circle(im,  # 绘制图像
                  (p[0], p[1]), 8,  # 圆心、半径
                   (0, 0, 255), 2)  # 颜色、粗细
    cv.imshow("im_cirle", im)

    # 计算长度
    # 集合无序，所以两个点可能前后顺序不确定，用x的差值绝对值
    len_up, len_down = 0, 0
    len_up = abs(cross_point_up[0][0] - cross_point_up[1][0]) # 上半部分中线长度
    len_down = abs(cross_point_down[0][0] - cross_point_down[1][0]) # 下半部分中线长度
    print("len_up:", len_up)
    print("len_down:", len_down)

    if abs(len_down - len_up) > 8: # 差值大于8像素
        print(img_path, ":上下不一样大")


if __name__ == "__main__":
    # 加载所有待预测图片
    img_dir = "capsules"
    img_files = os.listdir(img_dir)

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file) # 拼接完整路径
        if os.path.isdir(img_path): # 子目录直接跳过
            continue

        # 读取彩色图像和灰度图像
        im = cv.imread(img_path, 1)
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        cv.imshow("im", im)
        cv.imshow("im_gray", im_gray)

        # 气泡、黑点检测
        bub_check(img_path, img_file, im, im_gray)

        # 大小头检测
        # balance_check(img_path, img_file, im, im_gray)

        cv.waitKey()
        cv.destroyAllWindows()