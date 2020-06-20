# 铁轨交叉点、弯道识别
import cv2 as cv
import numpy as np
import math


# 根据线段两点，计算线段所有点坐标，并返回
def calc_line_points(x1, y1, x2, y2):
    k = float(y2 - y1) / float(x2 - x1)  # 斜率
    b = float(y1) - float(x1) * k  # 偏置

    points = []
    for x in range(x1, x2 + 1):
        y = int(k * x + b) # 计算每个点的坐标
        points.append((x, y))

    # print(points)
    return points


def cross_point(line1, line2):  # 计算交点函数
    # 是否存在交点
    point_is_exist = False
    x = 0
    y = 0
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]  # 取四点坐标
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    # 计算两条直线斜率
    k1 = float(y2 - y1) / float(x2 - x1)
    # print("k1:", k1)
    k2 = float(y4 - y3) / float(x4 - x3)
    # print("k2:", k2)

    # 如果斜率接近，则认为是平行线
    if abs(k1 - k2) < 0.3:
        # print("平行线 k1:%f, k2:%f" % (k1, k2))
        return False, [0, 0]

    # 如果一条线段x最大值，小于另一条线段x最小值，则不可能产生交点
    if max(x1, x2) < min(x3, x4):
        return False, [0, 0]
    if max(x3, x4) < min(x1, x2):
        return False, [0, 0]

    # 如果一条线段y最大值，小于另一条线段y最小值，则不可能产生交点
    if max(y1, y2) < min(y3, y4):
        return False, [0, 0]
    if max(y3, y4) < min(y1, y2):
        return False, [0, 0]

    # 根据斜率，计算直线的所有点
    points_1 = calc_line_points(x1, y1, x2, y2)
    points_2 = calc_line_points(x3, y3, x4, y4)
    # print(points_1)
    # print(points_2)

    # 判断两条直线是否有交点
    i = 0
    for p1 in points_1:
        for p2 in points_2:
            x1, y1 = p1
            x2, y2 = p2
            if int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)) < 4:  # 求两点集合距离
            # if x1 == x2 and y1 == y2:
                # print("p1:", p1, " p2:", p2)
                return True, [x1, y1]
        i += 1

    return False, [0, 0]


def cross_check(img_path):
    im = cv.imread(img_path, 1)
    cv.imshow("im", im)

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imwrite("out/im_gray.png", im_gray)

    # 模糊
    # im_mean_blur = cv.GaussianBlur(im, (3, 3), 3)
    # cv.imshow("mean_blur", im_mean_blur)

    # 对图像进行卷积
    flt = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    im_conv = cv.filter2D(im_gray, -1, flt, borderType=1)
    cv.imshow("im_conv", im_conv)
    cv.imwrite("out/im_conv.png", im_conv)

    # 二值化处理
    ret, im_bin = cv.threshold(im_conv, 180, 255, cv.THRESH_BINARY)
    cv.imshow("im_bin", im_bin)
    cv.imwrite("out/im_bin.png", im_bin)

    # 霍夫变换
    lines = cv.HoughLinesP(im_bin, 1, np.pi / 180, 1, minLineLength=70, maxLineGap=20)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        print("x1:", x1, " y1:", y1, " x2:", x2, " y2:", y2)
        cv.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("im_line", im)
    cv.imwrite("out/im_line.png", im)

    # 绘制交点
    lines1 = lines[:, 0, :]
    for x1, y1, x2, y2 in lines1[:]:
        for x3, y3, x4, y4 in lines1[:]:
            point_is_exist, [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
            if point_is_exist:
                print("发现交点:", int(x), int(y))
                cv.circle(im, (int(x), int(y)), 5, (0, 0, 255), 3)
    cv.imshow('Result', im)
    cv.imwrite("out/Result.png", im)

    # Canny边沿提取
    # canny = cv.Canny(im_conv,
    #                  10,  # 滞后阈值
    #                  200)  # 模糊度
    # cv.imshow('Canny', canny)


if __name__ == "__main__":
    # 交叉点检测
    cross_check("../data/rail_1.png")

    cv.waitKey()
    cv.destroyAllWindows()
