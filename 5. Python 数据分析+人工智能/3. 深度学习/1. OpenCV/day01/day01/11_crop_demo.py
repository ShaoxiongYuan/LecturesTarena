# 11_crop_demo.py
# 利用数组切片实现图像裁剪
import numpy as np 
import cv2

# 随机裁剪
def random_crop(im, w, h):
    # 随机产生起始x, 范围0~w
    start_x = np.random.randint(0, im.shape[1])  
    # 随机产生起始y, 范围0~h
    start_y = np.random.randint(0, im.shape[0])
    # 切片
    new_img = im[start_y:start_y+h, start_x:start_x+w]

    return new_img

# 中心裁剪
def center_crop(im, w, h):
    start_x = int(im.shape[1] / 2) - int(w / 2)
    start_y = int(im.shape[0] / 2) - int(h / 2 )
    # 切片
    new_img = im[start_y:start_y+h, start_x:start_x+w]

    return new_img


if __name__ == "__main__":
    im = cv2.imread("../data/banana_1.png", 1)
    # 随机裁剪
    new_img = random_crop(im, 200, 200)
    cv2.imshow("random_crop", new_img)
    # 中心裁剪
    new_img = center_crop(im, 200, 200)
    cv2.imshow("center_crop", new_img)

    cv2.waitKey()
    cv2.destroyAllWindows()

    