

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import time
import codecs
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.param_attr import ParamAttr
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

target_size = [1, 48, 512]
mean_rgb = 127.5
data_dir = 'data/data6927/word-recognition'
eval_file = "eval.txt"
label_list = "label_list.txt"
use_gpu = True
label_dict = {}
save_freeze_dir = "./freeze-model"

place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 加载模型
[inference_program, feed_target_names, fetch_targets] = \
    fluid.io.load_inference_model(dirname=save_freeze_dir, executor=exe)


# print(fetch_targets)


def init_eval_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    label_list_path = os.path.join(data_dir, label_list)
    index = 0

    # 读取样本文件内容，并存入字典
    with codecs.open(label_list_path, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.split()
            label_dict[int(parts[1])] = parts[0]


def resize_img(img):
    """
    重设图像大小
    :param img:
    :return:
    """
    percent_h = float(target_size[1]) / img.size[1]
    percent_w = float(target_size[2]) / img.size[0]
    percent = min(percent_h, percent_w)

    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))

    w_off = (target_size[2] - resized_width) / 2
    h_off = (target_size[1] - resized_height) / 2

    img = img.resize((resized_width, resized_height), Image.ANTIALIAS)

    array = np.ndarray((target_size[1], target_size[2], 3), np.uint8)
    array[:, :, 0] = 127
    array[:, :, 1] = 127
    array[:, :, 2] = 127
    ret = Image.fromarray(array)
    ret.paste(img, (np.random.randint(0, w_off + 1), int(h_off)))
    return ret


def read_image(img_path):
    """
    读取图像
    :param img_path:
    :return:
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_img(img)
    img = img.convert('L')  # 返回一个转换后的副本，L模式进行转换
    img = np.array(img).astype('float32') - mean_rgb
    img = img[..., np.newaxis]
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img


def infer(image_path):
    """
    执行预测
    :param image_path:
    :return:
    """
    tensor_img = read_image(image_path)

    label = exe.run(inference_program,
                    feed={feed_target_names[0]: tensor_img},
                    fetch_list=fetch_targets,
                    return_numpy=False)
    label = np.array(label[0])
    ret = ""
    if label[0] != -1:
        ret = ret.join([label_dict[int(c[0])] for c in label])
    return ret


def eval_all():
    """
    评估所有
    :return:
    """
    eval_file_path = os.path.join(data_dir, eval_file)  # 评估文件路径
    total_count = 0
    right_count = 0

    # with codecs.open(eval_file_path, encoding='utf-8') as flist:
    #     lines = [line.strip() for line in flist]
    #     t1 = time.time()
    #     random.shuffle(lines) # 打乱样本
    #
    #     i = 0
    #     for line in lines:
    #         i += 1
    #         if i > 3:
    #             break
    #
    #         total_count += 1
    #         parts = line.strip().split()
    #
    #         result = infer(parts[0])  # 执行推测
    #
    #         img = Image.open(parts[0])
    #         plt.imshow(img)
    #         plt.show()
    #
    #         print("infer result:{0} answer:{1}".format(result, parts[1]))
    #
    #         if str(result) == parts[1]:
    #             right_count += 1
    #
    #     period = time.time() - t1
    #     print("total eval count:{0} cost time:{1} predict accuracy:{2}".format(total_count, "%2.2f sec" % period,
    #                                                                            right_count / total_count))

    # 测试自定义图片
    img_file = "1.png"
    result = infer(img_file)  # 执行推测
    print("infer result:{0}".format(result))

    img = Image.open(img_file)
    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    init_eval_parameters()
    eval_all()