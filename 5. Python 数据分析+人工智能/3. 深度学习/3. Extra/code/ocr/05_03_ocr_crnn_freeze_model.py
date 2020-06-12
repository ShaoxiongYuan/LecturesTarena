
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
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

# 读取 label_list.txt 文件获取类别数量
class_dim = -1
all_file_dir = "data/data6927/word-recognition"
with codecs.open(os.path.join(all_file_dir, "label_list.txt")) as label_list:
    class_dim = len(label_list.readlines())
target_size = [1, 48, 512]
mean_rgb = 127.5
save_freeze_dir = "./crnn-model"


class CRNN(object):
    def __init__(self, num_classes, label_dict):
        self.outputs = None
        self.label_dict = label_dict
        self.num_classes = num_classes

    def name(self):
        return 'crnn'

    def conv_bn_pool(self, input, group, out_ch, act="relu", param=None, bias=None, param_0=None, is_test=False,
                     pooling=True, use_cudnn=False):
        tmp = input
        for i in six.moves.xrange(group):
            tmp = fluid.layers.conv2d(
                input=tmp,
                num_filters=out_ch[i],
                filter_size=3,
                padding=1,
                param_attr=param if param_0 is None else param_0,
                act=None,  # LinearActivation
                use_cudnn=use_cudnn)
            tmp = fluid.layers.batch_norm(
                input=tmp,
                act=act,
                param_attr=param,
                bias_attr=bias,
                is_test=is_test)
        if pooling:
            tmp = fluid.layers.pool2d(
                input=tmp,
                pool_size=2,
                pool_type='max',
                pool_stride=2,
                use_cudnn=use_cudnn,
                ceil_mode=True)

        return tmp

    def ocr_convs(self, input, regularizer=None, gradient_clip=None, is_test=False, use_cudnn=False):
        b = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0))
        w0 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0005))
        w1 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.01))
        tmp = input
        tmp = self.conv_bn_pool(
            tmp,
            2, [16, 16],
            param=w1,
            bias=b,
            param_0=w0,
            is_test=is_test,
            use_cudnn=use_cudnn)

        tmp = self.conv_bn_pool(
            tmp,
            2, [32, 32],
            param=w1,
            bias=b,
            is_test=is_test,
            use_cudnn=use_cudnn)
        tmp = self.conv_bn_pool(
            tmp,
            2, [64, 64],
            param=w1,
            bias=b,
            is_test=is_test,
            use_cudnn=use_cudnn)
        tmp = self.conv_bn_pool(
            tmp,
            2, [128, 128],
            param=w1,
            bias=b,
            is_test=is_test,
            pooling=False,
            use_cudnn=use_cudnn)
        return tmp

    def net(self, images, rnn_hidden_size=200, regularizer=None,
            gradient_clip=None, is_test=False, use_cudnn=True):
        conv_features = self.ocr_convs(
            images,
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            is_test=is_test,
            use_cudnn=use_cudnn)
        sliced_feature = fluid.layers.im2sequence(
            input=conv_features,
            stride=[1, 1],
            filter_size=[conv_features.shape[2], 1])

        para_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))
        bias_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))
        bias_attr_nobias = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))

        fc_1 = fluid.layers.fc(input=sliced_feature,
                               size=rnn_hidden_size * 3,
                               param_attr=para_attr,
                               bias_attr=bias_attr_nobias)
        fc_2 = fluid.layers.fc(input=sliced_feature,
                               size=rnn_hidden_size * 3,
                               param_attr=para_attr,
                               bias_attr=bias_attr_nobias)

        gru_forward = fluid.layers.dynamic_gru(
            input=fc_1,
            size=rnn_hidden_size,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu')
        gru_backward = fluid.layers.dynamic_gru(
            input=fc_2,
            size=rnn_hidden_size,
            is_reverse=True,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu')

        w_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))
        b_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0))

        fc_out = fluid.layers.fc(input=[gru_forward, gru_backward],
                                 size=self.num_classes + 1,
                                 param_attr=w_attr,
                                 bias_attr=b_attr)
        self.outputs = fc_out
        return fc_out

    def get_loss(self, label):
        cost = fluid.layers.warpctc(input=self.outputs, label=label, blank=self.num_classes, norm_by_times=True)
        sum_cost = fluid.layers.reduce_sum(cost)
        return sum_cost

    def get_infer(self):
        return fluid.layers.ctc_greedy_decoder(input=self.outputs, blank=self.num_classes)


def freeze_model():
    """
    保存模型
    :return:
    """
    exe = fluid.Executor(fluid.CPUPlace())
    image = fluid.layers.data(name='image', shape=target_size, dtype='float32')

    model = CRNN(class_dim, {})  # 创建CRNN模型
    pred = model.net(image)  # 组网
    out = model.get_infer()

    freeze_program = fluid.default_main_program()
    fluid.io.load_persistables(exe, save_freeze_dir, freeze_program)  # 加载模型
    freeze_program = freeze_program.clone(for_test=True)
    fluid.io.save_inference_model("./freeze-model", ['image'], out, exe, freeze_program)  # 保存模型


if __name__ == '__main__':
    freeze_model()