# 06_create_tensor.py
# 创建张量示例
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 调整警告级别

# 创建值全为0的张量
tensor_zeros = tf.zeros(shape=[2,3], dtype="float32")
# 创建值全为1的张量
tensor_ones = tf.ones(shape=[3,4], dtype="float32")
# 创建满足正态分布的张量
tensor_nd = tf.random_normal(shape=[10], # 1维10个元素
                             mean=1.7, # 中位数
                             stddev=0.2, # 标准差
                             dtype="float32") # 元素类型
# 产生和另一个张量形状相同的张量, 值全为0
tensor_zeros_like = tf.zeros_like(tensor_ones)

with tf.Session() as sess:
    print(tensor_zeros.eval()) # eval表示在session下执行该操作
    print(tensor_ones.eval())
    print(tensor_nd.eval())
    print(tensor_zeros_like.eval())
