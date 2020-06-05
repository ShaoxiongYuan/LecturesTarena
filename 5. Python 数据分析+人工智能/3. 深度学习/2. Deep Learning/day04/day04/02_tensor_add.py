# 02_tensor_add.py
# 张量相加
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 调整警告级别

a = tf.constant(5.0) # 定义张量a
b = tf.constant(1.0) # 定义张量b
c = tf.add(a, b) # 执行两个张量相加

with tf.Session() as sess:
    print(sess.run(c))
