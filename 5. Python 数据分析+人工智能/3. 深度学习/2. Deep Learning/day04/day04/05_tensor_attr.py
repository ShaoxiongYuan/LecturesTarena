# 05_tensor_attr.py
# 查看张量属性
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 调整警告级别

a = tf.constant(5.0) # 定义张量

with tf.Session() as sess:
    print(sess.run(a))
    print("name:", a.name) # name属性
    print("dtype:", a.dtype) # dtype属性
    print("shape:", a.shape) # shape属性
    print("graph:", a.graph) # graph属性
    print("op:", a.op) # op属性