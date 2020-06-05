# 03_graph.py
# 查看op/tensor/session的graph属性
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 调整警告级别

a = tf.constant(5.0) # 张量a
b = tf.constant(1.0) # 张量b
c = tf.add(a, b) # 执行两个张量相加

graph = tf.get_default_graph() # 获取默认graph
print(graph)

with tf.Session() as sess:
    print(sess.run(c)) # 执行c操作
    print(a.graph)
    print(c.graph)
    print(sess.graph)