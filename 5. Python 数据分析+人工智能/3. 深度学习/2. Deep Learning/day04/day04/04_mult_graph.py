# 04_mult_graph.py
# 一个程序中,包含多个多个graph, 指定运行那个graph
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 调整警告级别

# 下面的三个op放在默认graph中
a = tf.constant(5.0) # 张量a
b = tf.constant(1.0) # 张量b
c = tf.add(a, b) # 执行两个张量相加
# 定义一个graph
graph2 = tf.Graph()
with graph2.as_default(): # 在graph2上添加操作
    d = tf.constant(11.0) # d操作属于graph2

with tf.Session(graph=graph2) as sess: # 指定执行graph2图
    print(sess.run(d)) # OK
    # print(sess.run(c)) # 报错,c不属于graph2图
