# 11_tensorboard_demo.py
# 利用tensorboard工具实现可视化
import tensorflow as tf

a = tf.constant([1, 2, 3, 4, 5])  # 张量
var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0),
                  name="var")
b = tf.constant(3.0, name="a")  # 故意将名称起的不一样
c = tf.constant(4.0, name="b")
d = tf.add(b, c)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    # 将当前graph中的信息存入事件文件
    fw = tf.summary.FileWriter("../summary/",
                               graph=sess.graph)
    print(sess.run([a, var, d]))  # 执行指定的op
