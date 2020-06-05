# 10_variable_demo.py
# 变量使用示例
"""
变量: 变量存储的值是张量
     变量的值可以持久化保存
     变量使用前要执行初始化操作
"""
import tensorflow as tf

a = tf.constant([1, 2, 3, 4, 5])
var = tf.Variable(tf.random_normal([2,3], mean=0.0,stddev=1.0),
                  name="var") # 变量名称
init_op = tf.global_variables_initializer() # 初始化op

with tf.Session() as sess:
    sess.run(init_op) # 执行init_op操作
    print(sess.run([a, var])) # 执行a和var这两个op
