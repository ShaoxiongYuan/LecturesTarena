# 07_placeholder_demo.py
# 占位符使用示例
import tensorflow as tf

# 不确定数据内容,先占一个位置
plhd = tf.placeholder(tf.float32, [2, 3]) # 2行3列
plhd2 = tf.placeholder(tf.float32, [None, 3]) # N行3列

with tf.Session() as sess:
    d = [[1, 2, 3],
         [4, 5, 6]]
    print(sess.run(plhd, # 要执行的op
                   feed_dict={plhd: d})) # 喂入参数
    print(sess.run(plhd2,
                   feed_dict={plhd2: d}))
