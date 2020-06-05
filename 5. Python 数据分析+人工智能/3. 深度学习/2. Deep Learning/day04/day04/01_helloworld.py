# 01_helloworld.py
# 打印helloworld
import tensorflow as tf

# 定义一个常亮(张量), 是一个操作
hello = tf.constant("Hello, world!")
# 创建一个Session, 作用是用来执行定义的操作
sess = tf.Session()
# 执行操作并打印结果
print(sess.run(hello))
sess.close()