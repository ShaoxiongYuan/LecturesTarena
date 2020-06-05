# 09_math_demo.py
# 张量数学计算示例
import tensorflow as tf

x = tf.constant([[1, 2],
                 [3, 4]], dtype=tf.float32)
y = tf.constant([[4, 3],
                 [3, 2]], dtype=tf.float32)
x_add_y = tf.add(x, y) # 张量相加, 对应元素相加
x_mul_y = tf.matmul(x, y) # 按照矩阵规则相乘
log_x = tf.log(x)
x_sum = tf.reduce_sum(x, axis=1) # 0-列方向 1-行方向

data = tf.constant([1,2,3,4,5,6,7,8,9,10], dtype=tf.float32)
segment_idx = tf.constant([0,0,0,1,1,2,2,2,2,2], dtype=tf.int32)
x_seg_sum = tf.segment_sum(data, segment_idx)#分段求和

with tf.Session() as sess:
    print(x_add_y.eval())
    print(x_mul_y.eval())
    print(log_x.eval())
    print(x_sum.eval())
    print(x_seg_sum.eval())