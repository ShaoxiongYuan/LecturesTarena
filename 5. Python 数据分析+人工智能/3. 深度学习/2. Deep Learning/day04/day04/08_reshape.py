# 08_reshape.py
# 张量形状设置
"""
静态形状: 张量的初始形状, 一旦固定就不能改变
         不能进行跨阶设置
动态形状: 张量计算过程中使用, 可以进行多次设设置
        可以跨阶设置, 实际是产生一个新的张量并返回
        动态形状设置时, 元素个数要保持一致
"""
import tensorflow as tf

pld = tf.placeholder(tf.float32, [None,3])
print(pld)
pld.set_shape([4,3]) # 设置静态形状
print(pld)
# pld.set_shape([3,3]) # 报错

# 动态形状
new_pld = tf.reshape(pld, [3, 4])
print(new_pld)
new_pld = tf.reshape(pld, [2, 6])
print(new_pld)
# new_pld = tf.reshape(pld, [2, 4]) # 报错,元素个数不一致