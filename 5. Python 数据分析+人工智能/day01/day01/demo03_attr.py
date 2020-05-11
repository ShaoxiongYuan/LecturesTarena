"""
demo03_attr.py 属性
"""
import numpy as np

# shape属性
a = np.arange(1, 9)
print(a, a.shape)
a.shape = (2, 2, 2)
print(a, a.shape)

# dtype属性
b = np.arange(1, 5)
print(b, b.dtype)
# b.dtype = 'float32'
# print(b, b.dtype)
c = b.astype('str')
print(c, c.dtype)

# size
print(a, a.size, len(a))

# 基于索引的元素访问
d = np.arange(1, 13)
d.shape = (2 , 2, 3)
print(d[0])
print(d[0][0])
print(d[0][0][0])
print(d[0][1][1])
print(d[0, 1, 1])



