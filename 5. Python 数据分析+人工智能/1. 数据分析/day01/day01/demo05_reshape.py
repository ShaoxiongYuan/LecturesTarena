"""
demo05_reshape.py  变维
"""
import numpy as np
# 视图变维  reshape()  ravel()
a = np.arange(1, 19)
print(a, a.shape)
b = a.reshape(2, 3, 3)
print(b, b.shape)
b[0,0,0] = 999
print(a, a.shape)
print(b.ravel())
# 复制变维 flatten()
print(b.flatten())
# resize()
b.resize(2, 9)
print(b)


