"""
demo02_ndarray.py 
"""
import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a)

# arange()
b = np.arange(1, 10, 2)
print(b)

# zeros()
c = np.zeros(10, dtype='int32')
print(c)

# ones()    构建一个2行4列的全1数组
d = np.ones( (2, 4), dtype='int32')
print(d)

# zeros_like()   ones_like()
e = np.zeros_like(d)
print(e)
e = np.ones_like(a)
print(e)

print(np.ones(5)/5)


