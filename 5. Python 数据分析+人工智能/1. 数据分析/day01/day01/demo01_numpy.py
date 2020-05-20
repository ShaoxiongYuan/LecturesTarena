"""
demo01_numpy.py 
"""
import numpy as np

# 创建ndarray数组
ary = np.array([1,2,3,4,5,6])
print(ary, type(ary))

print(ary + 3)
print(ary + ary)
print(ary * 2)
print(ary > 3)

ary = np.array([ [1,2,3], [4,5,6] ])
print(ary)
print(ary * 2)