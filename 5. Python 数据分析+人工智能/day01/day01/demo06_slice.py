"""
demo06_slice.py
"""
import numpy as np
a = np.arange(1, 10)
print(a)  # 1 2 3 4 5 6 7 8 9
print(a[:3])  # 1 2 3
print(a[3:6])   # 4 5 6
print(a[6:])  # 7 8 9
print(a[::-1])  # 9 8 7 6 5 4 3 2 1
print(a[:-4:-1])  # 9 8 7
print(a[-4:-7:-1])  # 6 5 4
print(a[-7::-1])  # 3 2 1
print(a[::])  # 1 2 3 4 5 6 7 8 9
print(a[:])  # 1 2 3 4 5 6 7 8 9
print(a[::3])  # 1 4 7
print(a[1::3])  # 2 5 8
print(a[2::3])  # 3 6 9


b = np.arange(1, 19).reshape(2, 3, 3)
print(b)
print(b[:, :2, :2])
print(b[:, ::2, ::2])

# 掩码操作 - 布尔掩码
a = np.arange(1, 10)
mask = a % 2 == 0
print(a)
print(mask)
print(a[mask])
a[mask] = 999
print(a)
# 100以内3的倍数
a = np.arange(100)
print(a[(a%7==0) & (a%3==0)])

# 索引掩码    a[索引]  a[索引数组]
products = np.array(['Apple', 'Mi', 'Huawei', 'Oppo'])
inds = [1, 3, 2, 0, 1,2,3,1,2,3,1,2,3,1,2,3,1,3]
print(products[inds])
