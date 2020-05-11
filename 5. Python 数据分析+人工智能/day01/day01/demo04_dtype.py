"""
demo04_dtype.py
"""
import numpy as np

data=[	('zs', [90, 80, 85], 15),
	    ('ls', [92, 81, 83], 16),
	    ('ww', [95, 85, 95], 15)]

ary = np.array(data, dtype='U2, 3int32, int32')
print(ary, ary.dtype)
print(ary[0])       # ('zs', [90, 80, 85], 15)
print(ary['f0'])    # ['zs' 'ls' 'ww']

# 设置dtype的第二种方式  为字段起别名
b = np.array(data, dtype=[('name', 'str', 2), ('scores', 'i4', 3), ('age', 'i4', 1)])
print(b, b.dtype)
print(b['scores'])
print(b[2]['age'])   # 索引为2的学员的年龄

# 设置dtype的第三种方案
c = np.array(data, dtype={'names':['name', 'scores', 'age'],  
                          'formats':['U3', '3i4', 'i4']})
print(c['scores'])
print(c[2]['age'])   # 索引为2的学员的年龄                  


# 测试时间类型数据
f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
print(f, f.dtype)
f = f.astype('M8[D]')  # 精确到“天”的日期类型
print(f, f.dtype)

print(f[-1] - f[0])

