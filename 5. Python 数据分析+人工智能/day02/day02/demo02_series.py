"""
demo02_series.py  
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/
"""
import pandas as pd
import numpy as np

# 测试Series
s = pd.Series()
print(s, type(s))

ary = np.array(['张三', '李四', '王五', '赵柳'])
s = pd.Series(ary, index=['S01', 'S02', 'S03', 'S04'])
print(s, type(s), s['S03'], s[2])
print(s[1:3])
print(s['S01':'S03'])
print(s[['S01', 'S03']])
print(s[[True, True, False, True]])

# 通过字典创建Series
data = {'zs':85, 'ls':95, 'ww': 90, 'zl': 83}
s = pd.Series(data)
print(s, s['zs'])

# 通过标量创建Series
s = pd.Series(5, index=np.arange(3, 10))
print(s)
s = pd.Series(np.ones(5))
print(s)

# 通过字典创建Series
data = {'zs':85, 'ls':95, 'ww': 90, 'zl': 83}
s = pd.Series(data)
print(s)
print(s.values)  # 把series转成ndarray
print(s.index)


# pandas识别的日期字符串格式
dates = pd.Series(['2011', '2011-02', '2011-03-01', '2011/04/01', 
                   '2011/05/01 01:01:01', '01 Jun 2011', '7/1/2011'])
dates = pd.to_datetime(dates)
print(dates)
# 减法运算
delta = dates - pd.to_datetime('2011.1.1')
print(delta)

# 获取日期对象的某个日历字段的字段值
print(dates.dt.dayofweek)

# 把日期偏移量换算为天  返回
print(delta.dt.days)

# 生成一组日期序列
dates = pd.date_range('20190101', periods=7, freq='D')
dates = pd.date_range('20190101', periods=7, freq='B')
dates = pd.date_range('20190101', '20190110')
print(dates)

