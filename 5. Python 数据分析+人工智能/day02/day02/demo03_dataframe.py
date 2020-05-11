"""
demo03_dataFrame.py
"""
import pandas as pd
import numpy as np

df = pd.DataFrame()
print(df)

# 通过列表创建DataFrame
data = ['王伟超', '王小超', '王大超', '王年轻超']
df = pd.DataFrame(data)
print(df)
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data, index=['S01', 'S02', 'S03'], columns=['Name', 'Age'])
print(df)
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print(df)

# 通过字典创建DataFrame
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print(df)
print(df.axes)
print(df.index)
print(df.columns)
print(df.values)
print(df.head(2))
print(df.tail(2))

# 列访问
print('-' * 50)
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 
     'three' : pd.Series([1, 3, 4], index=['a', 'c', 'd'])}
df = pd.DataFrame(d)
print(df['one']) # 访问one这一列
print(df[['one', 'three']])
print(df[df.columns[:2]])

# 列添加
print('-' * 50)
# df['four'] = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# df['four'] = pd.Series([1, 2, 3, 4])
# df['four'] = [1, 2, 3, 4]
# df['four'] = [1, 2, 3]
df['four'] = pd.Series([1, 2, 3], index=['b', 'c', 'd'])
print(df)

# 列删除
# print('-' * 50)
# del(df['four'])
# df.pop('one')
# df2 = df.drop(['one', 'three'], axis=1)
# print(df2)


# 行访问
print('-' * 50)
print(df.loc['a'])
print(df.loc[['a', 'b']])
print(df.loc['a':'c'])  # 标签索引切片，结果中包含a b c

print(df.iloc[[0, 2]])
print(df.iloc[0:2])   # 数字索引切片，结果中包含a b

# 行添加
print('-' * 50)
print(df)
newline = pd.Series([2.2, 3.1, 4.5, 3.2], index=['one', 'two', 'three', 'four'], name='e')
df = df.append(newline)
print(df)
df = df.append(df)
print(df)
# 索引有重复的情况，希望重建索引
df.index = np.arange(10)
print(df)

# 行的删除
df = df.drop(np.arange(4, 10))
print(df)

# dataFrame元素的访问
print(df.loc[2]['four'])
print(df.loc[2, 'four'])
print(df['four'][2])
print(df.loc[2:2].loc[2, 'four'])


# 复合索引  
# random.normal() 返回一组服从正态分布随机数，shape：(6,3)，  期望85， 标准差为3
data = np.floor(np.random.normal(85, 3, (6,3)))
df = pd.DataFrame(data)
print('-' * 50)
print(df)

# 把行级索引改为复合索引
mindex = [('classA', 'F'), ('classA', 'M'), 
          ('classB', 'F'), ('classB', 'M'), 
          ('classC', 'F'), ('classC', 'M')]
df.index = pd.MultiIndex.from_tuples(mindex)
# 把列级索引改为复合索引
mindex = [('Age', '20+'), ('Age', '25+'), ('Age', '30+')]
df.columns = pd.MultiIndex.from_tuples(mindex)
print(df)
# 通过复合索引访问元素
print(df.loc['classA', 'F']['Age'])
print(df['Age', '30+'])