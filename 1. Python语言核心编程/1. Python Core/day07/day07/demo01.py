"""
    列表 推导式
        [处理变量 for 变量 in 可迭代对象 if 条件]
    字典 推导式
        {处理变量 : 处理变量  for 变量 in 可迭代对象 if 条件}
    集合 推导式
        {处理变量 for 变量 in 可迭代对象 if 条件}
    适用性：
        根据可迭代对象,构建容器时.
"""
# 需求1：range(10)  --> key:0-9   value: key的平方
dict01 = {item: item ** 2 for item in range(10)}
print(dict01)

# 需求2：range(10)  --> key:0-9   value: key的平方
#        只考虑奇数
dict01 = {item: item ** 2 for item in range(10) if item % 2}
print(dict01)
