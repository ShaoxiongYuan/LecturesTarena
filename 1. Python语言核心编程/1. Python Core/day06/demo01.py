"""
    元组tuple
    练习:exercise01,exercise02
"""

# 1. 创建
tuple01 = ()
tuple01 = (4, 545, 5665, 7, 78)
list01 = [54, 5, 56, 67]
# 列表（预留空间） -->  元组（按需分配）
tuple02 = tuple(list01)
#  元组（按需分配） --> 列表（预留空间）
list02 = list(tuple01)

# 特殊
# -- 如果元组只有一个数据,必须在后面添加逗号
tuple03 = (4,)
# -- 小括号可以省略
tuple04 = 4, 5, 6
# -- 可以直接将容器赋值给多个变量
a, b, c = (8, 9, 10)
print(type(tuple04))

# 2. 查询
# -- 索引
print(tuple01[-1])  # 获取最后一个
print(tuple01[2])  # 获取第三个
# -- 切片
# 原理：创建新容器
print(tuple01[-3:])  # 最后三个
# -- 循环
for item in tuple01:
    print(item)

for i in range(len(tuple01) - 1, -1, -1):
    print(tuple01[i])