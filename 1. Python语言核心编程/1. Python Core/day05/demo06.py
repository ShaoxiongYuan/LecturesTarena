"""
    列表推导式
        根据一个可迭代对象，构建另外一个列表。
    练习:exercise09
"""
# 需求1：将list01中每个元素增加10之后,存入list02
list01 = [34, 45, 54, 65, 67, 87]
# list02 = []
# for item in list01:
#     list02.append(item + 10)
list02 = [item + 10 for item in list01]

print(list02)

# 需求2：将list01中所有偶数增加10之后,存入list02
# list02 = []
# for item in list01:
#     if item % 2 == 0:
#         list02.append(item + 10)
list02 = [item + 10 for item in list01 if item % 2 == 0]
