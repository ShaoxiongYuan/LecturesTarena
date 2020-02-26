"""
    list01 = [34,8,56,9,8,9]
    判断列表中是否具有相同元素True  False
    思想：
    将第一位元素,与后面元素进行比较,发现相同则打印结论
       二
       三
    如果所有元素比较后，也没有发现相同，则打印结论。
"""
list01 = [34, 8, 56, 9, 8, 9]
result = False  # 假设结论是没有重复
# 取数据
for r in range(len(list01) - 1):
    # 作比较
    for c in range(r + 1, len(list01)):
        if list01[r] == list01[c]:
            result = True
            break  # 退出一层循环
    if result:
        break

print(result)
