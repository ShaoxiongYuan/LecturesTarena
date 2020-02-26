"""
    list01 = [54,5,65,67,78,8]
    删除列表中所有奇数
    三板斧：内存图  调试
    15:25
"""

list01 = [54, 5, 65, 67, 78, 8]
# for item in list01:
#     if item % 2:
#         list01.remove(item)

# 结论：在列表中删除多个元素,倒序删除。
for i in range(len(list01)-1,-1,-1):
    if list01[i] % 2:
        del list01[i]

print(list01)