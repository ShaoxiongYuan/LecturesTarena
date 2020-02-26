"""
    for for 练习
"""
# 二维列表
list01 = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
]
# -- 打印第三行第二列数据      [行索引][列索引]
print(list01[2][1])

# -- 从右向左，打印第二行数据(一行一个)
for i in range(len(list01[1])-1,-1,-1):
    print(list01[1][i])

# -- 从上向下，打印第三列数据(一行一个)
# [0][2] [1][2] [2][2]
for i in range(len(list01)):
    print(list01[i][2])

# -- 将二维列表以表格状打印出来
# 1 2 3 4
# for r in range(len(list01)):
#     for c in range(len(list01[r])):
#         print(list01[r][c],end = " ")
#     print()

for line in list01:
    for item in line:
        print(item,end = " ")
    print()