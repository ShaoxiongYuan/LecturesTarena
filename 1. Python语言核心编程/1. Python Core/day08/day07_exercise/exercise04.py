"""
    方阵转置
"""
list01 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [9, 10, 11, 12],
]
# [行索引][列索引]
# list01[1][0]  list01[0][1]
# list01[2][0]  list01[0][2]
# list01[3][0]  list01[0][3]

# for r in range(1,4):
#     # list01[r][0]  list01[0][r]
#     pass
#
# # list01[2][1]  list01[1][2]
# # list01[3][1]  list01[1][3]
# for r in range(2,4):
#     # list01[r][1]  list01[1][r]
#     pass
#
# # list01[3][2]  list01[2][3]
# for r in range(3,4):
#     # list01[r][2]  list01[2][r]
#     pass
# 10:10
for c in range(len(list01) - 1):
    for r in range(c + 1, len(list01)):
        list01[r][c], list01[c][r] = list01[c][r], list01[r][c]

print(list01)
