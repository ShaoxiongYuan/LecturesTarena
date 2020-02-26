"""
    矩阵转置
        算法：将list01中每列,存储到list02中每行
"""
list01 = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
]

list02 = []
# list01[0][0]
# list01[1][0]
# list01[2][0]

for c in range(len(list01[0])):
    line = []
    for r in range(len(list01)):
        line.append(list01[r][c])
    list02.append(line)

print(list02)



