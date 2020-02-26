"""
    内置生成器
        zip
"""
list01 = ["唐僧", "猪八戒", "悟空"]
list02 = [101, 102, 103]

for item in zip(list01, list02):
    print(item)

list01 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]

# * 拆

# 应用：矩阵转置
# zip([1,2,3,4],[5,6,7,8], [9,10,11,12])
# for line in zip(*list01):
#     print(line)
result = list(zip(*list01))
print(result)
