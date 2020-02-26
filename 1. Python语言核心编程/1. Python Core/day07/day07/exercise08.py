# 色子 1 -- 6
# 色子 1 -- 6
# 色子 1 -- 6
# 将3个色子所有组合数存储到列表中

# list_result = []
# for x in range(1,7):#                 1           2
#     for y in range(1,7):#       1      2    ..
#         for z in range(1,7):#123456  123456
#             list_result.append((x,y,z))

list_result = [(x, y, z) for x in range(1, 7) for y in range(1, 7) for z in range(1, 7)]
print(list_result)
