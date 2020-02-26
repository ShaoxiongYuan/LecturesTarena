"""
    字典 dict
    15:22
"""
# 1. 创建
dict01 = {}
dict01 = {101: "悟空", 102: "八戒", 103: "唐僧"}

# 列表(格式) --> 字典
# 格式：[(k,v),(k,v)]
dict02 = dict(["悟空", (102, "猪八戒"), [103, "唐僧"]])
# 字典 --> 列表(只保留key)
list01 = list(dict01)

# 2. 添加
# --通过键-值
dict01[104] = "沙僧"

# 3. 修改（如果key存在,则为修改,不存在则为添加）
dict01[101] = "孙悟空"

# 4. 删除
# 如果key不存在则报错
if 108 in dict01:
    del dict01[108]

# 5. 查询
# key/循环
print(dict01[101])

# -- 获取所有key
for key in dict01:
    print(key)

# --获取所有value
for value in dict01.values():
    print(value)

# --获取所有key - value
# for item in dict01.items():
#     print(item)# (k,v)

for key,value in dict01.items():
    print(key)
    print(value)