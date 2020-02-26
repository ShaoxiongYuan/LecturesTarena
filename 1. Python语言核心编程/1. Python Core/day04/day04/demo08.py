"""
    列表list   的基础操作
    练习1:exercise08
"""
# 1. 创建
# -- 直接存储数据
list01 = []
list01 = ["悟空", "八戒"]
# -- 将其他可迭代对象存入列表
list02 = list("我是齐天大圣")

# 2. 增加
# -- 追加(末尾)
list01.append("唐僧")
# -- 插入(在指定)
list01.insert(1, "沙僧")

# 3. 删除
# -- 根据元素
if "八戒2" in list01:
    list01.remove("八戒2")  # 备注：如果没有元素则报错
# -- 根据位置
# del list01[1]
del list01[1:3]

# 4. 修改
print(list02)
# -- 索引
# list02[0] = "你" # 索引
# -- 切片
# 原理：遍历右侧可迭代对象,将每个元素赋值给左边切片定位的元素
list02[2:3] = [1,2,3,4,5]
print(list02)

# 5. 查询
# -- 索引
print(list02[-1])
# -- 切片
# 原理：切片后返回新列表（拷贝）
print(list02[:])
# -- 循环
for item in list02:
    print(item)

# 需求倒序获取所有元素（不建议切片）
# for item in list02[::-1]:
#     print(item)

for index in range(len(list02)-1,-1,-1):
    print(list02[index])




