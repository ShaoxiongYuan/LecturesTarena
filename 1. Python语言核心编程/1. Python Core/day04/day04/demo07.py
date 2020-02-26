"""
    通用操作
    练习:exercise07
    
"""
# 1. 算数运算符
# -- +容器元素拼接
name01 = "悟空"
name01 += "八戒"
# -- *容器元素重复多次
name02 = "唐僧"
name02 *= 2
print(name02)
# -- == !=
# 依次比较两个容器中元素,一但不同则返回比较结果。
print("悟空" > "八戒")

# 2. 成员运算 in   not in
print("我爱" in "我爱你")  # true
print("爱我" in "我爱你")  # flase

message = "我是齐天大圣."
# 3. 索引:定位单个元素
print(message[1])
print(message[-2])
# 索引越界
# print(message[7])# 报错 IndexError
# print(message[-8])# 报错 IndexError
# 4. 切片:定位多个元素
# range(  ,  ,  )

# [开始索引:结束索引:间隔]    不包含结束索引
print(message[0:4:1])  # 我是齐天
# [:结束索引:]
print(message[:4:])  # 我是齐天
# [:]
print(message[:])  # 我是齐天大圣.
# message[::-1]
print(message[::-1])  # .圣大天齐是我
# [开始索引::]
print(message[3::])  # 天大圣.
# [::间隔]
print(message[::2])  # 我齐大.
# 切片不会越界
print(message[:100])  # 我是齐天大圣.
print(message[3:3])  # 获取空
print(message[1:-2])  # 是齐天大
print(message[1:5:-1])  # 获取空
