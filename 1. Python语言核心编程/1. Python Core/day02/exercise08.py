"""
    获取一个4位整数           1234
    输出每位相加和            1+2+3+4
"""
# # 方法1
# number = int(input("请输入整数："))
# # 个位
# unit01 = number % 10
# # 十位 1234 --> 123  --> 3
# unit02 = number // 10 % 10
# # 百位 1234 --> 12  --> 2
# unit03 = number // 100 % 10
# # 千位  1234 --> 1
# unit04 = number // 1000
# print(unit01 + unit02 + unit03 + unit04)

# 方法2
number = int(input("请输入整数："))
# 个位
result = number % 10
# 十位
result += number // 10 % 10
# 百位
result += number // 100 % 10
# 千位  1234 --> 1
result += number // 1000
print(result)
