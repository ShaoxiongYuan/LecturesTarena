"""
    运算符
        算法运算符
        增强运算符
"""
# 1. ＋　－　*  / //  %   幂运算**
number01 = 5
number02 = 2
result = number01 + number02  # 7
print(result)

print(number01 ** 2)  # 5 的2 次方: 5 * 5

number01 = "5"
number02 = "2"
result = number01 + number02  # 不是数学运算,而是文字拼接52
print(result)

# 需求1：整数的个位数是多少?  --> 余数
number = 56
print(number % 10)  # 6

# 需求2：70秒是几分钟零几秒? --> 地板除（整数除）
second = 70
print(second // 60)  # 分钟
print(second % 60)  # 秒

# 2. 增强运算符 +=　-= 　*=   /=   //=  %=  **=
# data01 = 6
# 运算后的结果，不影响变量自身
# print(data01 + 1)# 7
# print(data01)# ? 6

data01 = 6
# 运算后的结果，又赋值给了变量自身.
data01 += 1 # data01 = data01 + 1
print(data01)# 7