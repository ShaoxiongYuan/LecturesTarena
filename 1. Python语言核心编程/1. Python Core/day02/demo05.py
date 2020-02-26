"""
    类型转换
        语法逻辑：
            结果 = 类型名(待转类型)
"""

# input函数的结果是str类型

# str ---> int / float
# 适用性：需要对象字符串进行数学运算
# 注意：待转数据必须"像"转换的类型.
number = float("200.5")
print(number)

# int / float　---> str
# 适用性：需要按照某种格式显示计算后的数值
print(str(10.5))
