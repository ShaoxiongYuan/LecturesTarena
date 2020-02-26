"""
    根据内存图,写出变量交换代码.
"""

variable01 = "八戒"
variable02 = "沙僧"

# 变量交换原理：
temp = variable01
variable01 = variable02
variable02 = temp

# python代码交换：
variable01, variable02 = variable02, variable01

print(variable01)
print(variable02)
