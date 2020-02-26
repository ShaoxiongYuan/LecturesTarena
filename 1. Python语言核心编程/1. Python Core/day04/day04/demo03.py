"""
    for -- 循环计数
    练习:exercise02
"""

# 整数生成器：产生一个范围内的整数
# 开始值  结束值(不包含)  变化量
for item in range(0, 6, 1):
    print(item)  # 0 1 2 3 4 5

# 结束值(开始值默认为0，变化量默认为1)
for item in range(6):
    print(item)

# 开始值,结束值
for item in range(3, 6):
    print(item)
