# 练习1: 在list01中找出所有字符串
# 练习2: 在list01中找出所有偶数
list01 = [10, "悟空", 80, 33, 20, "八戒", 25]
for item in (target for target in list01 if type(target) == str):
    print(item)

for item in (target for target in list01 if type(target) == int and target % 2 == 0):
    print(item)
