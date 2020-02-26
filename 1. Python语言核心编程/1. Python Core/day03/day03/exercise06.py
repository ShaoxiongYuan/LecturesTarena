# 练习1:
# 获取一个整数,如果是奇数给变量state赋值为"奇数",否则赋值"偶数"

# if int(input("请输入整数：")) % 2 == 1:
#     state = "奇数"
# else:
#     state = "偶数"

# if int(input("请输入整数：")) % 2:
#     state = "奇数"
# else:
#     state = "偶数"

state = "奇数" if int(input("请输入整数：")) % 2 else "偶数"
print(state)

# 练习2:
# 获取一个年份,如果是闰年给变量day赋值为29,否则赋值28
year = int(input("请输入年份："))
# 建议
day = 29 if year % 4 == 0 and year % 100 != 0 or year % 400 == 0 else 28
# day = 29 if not year % 4 and year % 100 or not year % 400 else 28
# 15:20