# 练习1：使用列表推导式生成1--50之间能被3或者5整除的数字
list01 = [item for item in range(1, 51) if item % 3 == 0 or item % 5 == 0]
print(list01)

# 练习2：使用列表推导式生成5--60之间数字的平方
list02 = [item ** 2 for item in range(5, 61)]
print(list02)

# 练习3：将1970年到2050年之间的闰年存入列表
list03 = []
for year in range(1970, 2051):
    if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
        list03.append(year)
# list03 = [year for year in range(1970,2051) if year % 4 ==0 and year % 100 !=0 or year % 400 == 0 ]
print(list03)
