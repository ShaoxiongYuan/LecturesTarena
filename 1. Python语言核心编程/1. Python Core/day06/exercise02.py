"""
    获取年、月、日.
    计算这是这一年的第几天.
    算法：前几个月的总天数 + 当月天数
    2019,5,18
    31 28 31 30 + 18
"""
year = int(input("请输入年："))
month = int(input("请输入月："))
day = int(input("请输入天："))

day_of_second = 29 if year % 4 == 0 and year % 100 != 0 or year % 400 == 0 else 28
days_of_month = (31, day_of_second, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# 累加前几个月的总天数
# total_days = 0
# for i in range( month -1 ):# 0 1 2 3
#     total_days += days_of_month[i]
total_days = sum(days_of_month[: month - 1])

#  累加当月天数
total_days += day

print(total_days)


