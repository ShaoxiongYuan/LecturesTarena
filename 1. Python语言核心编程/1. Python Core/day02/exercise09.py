"""
    获取一个年份
    如果是闰年输入True，否则输出false
    条件：
        1. 年份能被4整除但是不能被100整除
        2. 年份能被400整除
"""
year = int(input("请输入年份："))
result = year % 4 == 0 and year % 100 != 0 or year % 400 == 0
print(result)
