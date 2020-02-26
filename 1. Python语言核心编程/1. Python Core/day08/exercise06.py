"""
    改造day03/exercise05代码
    定义　根据月份计算天数的　函数
"""


def is_leap_year(year):
    """
        判断是否为闰年
    :param year:
    :return:
    """
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0


def get_day_by_month(year, month):
    """
        获取天
    :param year:
    :param month:
    :return:
    """
    if month < 1 or month > 12:
        return 0
    if month == 2:
        return 29 if is_leap_year(year) else 28
    if month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    return 31
