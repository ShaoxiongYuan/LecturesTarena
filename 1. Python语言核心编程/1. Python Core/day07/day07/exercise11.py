"""
    改造 day04/exercise01 代码
    定义函数,计算整数每位相加和
"""


def each_unit_sum(number):
    """

    :param number:
    :return:
    """
    sum_value = 0
    for item in str(number):
        sum_value += int(item)
    return sum_value


# 测试
print(each_unit_sum(1234))
