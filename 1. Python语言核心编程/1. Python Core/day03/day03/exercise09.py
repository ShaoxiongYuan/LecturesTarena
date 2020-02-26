"""
    获取一个开始值,再获取一个结束值.
    打印中间值
    5      10   -->   6 7 8 9
    10     5   -->    9 8 7 6
"""
# 精神：顽强的拼搏力和无所畏惧的勇气.

begin = int(input("请输入开始值："))  # 5
end = int(input("请输入结束值："))  # 10

# 行为相同  数据不同
# while begin < end - 1:
#     begin += 1
#     print(begin)
#
# while begin > end + 1:
#     begin -= 1
#     print(begin)

dir = 1 if begin < end else -1
while begin != end - dir:
    begin += dir
    print(begin)

# 17:00
