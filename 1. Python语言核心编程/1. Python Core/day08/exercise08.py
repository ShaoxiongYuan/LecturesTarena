# 练习:定义函数,根据时,分,秒，计算总秒数.
# 测试：
#   时,分,秒 --> 总秒数
#   时,分 --> 总秒数
#   分 --> 总秒数
#   时,秒--> 总秒数
def get_total_second(hour=0, minute=0, second=0):
    return hour * 3600 + minute * 60 + second


# 1时,2分,3秒 --> 总秒数
print(get_total_second(1, 2, 3))
# 1时,2分 --> 总秒数
print(get_total_second(1, 2))
# 2分 --> 总秒数
print(get_total_second(minute=2))
# 1时,3秒--> 总秒数
print(get_total_second(1,second=3))
