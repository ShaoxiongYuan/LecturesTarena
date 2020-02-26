# 练习:定义函数,数值相加的函数.
# 测试：1,2
#      1,2,5,54,65,6,
def sum_numbers(*args):
    # sum_value = 0
    # for item in args:
    #      sum_value += item
    # return sum_value
    return sum(args)

print(sum_numbers())
print(sum_numbers(1, 2, 3, 54, 4, 56, 6, 7))