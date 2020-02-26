"""
    函数返回值
        应用

    设计思想：小而精
"""
# 需求：定义函数,两个数值相加.

def add(number01,number02):
    """
        两个数值相加
    :param number01: 数值1
    :param number02: 数值2
    :return:数值类型,相加的结果
    """
    result = number01 + number02
    return result

# 测试
n1 = float(input("请输入第一个数字："))
n2 = float(input("请输入第二个数字："))
re = add(n1,n2)
print("结果是："+str(re))


