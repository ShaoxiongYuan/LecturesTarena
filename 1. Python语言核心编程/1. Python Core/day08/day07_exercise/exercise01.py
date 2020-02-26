"""
3. 改造day07/exercise03代码
    定义函数,将二维列表以表格状打印出来
"""

def print_double_list(list_target):
    """
        打印二维列表
    :param list_target:需要打印的数据
    """
    for line in list_target:
        for item in line:
            print(item,end = "\t")
        print()

# 测试
list01 = [
    [35,6,7,8],
    [8,64,3,8],
    [3,654,0,3],
]
print_double_list(list01)