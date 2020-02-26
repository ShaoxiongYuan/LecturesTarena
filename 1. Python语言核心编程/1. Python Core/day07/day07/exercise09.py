# 练习1： 定义函数,在终端中打印列表一个元素一行）
#  list01 = [43,4,5,78]
#  list02 = [76,6,579]

def print_list(list_target):
    """
        打印列表,将列表每个元素打印在终端中（一行一个）
    :param list_target:list类型,需要打印的列表
    """
    for item in list_target:
        print(item)


list01 = [43, 4, 5, 78]
list02 = [76, 6, 579]
print_list(list01)
print_list(list02)
