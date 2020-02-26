"""
    改造day07/exercise05代码
    定义 排序 函数
    体会：传入可变对象、修改可变对象、无需返回值 
"""

def sort(list_target):
    """

    :param list_target:
    :return:
    """
    for r in range(len(list_target) - 1):  # 0       1
        for c in range(r + 1, len(list_target)):  # 123..   234..
            if list_target[r] > list_target[c]:
                list_target[r], list_target[c] = list_target[c], list_target[r]

list01 = [4, 54, 5, 6, 7, 8, 3]
sort(list01)
print(list01)
