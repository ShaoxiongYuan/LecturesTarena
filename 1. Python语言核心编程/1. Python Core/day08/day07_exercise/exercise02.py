"""
    4. 改造day07/exercise06代码
    定义函数,判断列表中是否具有相同元素。
"""


def is_repetition(list_target):
    """

    :param list_target:
    :return:
    """
    for r in range(len(list_target) - 1):
        for c in range(r + 1, len(list_target)):
            if list_target[r] == list_target[c]:
                return True
    return False


list01 = [34, 8, 34, 9, 8, 9]
print(is_repetition(list01))
