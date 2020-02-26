"""
    改造day07/exercise07代码
    定义 删除列表中相同元素 函数

    体会：传入可变对象、修改可变对象、无需返回值
"""


def delete_duplicates(list_target):
    """

    :param list_target:
    :return:
    """
    count = 0
    for r in range(len(list_target) - 1, -1, -1):
        for c in range(r):
            if list_target[r] == list_target[c]:
                del list_target[r]
                count += 1
                break
    return count

# 测试
list01 = [34, 8, 56, 9, 8, 9]
print(delete_duplicates(list01))
print(list01)