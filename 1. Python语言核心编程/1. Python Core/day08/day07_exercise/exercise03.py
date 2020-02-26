"""
    5. 定义函数,判断二维列表中是否具有某个元素
   例如：在list01中是否具有8
    list01 = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
    ]
"""


def is_exist_by_double_list(target,element):
    """
        判断二维列表中是否存在某个元素
    :param target:目标列表
    :param element:需要判断的元素
    :return:bool类型，是否存在
    """
    for line in target:
        if element in line:
            return True
    return False

# 测试
list01 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]
print(is_exist_by_double_list(list01,18))
