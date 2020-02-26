"""
    改造day07/exercise04代码
    定义 矩阵转置 函数

    改造day08/day07_exercise/exercise04代码
    定义 方阵转置 函数
"""

def matrix_transpose(list_matrix):
    """
        矩阵转置
    :param list_matrix:
    :return: 新矩阵
    """
    list_result = []
    for c in range(len(list_matrix[0])):
        line = []
        for r in range(len(list_matrix)):
            line.append(list_matrix[r][c])
        list_result.append(line)
    return list_result

# 测试
list01 = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
]
print(matrix_transpose(list01))