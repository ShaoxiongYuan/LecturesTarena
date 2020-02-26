
def square_matrix_transpose(square_matrix):
    for c in range(len(square_matrix) - 1):
        for r in range(c + 1, len(square_matrix)):
            # 3. 修改可变对象
            square_matrix[r][c], square_matrix[c][r] = square_matrix[c][r], square_matrix[r][c]
    # 3. 无需通过 return 返回结果
    # return square_matrix

# 测试
list01 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [9, 10, 11, 12],
]
#　1. 传入可变对象
# print(square_matrix_transpose(list01))
square_matrix_transpose(list01)

print(list01)