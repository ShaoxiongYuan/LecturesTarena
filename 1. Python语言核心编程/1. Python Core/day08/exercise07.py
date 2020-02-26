# 练习: 质数:大于１的整数,除了１和自身以外,不能被其他数字整除
#     定义函数,获取指定范围内的所有质数.
#     例如：2 ~ 20
#      [2,3,5,7,11,13,17,19]

# def get_primes(begin, end):
#     list_result = []
#     for number in range(begin, end + 1):  # 7
#         for item in range(2, number):  # 2 3 ...6
#             if number % item == 0:  # 7  % 2
#                 # 不是质数
#                 break
#         else:
#             # 是质数
#             list_result.append(number)
#     return list_result

def is_prime(number):
    for item in range(2, number):
        if number % item == 0:  # 7  % 2
            return False
    return True

def get_primes(begin, end):
    # list_result = []
    # for number in range(begin, end + 1):
    #     if is_prime(number):
    #         list_result.append(number)
    # return list_result
    return [number for number in range(begin, end + 1) if is_prime(number)]

# 测试
print(get_primes(2, 20))
