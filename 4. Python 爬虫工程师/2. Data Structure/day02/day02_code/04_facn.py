"""
利用递归求出n的阶乘
"""
def fac(n):
    # 递归出口
    if n == 1:
        return 1

    return n * fac(n-1)

print(fac(1000))





