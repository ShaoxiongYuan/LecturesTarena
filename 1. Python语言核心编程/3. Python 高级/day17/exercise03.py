"""
    在不改变函数(func01,func02)的定义与调用基础上,
    为其增加新功能(在终端中打印函数执行时间)。
    要求：调试查看程序执行过程
"""
import time
# 16：50
def print_excute_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("函数执行时间为：", end_time - start_time)
        return result

    return wrapper

@print_excute_time  # func01 = print_excute_time(func01)
def func01():
    for __ in range(10):
        pass

@print_excute_time  # func02 = print_excute_time(func02)
def func02():
    for __ in range(10000000):
        pass

func01()  # 执行内部函数wrapper
func02()
