"""
    装饰器（拦截）
        在不改变函数的定义与调用基础上,为其增加新功能。
"""

""" 装饰器本质思想
def print_func_name(func):
    def wrapper():
        # 执行新功能
        print("被调用的函数是：",func.__name__)
        # 执行旧功能
        func()
    return wrapper

# 旧功能
def say_hello():
    # print_func_name(say_hello)
    print("hello")

def say_goodbye():
    # print_func_name(say_goodbye)
    print("goodbye")

# say_hello = 旧功能 + 新功能
say_hello = print_func_name(say_hello)
# 拦截
say_goodbye = print_func_name(say_goodbye)


say_hello()
say_goodbye()
"""

"""
def print_func_name(func):
    def wrapper():
        # 执行新功能
        print("被调用的函数是：", func.__name__)
        # 执行旧功能
        func()

    return wrapper
 
# 旧功能
@print_func_name  # say_hello = print_func_name(say_hello)
def say_hello():
    print("hello")

@print_func_name  # say_goodbye = print_func_name(say_goodbye)
def say_goodbye():
    print("goodbye")

say_hello()
say_goodbye()
"""

""" 内部函数增加返回值
def print_func_name(func):
    def wrapper():
        # 执行新功能
        print("被调用的函数是：", func.__name__)
        # 执行旧功能
        return func()

    return wrapper

# 旧功能
@print_func_name  # say_hello = print_func_name(say_hello)
def say_hello():
    print("hello")

@print_func_name  # say_goodbye = print_func_name(say_goodbye)
def say_goodbye():
    print("goodbye")
    return 100

print(say_hello())
print(say_goodbye())
"""

def print_func_name(func):
    def wrapper(*args,**kwargs):# 合  ("qtx",)
        # 执行新功能
        print("被调用的函数是：", func.__name__)
        # 执行旧功能
        return func(*args,**kwargs)# 拆

    return wrapper

# 旧功能
@print_func_name  # say_hello = print_func_name(say_hello)
def say_hello(name1,name2):# "qtx"
    print(name1,name2,"hello")

@print_func_name  # say_goodbye = print_func_name(say_goodbye)
def say_goodbye():
    print("goodbye")
    return 100

print(say_hello("qtx",name2 = "lzmly"))
print(say_goodbye())
# 15：25
