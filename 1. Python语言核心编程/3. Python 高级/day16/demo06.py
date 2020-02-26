"""
    lambda 表达式
        匿名方法
"""
# 1. 写法1：有参数有返回值
# def func01(a,b):
#     return a > b
# print(func01(10,20))

func01 = lambda a, b: a > b
print(func01(10, 20))

# 2. 写法1：无参数有返回值
# def func02():
#     return 100
# print(func02())

func02 = lambda: 100
print(func02())

# 写法3：有参数无返回值
# def func03(p1):
#     p1[0] = 2
#
list01 = [1]


# func03(list01)
# print(list01)

# 注意：lambda 表达式不支持赋值语句
# func03 = lambda p1:  p1[0] = 2

def func03(p1):
    print(p1)


func03 = lambda p1: print(p1)
func03(list01)

# 写法4：无参数无返回值
# def func04():
#     print("hello world")

func04 = lambda: print("hello world")
func04()

# 注意： lambda 不支持多行语句
# def func05():
#    for item in range(5):
#        print(item)

# func05 = lambda : for item in range(5): print(item)
