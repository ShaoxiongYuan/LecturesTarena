"""
    创建module02模块,定义全局变量,实例方法,静态方法
    以三种方式,在当前模块(exercise02)中分别导入他们.
"""

# import module02
#
# print(module02.g01)
# module02.Wife().func01()
# module02.Wife.func02()

# from module02 import g01,Wife
# print(g01)
# Wife().func01()
# Wife.func02()

from module02 import *
print(g01)
Wife().func01()
Wife.func02()

