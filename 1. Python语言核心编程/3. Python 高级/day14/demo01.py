"""
    模块

    练习2:
        将信息管理系统模块student_manager_system.py拆分为四个模块
            model.py   -->  XXXModel
            bll.py  --> 业务逻辑层　business logic layer
            　　　　　　　　XXXController
            usl.py -->　用户显示层　user show layer
                        XXXView
            main.py -->  创建View的代码
"""
# 导入方式１
# 语法：import　模块名称
# 使用：模块名称.成员
# 更适合面向过程的技术（函数、全局变量）
# 本质：通过变量访问其他模块成员
import module01

module01.func01()

# import module01 as m
#
# m.func01()

# 方式2
# 语法：from 模块名称 import 成员
# 用法：直接使用成员
# 更适合面向对象的技术（类）
# 本质：将其他模块成员导入到当前模块作用域中
from module01 import MyClass

my_class = MyClass()
my_class.func02()

# 方式3
# 语法：from 模块名称 import *
# 用法：直接使用模块内所有成员
# 需要访问模块中多个成员
# (小心：名称冲突)
# 本质：将其他模块成员导入到当前模块作用域中
from module01 import *

MyClass02().func03()

func01()
