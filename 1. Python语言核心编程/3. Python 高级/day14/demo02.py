"""
    包
    python 程序结构
    项目根目录
    包
        模块
            类
                函数
                    语句
    导入是否成功的唯一条件：
        导入路径 + sys.path == 真实路径
"""


# 导入方式1
# 语法：import 路径.m01

# import package01.m01

# package01.m01.func01()

import package01.m01 as m1
import package01.package02.m02 as m2

m1.func01()
m2.func02()

# 导入方式2
# 语法：from 路径.模块名 import 成员
from package01.package02.m02 import func02

func02()

# 导入方式3
from package01.package02.m02 import *

func02()











