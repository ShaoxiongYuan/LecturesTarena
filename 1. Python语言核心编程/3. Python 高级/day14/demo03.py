"""
    模块与包的相关概念
"""

# 1. __all__ 定义可导出成员,仅对from xx import *起作用

# 不能导入隐藏成员
from module03 import *
# _func02()
# func03()

from module03 import _func02
from module03 import func03

_func02()
func03()

# from 包 import *
from package01 import *
m01.func01()

from package01 import m03
m03.func03()

# 2. __doc__ 文档字符串属性
print(__doc__)
print(m03.__doc__)
print(m03.func03.__doc__)

# 3.模块绝对路径
print(__file__)

# 4.模块名称
# 价值：区分当前模块是否是主模块
print(__name__)#__main__









