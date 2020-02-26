"""
    复习
        封装
            数据：多  -->  一
            行为：隐藏 --> 对外提供必要
            思想：
                分而治之：将需求分为多个类
                变则疏之：单独定义行为变化
                高内聚：类中各个方法都在完成一件事情
                       有且只有一个改变的原因
                低耦合：互不影响
"""


# 需求：存储新学生
#  界面逻辑         核心逻辑
#  input           append


# 写法1
# class A:
#     def func01(self):
#         b = B()
#         b.func02()
#
# class B:
#     def func02(self):
#         pass

# 写法2
# class A:
#     def __init__(self):
#         self.b = B()
#
#     def func01(self):
#         self.b.func02()
#
# class B:
#     def func02(self):
#         pass

# 写法3
class A:
    def func01(self, b):
        b.func02()

class B:
    def func02(self):
        pass

a = A()
b = B()
a.func01(b)
