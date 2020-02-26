"""
    多继承
        有多种变化需要隔离（不是代码复用）
    １６：５０
"""

class A:
    def func01(self):
        print("A -- func01")


class B(A):
    def func01(self):
        print("B -- func01")


class C(A):
    def func01(self):
        print("C -- func01")


class D(C, B):
    def func01(self):
        print("D -- func01")
        # 1. 如何调用某一个同名方法
        B.func01(self)

d = D()
d.func01()
# 2. 同名方法解析顺序
print(D.mro())
