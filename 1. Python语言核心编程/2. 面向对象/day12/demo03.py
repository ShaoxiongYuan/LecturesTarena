"""
    继承设计思想
"""


# 需求：老张开车去东北
# 变化点：飞机、火车、轮船...
# 违反面向对象设计原则
#   开闭原则：允许增加新功能,但是不允许改变之前的代码.
#            对扩展开放，对修改关闭
#   依赖倒置：使用

class Person:
    def go_to(self, vehicle, postion):
        print("去", postion)
        if type(vehicle) == Car:
            vehicle.run()
        elif type(vehicle) == Airplane:
            vehicle.fly()


class Car:
    def run(self):
        print("嘟嘟嘟...")


class Airplane:
    def fly(self):
        print("嗖嗖嗖...")


p01 = Person()
c01 = Car()
a01 = Airplane()
p01.go_to(c01, "东北")
p01.go_to(a01, "东北")
