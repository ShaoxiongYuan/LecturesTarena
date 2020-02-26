"""
    封装设计思想
        分而治之：划分为多个类
        变则疏之：识别变化点（行为）
        -------------
        高内聚：类有且只要一个变化点(原因)
        低耦合：类与类的关系松散
"""
# 类与类行为不同
# 对象与对象数据不同

# 需求：老张开车去东北
# 变化：老张、老李、老王...   --> 数据不同
#      划船、骑车、飞机...   --> 行为不同
#      西北、南北...        --> 数据不同

# 写法1
# 语义：老张去东北创建一辆新车
#　　　老张无论去哪里，都会创建一辆新车
# class Person:
#     def __init__(self, name=""):
#         self.name = name
#
#     def go_to(self,position):
#         print(self.name,"去",position)
#         car = Car()
#         car.run()
#
# class Car:
#     # 实例成员：对象.?
#     def run(self):
#         print("汽车行驶")
#
# lz = Person("老张")
# lz.go_to("东北")# 一辆新车
# lz.go_to("家")# 又一辆新车

# 写法2
# 语义：老张开车自己的车去东北
# class Person:
#     def __init__(self, name=""):
#         self.name = name
#         self.car = Car()
#
#     def go_to(self,position):
#         print(self.name,"去",position)
#         self.car.run()
#
# class Car:
#     def run(self):
#         print("汽车行驶")

# lz = Person("老张")
# lz.go_to("东北")
# lz.go_to("家")

# 写法3
# 语义：老张通过传递的参数,决定如何去东北.
class Person:
    def __init__(self, name=""):
        self.name = name

    def go_to(self,vehicle,position):
        print(self.name,"去",position)
        vehicle.run()

class Car:
    def run(self):
        print("汽车行驶")

lz = Person("老张")
bm = Car()
lz.go_to(bm,"东北")

