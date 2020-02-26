"""
    创建图形管理器
        记录所有图形
        提供计算所有图形总面积的功能
    圆形：pi * r ** 2
    矩形:l * w
    ...
    要求：增加新图形,管理器不变.
    写出体现：
        三大特征
            封装：根据需求分为图形管理器、圆形、矩形.
            继承：创建图形类,隔离图形管理器与具体图形的变化
            多态：图形管理器计算面积时，调用图形,执行圆形、矩形计算方法。
                  通过重写做计算面积方法
        设计原则
            开闭原则：增加新图形,图形管理器不变.
            依赖倒置：调用图形,不调用圆形、矩形.

"""
class GraphicManager:
    def __init__(self):
        self.__list_graphics = []

    def add_graphics(self, graphic):
        self.__list_graphics.append(graphic)

    def get_total_area(self):
        total_area = 0
        for item in self.__list_graphics:
            total_area += item.calculate_area()
        return total_area

class Graphic:
    def calculate_area(self):
        pass

# --------------------------
class Circle(Graphic):
    def __init__(self,r):
        self.r = r

    def calculate_area(self):
        return 3.14 * self.r ** 2

class Rectanle(Graphic):
    def __init__(self, l = 0,w = 0):
        self.l = l
        self.w = w

    def calculate_area(self):
        return self.l * self.w

manager = GraphicManager()
manager.add_graphics(Circle(5))
manager.add_graphics(Rectanle(2,3))
print(manager.get_total_area())











