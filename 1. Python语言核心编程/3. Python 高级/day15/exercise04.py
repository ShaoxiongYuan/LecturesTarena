"""
    创建图形管理器
    记录多个图形.
    迭代图形管理器,打印多个图形.
    要求：以最快的速度完成
"""

class GraphicIterator:
    def __init__(self, data):
        self.__data = data
        self.__index = -1

    def __next__(self):
        self.__index += 1
        if self.__index > len(self.__data) - 1:
            raise StopIteration()
        return self.__data[self.__index]

class GraphicManager:
    def __init__(self):
        self.__all_graphics = []

    def add_graphic(self, graphic):
        self.__all_graphics.append(graphic)

    def __iter__(self):
        return GraphicIterator(self.__all_graphics)

manager = GraphicManager()
manager.add_graphic("圆形")
manager.add_graphic("三角")
manager.add_graphic("扇形")

for item in manager:
    print(item)
