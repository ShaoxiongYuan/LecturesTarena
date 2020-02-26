"""
    封装
        目标：保障数据有效性
        原理
        1. 私有化数据（不隐藏在类外就可以随意操作,无法进行限制）
        2. 提供读取与写入方法（数据验证）
"""

class Wife:
    def __init__(self, name="", age=0):
        self.name = name
        # self.__age = age
        self.set_age(age)

    def set_age(self, value):
        if 22 <= value <= 32:
            self.__age = value
        else:
            raise Exception("我不要")

    def get_age(self):
        return self.__age


w01 = Wife("双儿", 25)
print(w01.name)
# print(w01.__age) # 不能访问私有变量
print(w01.get_age()) # 通过方法读取数据
