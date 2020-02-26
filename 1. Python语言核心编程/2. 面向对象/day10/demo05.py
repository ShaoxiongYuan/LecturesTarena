"""
    封装
        目标：保障数据有效性
        property 核心逻辑：拦截
        1. 创建实例变量
        2. 提供读取与写入方法（数据验证）
        3. 创建类变量(与实例变量名称相同),存储property对象
"""


class Wife:

    def __init__(self, name="", age=0):
        self.name = name
        # self.set_age(age)
        self.age = age

    def set_age(self, value):
        if 22 <= value <= 32:
            self.__age = value
        else:
            raise Exception("我不要")

    def get_age(self):
        return self.__age

    # property(读取方法,写入方法)
    age = property(get_age, set_age)

w01 = Wife("双儿", 25)
print(w01.name)
# print(w01.get_age())
print(w01.age)

