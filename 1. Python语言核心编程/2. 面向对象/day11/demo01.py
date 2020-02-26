"""
    python 属性全部语法
        核心逻辑：拦截
        目标：保护数据
    10:30
"""

# 1.  读 + 写属性 快捷键：props + 回车
# class Wife:
#     def __init__(self, age=0):
#         self.age = age
#
#     @property
#     def age(self):
#         return self.__age
#
#     @age.setter
#     def age(self, value):
#         self.__age = value
#
# w01 = Wife(25)
# print(w01.age)

# 2. 读 属性   快捷键：prop + 回车
# class Wife:
#     def __init__(self):
#         # 数据从何而来,无所谓(可以从参数来、某个方法来、写死).
#         self.__age = 25
#
#     @property
#     def age(self):
#         return self.__age
#
# w01 = Wife()
# print(w01.age)
#
# w01.age = 222



# 3. 写 属性   快捷键：prop + 回车
# class Wife:
#     def __init__(self, age=0):
#         self.age = age
#
#     age = property()

#     @age.setter# age.setter(..)
#     def age(self, value):
#         self.__age = value
class Wife:
    def __init__(self, age=0):
        self.age = age

    def age(self, value):
        self.__age = value

    age = property(None,age)

# w01 = Wife(25)
# print(w01.__dict__)
# print(w01.age)









