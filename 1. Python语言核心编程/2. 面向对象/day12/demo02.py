"""
    继承语法 -- 数据 
"""
class Person:
    def __init__(self, name=""):
        self.name = name

class Student(Person):
    # 子类构造函数(父类构造函数参数,子类构造函数参数)
    def __init__(self,name = "", score=0):
        # 调用父类构造函数
        super().__init__(name)
        self.score = score

# 创建子类对象,执行子类构造函数.
s01 = Student("悟空",100)
print(s01.name)







