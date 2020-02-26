"""
    继承语法 -- 方法
       财产：钱不用孩子挣,但是可以花.
       皇位：江山不用孩子打,但是可以坐.
       代码：代码不用子类写,但是可以用.
"""


# 设计角度：现有子，再有父.
# 编码角度：现有父，再有子.

class Person:
    def say(self):
        print("说话")


class Student(Person):
    def study(self):
        print("学习")


class Teacher(Person):
    def teach(self):
        print("教学")


# 创建父类对象
p01 = Person()
# 只能访问父类成员
p01.say()

# 创建子类对象
s01 = Student()
# 既能访问父类成员，也能访问子类成员
s01.say()
s01.study()

t01 = Teacher()
# 不能访问兄弟类成员
# t01.study()

# 内置函数
# ?对象 是一种?类型
# 人对象是一种人类型
print(isinstance(p01, Person))  # True
# 学生对象是一种人类型
print(isinstance(s01, Person))  # True
# 人对象是一种学生类型
print(isinstance(p01, Student))  # False
# 学生对象是一种老师类型
print(isinstance(s01, Teacher))  # False

# ?类型 是一种?类型
# 人类型是一种人类型
print(issubclass(Person, Person))  # True
# 学生类型是一种人类型
print(issubclass(Student, Person))  # True
# 人类型是一种学生类型
print(issubclass(Person, Student))  # False
# 学生类型是一种老师类型
print(issubclass(Student, Teacher))  # False

# ?类型是?类型
# 人(对象的)类型是人类型
print(type(p01) == Person)  # True
# 学生类型是人类型
print(type(s01) == Person)  # False
