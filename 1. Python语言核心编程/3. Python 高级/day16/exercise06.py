"""
    需求1：定义函数,在老婆列表中,查找颜值大于90的所有老婆
         定义函数,在老婆列表中,查找存款小于十万的所有老婆
    需求2：定义函数,在老婆列表中,查找姓名是苏荃的老婆对象
          定义函数,在老婆列表中,查找颜值是100的老婆对象(如有多个返回第一个)
    需求3：定义函数,在老婆列表中,查找所有老婆的姓名
          定义函数,在老婆列表中,查找所有老婆的姓名和颜值
    需求4：定义函数,在老婆列表中,查找颜值最高的老婆
          定义函数,在老婆列表中,查找存款最多的老婆
    需求5：定义函数,根据年龄对老婆列表进行升序排列
          定义函数,根据颜值对老婆列表进行升序排列
    步骤：
        ×1. 根据需求创建函数.
        ×2. 运用函数式编程思想：分、隔、做
        3. 将通用代码定义到IterableHelper中
        4. 在当前模块中测试
"""
from common.iterable_tools import IterableHelper


class Wife:
    def __init__(self, name, face_score, money, age):
        self.name = name
        self.face_score = face_score
        self.money = money
        self.age = age


list_wifes = [
    Wife("建宁", 86, 999999, 25),
    Wife("双儿", 95, 5000, 23),
    Wife("苏荃", 98, 10000, 30),
    Wife("阿珂", 100, 6000, 23),
    Wife("铁锤", 80, 0, 35),
]


# def find01():
#     for item in list_wifes:
#         if item.face_score > 90:
#             yield item
#
#
# def find02():
#     for item in list_wifes:
#         if item.money < 10000:
#             yield item

def condition01(item):
    return item.face_score > 90


def condition02(item):
    return item.money < 10000

# 万能查找
# def find(func):
#     for item in list_wifes:
#         # if item.money < 10000:
#         # if condition02(item):
#         if func(item):
#             yield item

# for item in find(condition02):
#     print(item.__dict__)

for item in IterableHelper.find_all(list_wifes, condition01):
    print(item.__dict__)

for item in IterableHelper.find_all(list_wifes, lambda item: item.face_score > 90):
    print(item.__dict__)

# 需求2：定义函数,在老婆列表中,查找姓名是苏荃的老婆对象
#        定义函数,在老婆列表中,查找颜值是100的老婆对象(如有多个返回第一个)
# def find01():
#     for item in list_wifes:
#         if item.name == "苏荃":
#             return item
#
# def find02():
#     for item in list_wifes:
#         if item.face_score == 100:
#             return item
#
# def condition01(item):
#     return item.name == "苏荃"
#
# def condition02(item):
#     return item.face_score == 100
#
# def find(func):
#     for item in list_wifes:
#         # if item.face_score == 100:
#         if func(item):
#             return item

result = IterableHelper.find(list_wifes,lambda item: item.name == "苏荃" )
print(result.__dict__)

for item in IterableHelper.select(list_wifes,lambda wife:wife.name):
    print(item)

for item in IterableHelper.select(list_wifes,lambda wife:(wife.name,wife.face_score)):
    print(item)

result = IterableHelper.get_max(list_wifes,lambda wife:wife.money)
print(result.__dict__)


IterableHelper.order_by(list_wifes,lambda wife:wife.face_score)

for item in list_wifes:
    print(item.__dict__)







