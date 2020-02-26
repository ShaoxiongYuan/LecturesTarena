"""

"""


class Wife:

    def __init__(self, name, face_score, money):
        self.name = name
        self.face_score = face_score
        self.money = money

    def print_self(self):
        print("%s--%d--%d" % (self.name, self.face_score, self.money))


w01 = Wife("建宁", 86, 999999)
list_wifes = [
    w01,
    Wife("双儿", 95, 5000),
    Wife("苏荃", 98, 10000),
    Wife("阿珂", 100, 6000),
    Wife("铁锤", 80, 0),
]


# 练习1：定义函数,在老婆列表中查找双儿对象
#       测试：调用双儿对象的print_self方法
def find01():
    for item in list_wifes:
        if item.name == "双儿2":
            return item


# 测试：
result = find01()
# 如果找到了
if result:
    result.print_self()


# 练习2：定义函数,在老婆列表中查找所有老婆的姓名
#       测试：打印列表
def find02():
    # list_result = []
    # for item in list_wifes:
    #     list_result.append(item.name)
    # return list_result
    return [item.name for item in list_wifes]


print(find02())
