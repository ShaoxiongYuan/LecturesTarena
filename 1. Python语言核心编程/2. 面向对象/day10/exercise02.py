"""
    创建对象计数器
    提示：统计__init__被调用的次数
    画出内存图
"""


class Wife:
    count = 0

    @classmethod
    def print_count(cls):
        print("总共娶了%d个老婆" % cls.count)

    def __init__(self, name=""):
        self.name = name
        Wife.count += 1


w01 = Wife("")
w02 = Wife("")
w03 = Wife("")
w04 = Wife("")
w05 = Wife("")
w06 = Wife("")
Wife.print_count()
