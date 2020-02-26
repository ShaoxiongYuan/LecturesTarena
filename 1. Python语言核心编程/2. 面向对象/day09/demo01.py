"""
    类和对象
        现实事物  -抽象化->  类  -实例化-> 对象
    15：20
"""

class Wife:
    """
        抽象的老婆
    """

    # 数据：姓名、颜值、钱....
    def __init__(self, name, face_score, money):
        self.name = name
        self.face_score = face_score
        self.money = money

    # 方法(函数)：玩...
    def play(self):
        print(self.name, "在玩耍")

jg = Wife("金刚",6,5000)
jg.play()

