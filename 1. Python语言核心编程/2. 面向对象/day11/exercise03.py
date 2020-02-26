"""
    张无忌教赵敏九阳神功
    赵敏教张无忌化妆
    张无忌上班挣了8000
    赵敏上班挣了10000元
"""
class Person:
    def __init__(self, name=""):
        self.name = name

    def teach(self,person_other,skill):
        print(self.name,"教",person_other.name,skill)

    def work(self,salary):
        print(self.name,"上班挣了",salary,"元")

zwj = Person("张无忌")
zm = Person("赵敏")

zwj.teach(zm,"九阳神功")
zm.teach(zwj,"化妆")

zwj.work(8000)
zm.work(10000)
