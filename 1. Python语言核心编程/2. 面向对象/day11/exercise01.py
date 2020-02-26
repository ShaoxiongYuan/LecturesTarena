"""
    需求：小明在银行取钱
        现象：人的钱多了   银行的钱少了
"""
class Person:
    def __init__(self,money):
        self.money = money

class Bank:
    def __init__(self,money):
        self.money = money

    def draw_money(self,person,value):
        self.money -= value# 银行钱少了
        person.money += value# 人钱多了

xm = Person(0)
zsyh = Bank(100000)
zsyh.draw_money(xm,5000)
print("小明的钱：",xm.money)
print("银行的钱：",zsyh.money)
