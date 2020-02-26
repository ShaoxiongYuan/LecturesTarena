"""
    手雷爆炸,伤害玩家、敌人生命.
    变化点：还可能伤害房子,鸭子....
    要求：增加新事物,不影响手雷类.
    体会：开闭原则、依赖倒置、继承(设计思想)、多态(设计思想)
                 强调调用谁             强调执行/实现子类不同逻辑
"""

class Granade:
    """
        手雷
    """

    def __init__(self, atk=0):
        self.atk = atk

    def explode(self, target):
        print("手雷爆炸啦")
        target.damage(self.atk)

class Target:
    def damage(self, value):
        pass


# ------------------------------------------
class Player(Target):
    def __init__(self, hp=0):
        self.hp = hp

    def damage(self, value):
        print("玩家受伤,屏幕碎屏")
        self.hp -= value

class Enemy(Target):
    def __init__(self, hp=0):
        self.hp = hp

    def damage(self, value):
        print("敌人受伤,掉下装备")
        self.hp -= value

g01 = Granade()
g01.explode(Player())
g01.explode(Enemy())