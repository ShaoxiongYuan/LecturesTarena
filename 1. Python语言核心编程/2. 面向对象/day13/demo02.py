"""
   里氏替换
"""

class Granade:
    def __init__(self, atk=0):
        self.atk = atk

    def explode(self, target):
        print("手雷爆炸啦")
        target.damage(self.atk)

class Target:
    def __init__(self, hp=0):
        self.hp = hp

    def damage(self, value):
        self.hp -= value

# ------------------------------------------
class Player(Target):
    def damage(self, value):
        super().damage(value)
        print("玩家受伤,屏幕碎屏")

class Enemy(Target):
    def damage(self, value):
        super().damage(value)
        print("敌人受伤,掉下装备")
      
g01 = Granade()
g01.explode(Player())
g01.explode(Enemy())