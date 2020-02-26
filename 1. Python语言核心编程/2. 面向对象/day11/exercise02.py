"""
    需求：玩家(攻击力)攻击敌人(血量),
         敌人受伤(减血)后可能死亡(播放死亡动画).
"""
class Player:
    def __init__(self, atk=0):
        self.atk = atk

    def attack(self,enemy):
        print("玩家打敌人")
        enemy.damage( self.atk )

class Enemy:
    def __init__(self, hp=0):
        self.hp = hp

    def damage(self,atk):
        self.hp -= atk
        print("额～受伤啦")
        if self.hp <= 0:
            self.__death()

    def __death(self):
        print("播放死亡动画")

p01 = Player(50)
e01 = Enemy(100)
p01.attack(e01)
p01.attack(e01)




  



