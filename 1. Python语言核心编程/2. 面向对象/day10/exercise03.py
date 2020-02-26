"""
    属性练习
        创建敌人类
            数据：姓名,血量（0--500）,攻击力（10,--100）
        创建敌人对象,体会拦截的核心逻辑.
"""


class Enemy:
    def __init__(self, name="", hp=0, atk=0):
        self.name = name
        self.hp = hp
        self.atk = atk

    def set_hp(self, value):
        if 0 <= value <= 500:
            self.__hp = value
        else:
            raise Exception("血量超过范围")

    def get_hp(self):
        return self.__hp

    hp = property(get_hp, set_hp)

    def set_atk(self, data):
        if 10 <= data <= 100:
            self.__atk = data
        else:
            raise Exception("攻击力超过范围")

    def get_atk(self):
        return self.__atk

    atk = property(get_atk, set_atk)


e01 = Enemy("灭霸", 500, 100)
print(e01.name)
print(e01.hp)
print(e01.atk)
e01.a = 9999999
