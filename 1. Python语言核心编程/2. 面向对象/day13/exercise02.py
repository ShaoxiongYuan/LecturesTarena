"""
    1. 创建技能对象,直接print到终端中。
       格式：xx技能的持续时间是xxx,攻击力是xxx,消耗法力是xxx.

    2. 克隆技能对象,修改其中一个对象变量,查看另外一个对象变量.
    15:20
"""

class Skill:
    def __init__(self, name="", duration=0, atk=0, cost_sp=0):
        self.name = name
        self.duration = duration
        self.atk = atk
        self.cost_sp = cost_sp

    def __str__(self):
        return "%s技能的持续时间是%d,攻击力是%d,消耗法力是%d."%(self.name,self.duration,self.atk,self.cost_sp)

    def __repr__(self):
        return 'Skill("%s",%d,%d,%d)'%(self.name,self.duration,self.atk,self.cost_sp)

s01 = Skill("降龙十八掌",50,500,600)
print(s01)

s02 = eval(s01.__repr__())
s01.name = "降龙"
print(s02)
print(s01)









