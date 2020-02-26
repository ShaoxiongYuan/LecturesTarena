"""
    1. 在技能列表中，查找所有技能的名称name,攻击力atk与消耗法力cost_sp
    2. 在技能列表中,查找攻击力大于200的所有技能对象
    3. 在技能列表中,查找持续时间小于等于50的所有技能名称
    4. 在列表中找出最短的元素[(1,1),(2,),(3,3,3),(4,4)]
    5. 根据消耗法力对技能列表进行降序排列.
"""


class Skill:
    def __init__(self, name="", duration=0, atk=0, cost_sp=0):
        self.name = name
        self.duration = duration
        self.atk = atk
        self.cost_sp = cost_sp


list_skills = [
    Skill("乾坤大挪移", 50, 200, 500),
    Skill("降龙十八掌", 60, 300, 800),
    Skill("金钟罩", 60, 0, 200),
    Skill("猴子偷桃", 50, 150, 0),
]

# 1.
for item in map(lambda skill: (skill.name, skill.atk, skill.cost_sp), list_skills):
    print(item)
# 2.
for item in filter(lambda s: s.atk > 200, list_skills):
    print(item.__dict__)
# 3.
# result = filter(lambda s:s.duration <= 50,list_skills)
# for item in map(lambda skill:skill.name , result):
#     print(item)
for item in map(lambda skill: skill.name, filter(lambda s: s.duration <= 50, list_skills)):
    print(item)
# 4.
list01 = [(1, 1), (2,), (3, 3, 3), (4, 4)]
print(min(list01, key=lambda item: len(item)))
# 5.
for item in sorted(list_skills, key=lambda skill: skill.cost_sp, reverse=True):
    print(item.__dict__)
