"""
3. 创建技能类
     数据:技能名称、持续时间(1-60)、攻击力(0-500)、消耗法力(0--1000)

   创建技能列表(3个以上)
   定义函数实现下列功能：
   -- 在技能列表中查找攻击力大于100的所有技能名称,攻击力与消耗的法力
   -- 在技能列表中查找消耗法力最大的技能对象
   -- 在技能列表中删除持续时间大于50的技能对象
   -- 根据消耗的法力对技能列表进行升序排列
   -- 将技能列表中攻击力小于100的技能消耗法力设置为0
   -- 删除技能列表中持续时间相同的技能(只保留一个)
"""


class Skill:
    def __init__(self, name="", duration=0, atk=0, cost_sp=0):
        self.name = name
        self.duration = duration
        self.atk = atk
        self.cost_sp = cost_sp

    @property
    def duration(self):
        return self.__duration

    @duration.setter
    def duration(self, value):
        if 1 <= value <= 60:
            self.__duration = value
        else:
            raise Exception("数据超过范围")

    @property
    def atk(self):
        return self.__atk

    @atk.setter
    def atk(self, value):
        if 0 <= value <= 500:
            self.__atk = value
        else:
            raise Exception("数据超过范围")

    @property
    def cost_sp(self):
        return self.__cost_sp

    @cost_sp.setter
    def cost_sp(self, value):
        if 0 <= value <= 1000:
            self.__cost_sp = value
        else:
            raise Exception("数据超过范围")


list_skills = [
    Skill("乾坤大挪移", 50, 200, 500),
    Skill("降龙十八掌", 60, 300, 800),
    Skill("金钟罩", 60, 0, 200),
    Skill("猴子偷桃", 50, 150, 0),
]


# 1. -- 在技能列表中查找攻击力大于100的所有技能名称,攻击力与消耗的法力
def find01():
    list_result = []
    for item in list_skills:
        if item.atk > 100:
            list_result.append((item.name, item.atk, item.cost_sp))
    return list_result


# 测试
re = find01()
for item in re:
    print(item)


# 2. 在技能列表中查找消耗法力最大的技能对象
def get_max():
    max_value = list_skills[0]
    for i in range(1, len(list_skills)):
        if max_value.cost_sp < list_skills[i].cost_sp:
            max_value = list_skills[i]
    return max_value


re = get_max()
print(re.__dict__)


# 3.    -- 在技能列表中删除持续时间大于50的技能对象
def delete_all():
    count = 0
    for i in range(len(list_skills) - 1, -1, -1):
        if list_skills[i].duration > 50:
            del list_skills[i]
            count += 1
    return count


# print(delete_all())

# 4.    -- 根据消耗的法力对技能列表进行升序排列
def sort():
    for r in range(len(list_skills) - 1):
        for c in range(r + 1, len(list_skills)):
            if list_skills[r].cost_sp > list_skills[c].cost_sp:
                list_skills[r], list_skills[c] = list_skills[c], list_skills[r]


sort()
for item in list_skills:
    print(item.__dict__)

# 5.    -- 将技能列表中攻击力小于100的技能消耗法力设置为0
def update():
    for item in list_skills:
        if item.atk < 100:
            item.cost_sp = 0

update()
for item in list_skills:
    print(item.__dict__)


# 6 .-- 删除技能列表中持续时间相同的技能(只保留一个)
def delete_all02():
    count = 0
    for r in range(len(list_skills)-1,-1,-1):
        for c in range(r):
            if list_skills[c].duration == list_skills[r].duration:
                del list_skills[r]
                count += 1
                break
    return count


print(delete_all02())



