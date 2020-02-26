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


# 练习1:定义函数，在技能列表中找出cost_sp大于300的所有技能
def find01():
    for item in list_skills:
        if item.cost_sp > 300:
            yield item


# 惰性操作 --> 节省内存
for item in find01():
    print(item.__dict__)
# 立即操作 --> 获取数据灵活(索引/切片)
list_result = list(find01())
print(list_result[-1])


# 练习2:定义函数，在技能列表中找出持续时间等于50的所有技能名称与持续时间
def find02():
    for item in list_skills:
        if item.duration > 50:
            yield (item.name, item.duration)


for item in find02():
    print(item)


# 练习3:定义函数，在技能列表中找出名称是"猴子偷桃"的单个技能对象
def find03():
    for item in list_skills:
        if item.name == "猴子偷桃":
            return item


print(find03().__dict__)
# 11：30