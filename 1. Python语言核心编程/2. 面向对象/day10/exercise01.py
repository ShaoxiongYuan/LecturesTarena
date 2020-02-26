"""
    创建类：敌人类
        实例变量：姓名,血量,防御力,攻击力
        方法：受伤 - 血量减少
             打印自身信息 - 打印所有实例变量
    创建敌人列表(3)
        -- 定义函数,在敌人列表中找出所有活的敌人
        -- 定义函数,在敌人列表中找出攻击力大于50的敌人名称与攻击力
        -- 定义函数,在敌人列表中找出防御力最小的敌人
        -- 定义函数,在敌人列表中删除血量大于50的所有敌人
        -- 定义函数,根据攻击力进行降序排列
    体会：
        对象.?
"""


class Enemy:
    def __init__(self, name="", hp=0, defense=0, atk=0):
        self.name = name
        self.hp = hp
        self.defense = defense
        self.atk = atk

    def damage(self):
        self.hp -= 10
        print(self.name, "被攻击,血量减少10", )

    def print_self(self):
        print("我是%s血量%d防御力%d攻击力%d" % (self.name, self.hp, self.defense, self.atk))


list_enemys = [
    Enemy("成昆", 0, 10, 90),
    Enemy("玄冥一老", 90, 8, 80),
    Enemy("玄冥二老", 95, 9, 95),
]


# 练习1
def find01():
    list_result = []
    for item in list_enemys:
        if item.hp > 0:
            list_result.append(item)
    return list_result


# 测试
# for item in find01():
#     item.print_self()

for item in find01():
    item.damage()


# 练习2
def find02():
    list_result = []
    for item in list_enemys:
        if item.atk > 50:
            list_result.append((item.name, item.atk))
    return list_result


# 测试    11:30
for item in find02():
    print(item)


# 练习3
def get_min_by_defense():
    min_value = list_enemys[0]
    for i in range(1, len(list_enemys)):
        if min_value.defense > list_enemys[i].defense:
            min_value = list_enemys[i]
    return min_value


# 测试
result = get_min_by_defense()
result.damage()


# 练习3  -- 定义函数,在敌人列表中删除血量大于50的所有敌人
def delete_all_by_hp():
    count = 0
    for i in range(len(list_enemys) - 1, -1, -1):
        if list_enemys[i].hp > 50:
            del list_enemys[i]
            count += 1
    return count


# print(delete_all_by_hp())

# 定义函数,根据攻击力进行降序排列
def sort():
    for r in range(len(list_enemys) - 1):
        for c in range(r + 1, len(list_enemys)):
            if list_enemys[r].atk < list_enemys[c].atk:
                list_enemys[r], list_enemys[c] = list_enemys[c], list_enemys[r]

sort()
for item in list_enemys:
    item.print_self()




