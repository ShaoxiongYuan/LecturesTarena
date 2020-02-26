"""
    迭代器 --> yield
    练习:改造图形管理器
        改造MyRange类
        exercise07
    要求：调试程序

"""

class SkillManager:
    def __init__(self):
        self.__all_skills = []

    def add_skill(self, skill):
        self.__all_skills.append(skill)

    # def __iter__(self):
    #     # yield 核心逻辑： 标记
    #     # 生成迭代器代码的大致逻辑：
    #     # 1. 将yield语句以前的代码,定义到next方法体中
    #     # 2. 将yield语句以后的数据,作为到next方法返回值
    #     print("准备")
    #     yield self.__all_skills[0]
    #     print("准备")
    #     yield self.__all_skills[1]
    #     print("准备")
    #     yield self.__all_skills[2]

    def __iter__(self):
        for item in self.__all_skills:
            yield item

manager = SkillManager()
manager.add_skill("降龙十八掌")
manager.add_skill("六脉神剑")
manager.add_skill("猴子偷桃")

for item in manager:
    print(item)

# iterator = manager.__iter__()
# while True:
#     try:
#         item = iterator.__next__()
#         print(item)
#     except StopIteration:
#         break
