"""
    迭代器
    练习:exercise04 05
"""
class SkillIterator:
    def __init__(self, data):
        self.__data = data
        self.__index = -1

    def __next__(self):
        self.__index += 1
        if self.__index > len(self.__data) - 1:
            raise StopIteration()
        return self.__data[self.__index]


class SkillManager:
    def __init__(self):
        self.__all_skills = []

    def add_skill(self, skill):
        self.__all_skills.append(skill)

    def __iter__(self):
        return SkillIterator(self.__all_skills)


manager = SkillManager()
manager.add_skill("降龙十八掌")
manager.add_skill("六脉神剑")
manager.add_skill("猴子偷桃")

# for item in manager:
#     print(item)

# for item in manager:
#     print(item)

iterator = manager.__iter__()
while True:
    try:
        item = iterator.__next__()
        print(item)
    except StopIteration:
        break
