"""
    内置高阶函数
    10：40
"""

class Wife:
    def __init__(self, name, face_score, money, age):
        self.name = name
        self.face_score = face_score
        self.money = money
        self.age = age


list_wifes = [
    Wife("建宁", 86, 999999, 25),
    Wife("双儿", 95, 5000, 23),
    Wife("苏荃", 98, 10000, 30),
    Wife("阿珂", 100, 6000, 23),
    Wife("铁锤", 80, 0, 35),
]

# 需求：在老婆列表中, 查找颜值大于90的所有老婆
#       在老婆列表中, 查找存款小于十万的所有老婆

# for item in IterableHelper.find_all(list_wifes, lambda item: item.face_score > 90):
#     print(item.__dict__)

# 1. 过滤器
for item in filter(lambda wife: wife.face_score>90,list_wifes):
    print(item.__dict__)

# 需求：在老婆列表中,查找所有老婆的姓名
#      在老婆列表中,查找所有老婆的姓名和颜值
# for item in IterableHelper.select(list_wifes,lambda wife:wife.name):
#     print(item)
# 2. 映射
for item in map(lambda item:item.name,list_wifes):
    print(item)


# 需求：在老婆列表中,查找颜值最高的老婆
#      在老婆列表中,查找存款最多的老婆

#  3. 获取最大/最小
# result = IterableHelper.get_max(list_wifes,lambda wife:wife.money)
# print(result.__dict__)
resutl = max(list_wifes,key = lambda item:item.money)
print(resutl.__dict__)

# 4.排序
# 需求： 根据年龄对老婆列表进行升序排列
#       根据颜值对老婆列表进行升序排列
# IterableHelper.order_by(list_wifes,lambda wife:wife.face_score)
# for item in list_wifes:
#     print(item.__dict__)

# new_list = list_wifes[:]
# IterableHelper.order_by(new_list,lambda wife:wife.face_score)
# for item in new_list:
#     print(item.__dict__)

# 备注：(1)sorted没有改变原有列表元素,而是返回排序后的结果
for item in sorted(list_wifes,key = lambda item:item.age ):
    print(item.__dict__)
#      (2)支持降序排列
for item in sorted(list_wifes,key = lambda item:item.age, reverse=True):
    print(item.__dict__)






