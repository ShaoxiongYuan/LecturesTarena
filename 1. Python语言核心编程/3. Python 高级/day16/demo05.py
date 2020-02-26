"""
    函数式编程应用
"""
list01 = [43, 54, 5667, 7, 7, 8]


# 多个函数/方法, 在大致行为上是一致的,在个别算法上不同,需要使用函数式编程思想.

def find01():
    for item in list01:
        if item % 2:
            yield item


def find02():
    for item in list01:
        if item > 10:
            yield item


# "封装":分
def condition01(item):
    return item % 2
def condition02(item):
    return item > 10

# "继承":隔
def find(func):
    for item in list01:
        # if item > 10:
        # if condition02(item):
        if func(item):
            yield item

# "多态":做
for item in find(condition02):
    print(item)