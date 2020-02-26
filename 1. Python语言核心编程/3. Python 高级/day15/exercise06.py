"""
    自定义range(整数生成器)类

    for item in MyRange(5):
        print(item)# 0 1 2 3 4

    16：40
"""
class MyRangeIterator:
    def __init__(self, end):
        self.__start = -1
        self.__end = end

    def __next__(self):
        self.__start += 1
        if self.__start >= self.__end:
            raise StopIteration()
        return self.__start

class MyRange:
    def __init__(self, end):
        self.__end = end

    def __iter__(self):
        return MyRangeIterator(self.__end)

# 循环一次  计算一次  返回一次
for item in MyRange(5):
    print(item)  # 0 1 2 3 4

# range = MyRange(9999999999999999999999999999999999999)
# iterator = range.__iter__()
# while True:
#     try:
#         item = iterator.__next__()
#         print(item)
#     except StopIteration:
#         break
