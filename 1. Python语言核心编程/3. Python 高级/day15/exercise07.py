"""

"""


class MyRange:
    def __init__(self, end):
        self.__end = end

    def __iter__(self):
        start = 0
        while start < self.__end:
            yield start
            start += 1

# 循环一次  计算一次  返回一次
for item in MyRange(9999999999999999999999999999999999999):
    print(item)

# range = MyRange(9999999999999999999999999999999999999)
# iterator = range.__iter__()
# while True:
#     try:
#         item = iterator.__next__()
#         print(item)
#     except StopIteration:
#         break
