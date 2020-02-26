"""
    生成器函数
"""
"""
生成器 = 可迭代对象 + 迭代器（主要）
class Generator:
    def __iter__(self):  
        return self 
    
     def __next__(self): 
        return ... 
"""

def my_range(end):
    start = 0
    while start < end:
        yield start
        start += 1

# 循环一次  计算一次  返回一次
for item in my_range(5):
    print(item)

# range = my_range(5)
# iterator = range.__iter__()
# while True:
#     try:
#         item = iterator.__next__()
#         print(item)
#     except StopIteration:
#         break
