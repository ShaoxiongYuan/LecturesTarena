"""
    迭代
    练习：exercise03
"""
list01 = [4, 3, 56, 58, 52]
for item in list01:
    print(item)
# 笔试题：
#    对象能够参与for循环的条件：
#           对象可以获取迭代器对象（对象必须具有__iter__方法）

# for循环原理
# 1. 获取迭代器对象
iterator = list01.__iter__()
# 2. 获取下一个元素
while True:
    try:
        item = iterator.__next__()
        print(item)
    # 3. 如果没有元素则停止循环
    except StopIteration:
        break
