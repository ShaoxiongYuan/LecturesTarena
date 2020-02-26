# 体会：无论神马容器,获取元素的代码相同

# 练习1： 通过迭代思想,遍历元组("悟空","八戒","唐僧")所有元素
tuple01 = ("悟空", "八戒", "唐僧")
iterator = tuple01.__iter__()
while True:
    try:
        item = iterator.__next__()
        print(item)
    except StopIteration:
        break

# 练习2： 获取字典所有元素{"悟空":101,"八戒":102,"唐僧";103}
#        要求：不能使用for循环
dict01 = {"悟空": 101, "八戒": 102, "唐僧": 103}
iterator = dict01.__iter__()
while True:
    try:
        key = iterator.__next__()
        print(key, dict01[key])
    except StopIteration:
        break
