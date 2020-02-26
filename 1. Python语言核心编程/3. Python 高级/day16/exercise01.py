
# 练习1：参照enumerate自定义my_enumerate函数。
def my_enumerate(iterable):
    index = 0
    for item in iterable:
        yield (index,item)
        index += 1

list01 = ["唐僧", "猪八戒", "悟空"]
dict01 = {"唐僧": 101, "八戒": 102, "悟空": 103}

for i,item in my_enumerate(list01):
    print(i,item)

# 练习2：将list01中长度大于2的元素设置为""。
# for i in range(len(list01)):
#     if len(list01[i]) > 2:
#         list01[i] = ""

for i, item in enumerate(list01):
    if len(item) > 2:
        list01[i] = ""

print(list01)












