# 练习：自定义my_zip函数
def my_zip(iterable01,iterable02):
# def my_zip(*args):
#                       长度最小的元素
    for i in range(len(iterable01)):
        yield (iterable01[i],iterable02[i])

list01 = ["唐僧", "猪八戒", "悟空"]
list02 = [101, 102]

for item in my_zip(list01,list02):
    print(item)
