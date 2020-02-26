"""
    在终端中循环录入学生姓名，如果录入空，则停止.倒序输出所有人。
    要求：姓名不能重复(如果重复提示，不存储.)
"""
list_names = []
while True:
    name = input("请输入姓名：")
    if name == "":
        break
    if name not in list_names:
        list_names.append(name)
    else:
        print(name + "已经存在")

for i in range(len(list_names) - 1, -1, -1):
    print(list_names[i])
