"""
    在终端中循环录入学生信息(名称,年龄,性别,成绩...)
    如果名称为空,停止录入.
    -- 打印所有学生信息(一行一个)
        格式：xxx的年龄是xxx,性别是xxx,成绩是xxx.
    -- 打印最后一个学生的信息
           ...
    核心：数据结构  列表内嵌字典
        [
            {"name":"悟空", "age":26,"sex":"男","score":86},
            {"name":"唐僧", "age":24,"sex":"男","score":90},
        ]
    17:02
    总结：
        列表：通过索引/切片获取数据,定位元素灵活.      【灵活】
        字典：通过key查找值,代码可读性相对较高
              (列表不适合存储多类信息，姓名/年龄/性别)
             利用空间,换取时间,定义单个元素最快.      【快速】
"""
list_persons = []
while True:
    name = input("请输入学生姓名：")
    if name == "":
        break
    age = int(input("请输入学生年龄："))
    sex = input("请输入学生性别：")
    score = float(input("请输入学生成绩："))
    dict_person = {"name":name, "age":age,"sex":sex,"score":score}
    list_persons.append( dict_person )

for person in list_persons:
    print("%s的年龄是%d,性别是%s,成绩是%f." % (person["name"], person["age"],person["sex"],person["score"]))

# 最后一个学生信息
person = list_persons[-1]
print("%s的年龄是%d,性别是%s,成绩是%f." % (person["name"], person["age"], person["sex"], person["score"]))
