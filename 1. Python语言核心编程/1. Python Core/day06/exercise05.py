"""
    在终端中循环录入学生信息(名称,年龄,性别,成绩...)
    如果名称为空,停止录入.
    -- 打印所有学生信息(一行一个)
        格式：xxx的年龄是xxx,性别是xxx,成绩是xxx.
    -- 如果录入了"唐僧",单独打印其成绩
    核心：数据结构  字典内嵌列表
        {
            "悟空":[26,"男",86],
            "唐僧":[24,"男",90]
        }
"""
dict_persons = {}
while True:
    name = input("请输入学生姓名：")
    if name == "":
        break
    age = int(input("请输入学生年龄："))
    sex = input("请输入学生性别：")
    score = float(input("请输入学生成绩："))
    dict_persons[name] = [age,sex,score]

for k_name, v_infos in dict_persons.items():
    print("%s的年龄是%d,性别是%s,成绩是%f." % (k_name, v_infos[0],v_infos[1],v_infos[2]))

if "唐僧" in dict_persons:
    print(dict_persons["唐僧"][2])