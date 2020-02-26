"""
    5. 在终端中重复录入年龄显示信息,如果录入空字符串,则停止录入(程序结束).
    0-1 --> 婴儿
    2-13 --> 儿童
    14-20 --> 青少年
    21-65 --> 成年人
    66 -  --> 老年人
"""

while True:
    str_age = input("请输入年龄：")
    if str_age == "":
        break
    age = int(str_age)
    if age < 0:
        print("输入有误")
    elif age <= 1:
        print("婴儿")
    elif age <= 13:
        print("儿童")
    elif age <= 20:
        print("青少年")
    elif age <= 65:
        print("成年人")
    else:
        print("老年人")
