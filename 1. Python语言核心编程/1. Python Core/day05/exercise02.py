"""
    在终端中循环录入学生成绩，如果录入空，则停止。
    打印最高分、最低分、平均分.
    体会：容器
"""
list_scores = []
while True:
    str_score = input("请输入成绩：")
    if str_score == "":
        break
    list_scores.append(int(str_score))

print("最高分：%d" % max(list_scores))
print("最低分：%d" % min(list_scores))
print("平均分：%f" % (sum(list_scores) / len(list_scores)))
