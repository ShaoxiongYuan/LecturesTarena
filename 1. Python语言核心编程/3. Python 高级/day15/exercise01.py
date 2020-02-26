# 练习：通过异常处理，保证get_score可以获取正确的成绩
def get_score():
    while True:
        try:
            score = float(input("请输入成绩："))
            return score
        except:
            print("输入有误")

print(get_score())