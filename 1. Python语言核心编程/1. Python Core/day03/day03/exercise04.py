"""
    获取成绩
    判断等级(优秀 良好 及格 不及格  不在范围内0-100)
"""

score = float(input("请输入成绩："))
# if 90 <= score <= 100:
#     print("优秀")
# elif 80 <= score < 90:
#     print("良好")
# elif 60 <= score < 80:
#     print("及格")
# elif 0 <= score < 60:
#     print("不及格")
# else:
#     print("不在范围内0-100")

if score > 100 or score < 0:
    print("不在范围内0-100")
elif 90 <= score:
    print("优秀")
elif 80 <= score:
    print("良好")
elif 60 <= score:
    print("及格")
else:
    print("不及格")




