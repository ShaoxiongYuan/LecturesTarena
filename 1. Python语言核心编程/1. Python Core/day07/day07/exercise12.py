"""
    改造 day03/exercise04 代码
    定义函数,根据成绩判断等级.
"""


# def calculate_score_level(score):
#     if score > 100 or score < 0:
#         return "不在范围内0-100"
#     elif 90 <= score:
#         return "优秀"
#     elif 80 <= score:
#         return "良好"
#     elif 60 <= score:
#         return "及格"
#     else:
#         return "不及格"

# 因为return 退出函数，所以无需else语句
def calculate_score_level(score):
    if score > 100 or score < 0:
        return "不在范围内0-100"
    if 90 <= score:
        return "优秀"
    if 80 <= score:
        return "良好"
    if 60 <= score:
        return "及格"
    return "不及格"

print(calculate_score_level(95))
