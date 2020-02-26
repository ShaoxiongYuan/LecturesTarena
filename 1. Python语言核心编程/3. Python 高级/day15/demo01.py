"""
    异常处理
"""


def div_apple(apple_count):
    person_count = int(input("请输入人数："))
    result = apple_count / person_count
    print("每个人分得%d个苹果" % result)


# 写法1：
# try:
#     div_apple(10)
# except Exception:
#     print("出错啦")

# 写法2:
# try:
#     div_apple(10)
# except ValueError:
#     print("不能输入非整数")
# except ZeroDivisionError:
#     print("不能输入零")

# 写法3:
# try:
#     div_apple(10)
# except ValueError:
#     print("不能输入非整数")
# except ZeroDivisionError:
#     print("不能输入零")
# else:
#     print("分苹果成功啦")

# 写法4:
try:
    div_apple(10)
finally:
    print("无论对错一定执行的逻辑")
