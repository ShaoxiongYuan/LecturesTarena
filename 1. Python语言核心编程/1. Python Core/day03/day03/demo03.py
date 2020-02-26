"""
    真值表达式
    条件表达式
    练习:exercise06
"""

# 真值表达式
message = input("请输入")
# if message != "":
if message:  # 有值
    print("输入了内容")
else:
    print("没有输入内容")

# 条件表达式
# if input("请输入性别：") == "男":
#     sex_code = 1
# else:
#     sex_code = 0
sex_code = 1 if input("请输入性别：") == "男" else 0
