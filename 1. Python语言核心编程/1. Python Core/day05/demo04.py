"""
    list --> str
    练习:exercise07
"""
# 根据某些逻辑,拼接字符串.
list01 = ["3", "4", "5", "7"]
# str_result = ""
# for item in list01:
#     # 每次循环,每次拼接,每次创建新对象
#     str_result = str_result + item
# print(str_result)
str_result = "——".join(list01)
print(str_result)
