"""
    while
    练习：exercise07
"""

# 　死循环
while True:
    str_usd = input("请输入美元:")
    result = float(str_usd) * 6.9845
    print(str_usd + "美元=" + str(result) + "人民币")

    if input("如果输入y则继续：") != "y":
        break  # 退出循环
