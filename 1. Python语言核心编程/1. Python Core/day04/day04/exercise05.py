"""
    练习1:在终端中获取一个字符串,打印出每个文字的编码值.
    练习2:在终端中循环录入编码值,打印每个文字.
          要求：如果录入空字符串,则程序退出。
"""
content = input("请输入：")
for item in content:
    print(ord(item))

while True:
    value = input("请输入编码值：")
    if value == "":
        break
    char = chr(int(value))
    print(char)
