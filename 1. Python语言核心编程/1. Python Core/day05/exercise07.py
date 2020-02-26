# 在终端中循环录入字符串,如果录入为空，则停止.
# 打印所有录入的内容(一个字符串)

# 核心思想：使用可变对象代替不可变对象,进行频繁操作.
list_result = []
while True:
    content = input("请输入内容：")
    if content == "":
        break
    list_result.append(content)

str_result =  "".join(list_result)
print(str_result)
