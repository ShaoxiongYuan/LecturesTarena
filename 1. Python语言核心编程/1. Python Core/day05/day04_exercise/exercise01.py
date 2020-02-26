"""
3. 在终端中获取一个矩形边长,打印矩形。
   例如：4
   ****
   *  *
   *  *
   ****
   例如：6
   ******
   *    *
   *    *10
   *    *
   *    *
   ******
"""
number = int(input("请输入整数："))

print("*" * number)

for __ in range(number - 2):
    print("*" + " " * (number - 2) + "*")

print("*" * number)
