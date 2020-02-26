"""
    增加需求：最多只能猜3次,如果超过次数则提示"游戏失败"
"""

# 准备一个随机数工具
import random

# 生成一个随机数
random_number = random.randint(1, 100)
print(random_number)
count = 0
while count < 3:
    count += 1
    input_number = int(input("请输入："))
    if input_number > random_number:
        print("大了")
    elif input_number < random_number:
        print("小了")
    else:
        print("终于猜对了")
        break
else:# 否则：当循环条件不满足时执行(从循环体中通过break退出不执行)
    print("游戏失败")
