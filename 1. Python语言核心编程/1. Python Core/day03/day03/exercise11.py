"""
    猜数字游戏：
        程序产生1--100之间的随机数.
        在终端中重复猜测,直到猜对为止
        提示：请输入  大了  小了   终于猜对了
"""
# 准备一个随机数工具
import random

# 生成一个随机数
random_number = random.randint(1, 100)

while True:
    input_number = int(input("请输入："))
    if input_number > random_number:
        print("大了")
    elif input_number < random_number:
        print("小了")
    else:
        print("终于猜对了")
        break
