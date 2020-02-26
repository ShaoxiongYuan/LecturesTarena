"""
    随机加法考试题
        程序产生2个随机数
        获取( a+b=? )两个数相加结果
        如果答对加10分,否则减5分.
        总共3道题,最后打印分数。
"""
import random

score = 0
for __ in range(3):
    number01 = random.randint(1, 100)
    number02 = random.randint(1, 100)
    message = str(number01) + "+" + str(number02) + "=?"
    input_number = int(input(message))
    # if input_number  ==  number01 + number02:
    #     score += 10
    # else:
    #     score -= 5
    score += 10 if input_number == number01 + number02 else -5

print("总分："+str(score))