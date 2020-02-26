"""
    函数
"""


# 代码的重复,是万恶之源

# 创建函数
def attack():
    """
        攻击
    """
    print("直拳")
    print("勾拳")
    print("侧踹")


# 参数：调用者  给  创建者 传递的信息
#      创建者 要求 调用者 提供的信息
def attack_repeat(count):  # 形式参数
    """
        重复攻击
    :param count:  int类型的 攻击次数
    """
    for __ in range(count):
        attack()


# 调用函数
attack()
attack_repeat(3)  # 实际参数
attack_repeat(10)
