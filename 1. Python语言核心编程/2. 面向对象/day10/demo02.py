"""
    类成员
        类变量：大家的数据
        类方法：大家的行为
        核心逻辑：
            类.?
"""



class ICBC:

    # 总行的钱
    total_money = 1000000

    @classmethod
    def print_total_money(cls):
        print("总行的钱：", cls.total_money)

    def __init__(self, name="", money=0):
        # 实例变量
        self.name = name
        # 支行的钱
        self.money = money
        # 从总行扣除当前支行需要的钱
        ICBC.total_money -= money


i01 = ICBC("天坛支行", 100000)
i02 = ICBC("陶然亭支行", 200000)
# print("总行的钱：",ICBC.total_money)
ICBC.print_total_money()  # print_total_money(ICBC)
