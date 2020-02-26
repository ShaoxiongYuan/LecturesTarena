"""
    闭包
        字面意思：封闭，内存
                   保存外部函数执行的栈帧
        价值：逻辑连续
"""
def give_gife_money(money):
    """
        获取压岁钱
    """
    print("孩子获得了%d元压岁钱" % money)

    def child_buy(target, price):
        """
            孩子购买商品
        """
        nonlocal money
        money -= price
        print("孩子购买了%s商品,花了%d元,还剩下%d元" % (target, price, money))

    return child_buy

action = give_gife_money(1000)# 调用外部函数,返回内部函数
action("变形金刚", 200)# 调用内部函数
action("芭比娃娃", 300)
action("手机", 500)
