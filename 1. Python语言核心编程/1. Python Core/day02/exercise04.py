"""
    练习1：收银台
        获取商品价格
        获取购买数量
        获取支付金额
        显示:应该找回?元.
        算法：支付金额 - 价格 × 数量
"""
# 15：22
commodity_price = input("请输入商品价格：")
buy_count = input("请输入购买数量：")
money = input("请输入金额：")

result = float(money) - float(commodity_price) * int(buy_count)

print("应该找回" + str(result) + "元.")




