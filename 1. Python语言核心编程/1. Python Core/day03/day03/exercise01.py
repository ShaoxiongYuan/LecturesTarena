"""
    修改day02/exercise04练习
    显示结果：
        应找回...
        钱不够
"""

commodity_price = input("请输入商品价格：")
buy_count = input("请输入购买数量：")
money = input("请输入金额：")
result = float(money) - float(commodity_price) * int(buy_count)
if result >= 0:
    print("应该找回" + str(result) + "元.")
else:
    print("钱不够")
