"""
4. 彩票：双色球
    蓝色：6个  1-33之间整数  不能重复
    红色：1个  1-16之间整数
   -- 随机创建一注彩票（列表,红色作为最后一个元素）
   -- 在终端中录入（购买）一注彩票
    提示："请输入第1个蓝色号码"   "数字超过范围"   "号码已经存在"
"""
# -- 随机创建一注彩票
import random

# list_ticket = []
# while len(list_ticket) < 6:
#     ticket = random.randint(1, 33)
#     if ticket not in list_ticket:
#         list_ticket.append(ticket)
# list_ticket.append(random.randint(1, 16))
#
# print(list_ticket)

# -- 在终端中录入（购买）一注彩票
list_ticket = []
while len(list_ticket) < 6:
    ticket = int(input("请输入第%d个蓝球号码：" % (len(list_ticket) + 1)))
    if ticket < 1 or ticket > 33:
        print("超过范围")
    elif ticket in list_ticket:
        print("号码已经存在")
    else:
        list_ticket.append(ticket)

while len(list_ticket) < 7:
    ticket = int(input("请输入红球号码："))
    if 1 <= ticket <= 16:
        list_ticket.append(ticket)
    else:
        print("超过范围")

print(list_ticket)
