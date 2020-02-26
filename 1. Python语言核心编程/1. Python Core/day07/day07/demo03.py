"""
    推导式嵌套
        练习:exercise08
"""

list_fruits = ["香蕉", "苹果", "哈密瓜"]
list_drinks = ["可乐", "雪碧", "牛奶", "咖啡"]
# list_result = []
# for fruit in list_fruits:
#     for drink in list_drinks:
#         list_result.append(fruit + drink)
list_result = [fruit + drink for fruit in list_fruits for drink in list_drinks]
print(list_result)
