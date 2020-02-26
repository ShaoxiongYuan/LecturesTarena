"""
    continue
    练习:exercise04

    语句小结：
        循环语句：
            while :根据条件重复执行(不知道次数)
                例如：对折到超过珠穆朗玛则停止
            for : 根据次数执行(循环计数)
                例如：3道考试题/对折20次
        跳转语句：
            break：跳出(循环停止)
            continue：跳过(循环继续)
            备注：如果循环嵌套，跳转语句只对一层起作用.

        range 属于可迭代对象,所以获取数据使用for.
"""
# 累加：1--100之间能被3整除的整数
# sum_value = 0
# for number in range(1, 101):
#     满足条件则累加
#     if number % 3 == 0:
#         sum_value += number
# print(sum_value)

sum_value = 0
for number in range(1, 101):
    #     不满足条件则跳过
    if number % 3 != 0:
        continue  # 跳过(当前元素,继续循环)
    sum_value += number
print(sum_value)



