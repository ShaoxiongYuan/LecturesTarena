"""
    斐波那契数列：从第三项开始，每一项都等于前两项之和.
        1,1,2,3,5,8,....
    获取一个斐波那契数列长度,打印列表.
"""
lenght = int(input("请输入斐波那契数列长度："))
# 6
list_sequence = [1,1]
# 0
# 1
# 2 -->  2-2  2-1
# 3 -->  3-2  3-1
# ? -->  ?-2  ?-1

# for i in range(2,lenght):# 2 3 4 5
#     element = list_sequence[i-2] +  list_sequence[i-1]
#     list_sequence.append(element)

for __ in range(lenght - 2):
    # 当前元素 = 最后两个元素之和
    element = list_sequence[-2] +  list_sequence[-1]
    list_sequence.append(element)
print(list_sequence)



