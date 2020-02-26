"""
    累加10 -- 80 之间个位不是3/5/9的整数
    调试：体会continue的过程
"""
sum_value = 0
for number in range(10, 81):
    unit01 = number % 10
    # 是他们就跳过
    if unit01 == 3 or unit01 == 5 or unit01 == 9:
        continue
    sum_value += number
print(sum_value)
