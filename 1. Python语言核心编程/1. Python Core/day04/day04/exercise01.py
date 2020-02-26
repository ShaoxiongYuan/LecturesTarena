# 练习：
# 获取一个任意整数           123456
# 输出每位相加和            1+2+3+4+...
str_number = input("请输入任意整数：")

# 1. 循环以前定义变量(存储累加和)
sum_value = 0

for item in str_number:
    # 2. 循环以内进行累加
    sum_value += int(item)

# 3. 循环以外获取结果
print(sum_value)
