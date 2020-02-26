# 根据格式显示字符串
# ?天?小时?分钟
# print(day + "天" + hour + "小时" + minute + "分钟")
day = 1
hor = 2
minute = 3
print("%d天%d小时%d分钟" % (day, hor, minute))

# ?斤零?两
# print(str(jin) + "斤零" + str(liang) + "两")
jin = 3
liang = 2
message = "%d斤零%d两" % (jin, liang)
print(message)

# ?+?=?
# print(str(number01) + "+" + str(number02) + "=?")
number01 = 10
number02 = 5
print("%d+%d=%d" % (number01, number02, number01 + number02))
