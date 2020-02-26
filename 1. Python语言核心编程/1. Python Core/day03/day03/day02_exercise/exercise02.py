"""
    5. 获取总秒数
       打印：几小时几分钟几秒
"""
total_second = int(input("请输入总秒数"))
hour = total_second // 60 // 60
second = total_second % 60
minute = total_second // 60 % 60
print(str(hour) + "小时" + str(minute) + "分钟" + str(second) + "秒")

# 10:05