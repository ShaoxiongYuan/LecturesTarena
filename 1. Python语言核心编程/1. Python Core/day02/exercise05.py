"""
    练习:时间换算
        获取分钟数
        获取小时数
        获取天数
        显示：?天?小时?分钟总共是?秒.
"""
minute = input("请输入分钟数：")
hour = input("请输入小时数：")
day = input("请输入天数：")
result = int(minute) * 60 + int(hour) * 60 * 60 + int(day) * 24 * 60 * 60
print(day + "天" + hour + "小时" + minute + "分钟总共是" + str(result) + "秒.")
