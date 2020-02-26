"""
    标准库模块
        time
"""
import time
# 获取当前时间戳 1970年1月1日 到现在 经过的秒数
print(time.time())# 1579161897.7612894
# 获取当前时间元组(年,月,日,时,分,秒,星期,这年的第几天,与夏令时的偏移量)
print(time.localtime())

# 时间戳 --> 时间元组
time_tuple = time.localtime(1579161897.7612894)
# 时间元组 -->时间戳
print(time.mktime(time_tuple))

# 时间元组 --- > str
# 20/01/16 16:04:57
print(time.strftime("%y/%m/%d %H:%M:%S",time_tuple))
# 2020/01/16 16:04:57
print(time.strftime("%Y/%m/%d %H:%M:%S",time_tuple))

#  str--- > 时间元组
new_time_tuple = time.strptime("2020/01/16 16:04:57","%Y/%m/%d %H:%M:%S")
print(new_time_tuple)














