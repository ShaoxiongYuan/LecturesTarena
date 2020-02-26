"""
    5. (扩展)计算字符中每个字符出现的数量
    "abacdce"
        a出现的次数是2次
        b出现的次数是1次
        c出现的次数是2次
        d出现的次数是1次
        e出现的次数是1次
"""
message = "abacdce"
# 数据结构
#    dict_result = { 字符:次数 }
# 核心算法
#    添加(新字) dict_result[字符] = 1
#    修改(旧字) dict_result[字符] += 1

dict_result = {}
for item in message:
    if item in dict_result:# 如果字符已经记录过
        dict_result[item] += 1
    else:
        dict_result[item] = 1

for k,v in dict_result.items():
    print("%s出现的次数是%d次"%(k,v))

