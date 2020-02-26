# 定义函数,在列表中找出大于10的数字
#  -- 使用传统方法实现  定义列表返回结果
#  -- 使用生成器实现    通过yield返回结果
list01 = [3,42,34,435,5,64,5,67]
# 适用性：
# 函数向外返回单个结果   --->  return
# 函数向外返回多个结果   --->  yield

def get_numbers01():
    list_result = []
    for item in list01:
        if item > 10:
            list_result.append(item)
    return list_result

result = get_numbers01()
for item in result:
    print(item)

def get_numbers02():
    for item in list01:
        if item > 10:
            yield item

result = get_numbers02()
for item in result:
    print(item)