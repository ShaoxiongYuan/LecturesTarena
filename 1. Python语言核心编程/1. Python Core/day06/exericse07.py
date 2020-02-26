"""
    一个公司有如下员工
        经理:"曹操","刘备","孙权"
        技术员:"曹操","刘备","张飞","关羽"
    1. 选择数据结构,存储以上信息.
    2. 是经理也是技术的都有谁？
    3. 是经理也不是技术的都有谁？
    4. 不是经理也是技术的都有谁？
    5. 公司总共有多少人?
    6. 张飞是经理吗？
"""
dict_persons = {
    "经理": {"曹操", "刘备", "孙权"},
    "技术员": {"曹操", "刘备", "张飞", "关羽"}
}
print(dict_persons["经理"] & dict_persons["技术员"])
print(dict_persons["经理"] - dict_persons["技术员"])
print(dict_persons["技术员"] - dict_persons["经理"])
print(len(dict_persons["技术员"] | dict_persons["经理"]))
print("张飞" in dict_persons["经理"])
