"""
4. 定义数据结构，存储以下数据
    (1) "qtx"喜欢"编码","看书","跑步","音乐"
        "lzmly"喜欢"刷抖音","看电影","吃"
         ...
        -- 打印qtx的所有喜好(一行一个)
        -- 打印所有人的名称(一行一个)
        -- 打印所有人的喜好(一行一个)
    (2)
        "北京"
            "景区" -- "故宫","天安门","天坛"
            "美食" -- "豆汁","焦圈","烤鸭"
        "天津"
            "景区" -- "天津之眼","瓷房子"
            "美食" -- "狗不理包子","大麻花","煎饼"
        -- 打印北京的所有美食(一行一个)
        -- 打印天津的所有景区(一行一个)
        -- 打印所有城市(一行一个)
        -- 打印所有景区(一行一个)
        -- 给北京增加一个"美食"炸酱面
"""
dict_infos = {
    "qtx": ["编码", "看书", "跑步", "音乐"],
    "lzmly": ["刷抖音", "看电影", "吃"]
}

# 1.
for item in dict_infos["qtx"]:
    print(item)

for key in dict_infos:
    print(key)

for key in dict_infos:  # 遍历所有key
    for item in dict_infos[key]:  # 遍历当前key对象的值
        print(item)

# 2.
dict_infos = {
    "北京": {
        "景区": ["故宫", "天安门", "天坛"],
        "美食": ["豆汁", "焦圈", "烤鸭"]
    },
    "天津": {
        "景区": ["天津之眼", "瓷房子"],
        "美食": ["狗不理包子", "大麻花", "煎饼"]
    }
}

for item in dict_infos["北京"]["美食"]:
    print(item)
for item in dict_infos["天津"]["景区"]:
    print(item)
for key in dict_infos:
    print(key)

for key in dict_infos:# 获取所有城市
    for item in dict_infos[key]["景区"]:# 获取当前城市景区
        print(item)

dict_infos["北京"]["美食"].append("炸酱面")