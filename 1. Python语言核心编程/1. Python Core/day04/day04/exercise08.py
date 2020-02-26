"""
    八大行星：
    1. 创建列表存储行星：
      "金星","地球","火星","木星","土星","天王星"
    2. 在第一个位置插入 "水星"
    3. 在末尾追加"海王星"
    4. 打印从太阳到地球之间的行星(前两个行星)
    5. 打印地球以后的行星(一行一个)
    6. 倒序打印所有行星
"""
list_planets = ["金星","地球","火星","木星","土星","天王星"]
list_planets.insert(0,"水星")
list_planets.append("海王星")
print(list_planets[:2])

# for item in list_planets[3:]:# 拷贝了新列表
#     print(item)

for i in range(3,len(list_planets)):
    print(list_planets[i])

for i in range(len(list_planets)-1,-1,-1):
    print(list_planets[i])