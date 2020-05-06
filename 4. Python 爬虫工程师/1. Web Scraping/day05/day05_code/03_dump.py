"""
json.dump()
使用场景：
    把抓取的数据存到本地json文件中
"""
import json

all_film_list = [
    {'rank':'1', 'name':'霸王别姬', 'star':'张国荣'},
    {'rank':'2', 'name':'大话西游', 'star':'周星驰'}
]
# 保存到 film.json 文件中
with open('film.json', 'w', encoding='utf-8') as f:
    json.dump(all_film_list, f, ensure_ascii=False)





















