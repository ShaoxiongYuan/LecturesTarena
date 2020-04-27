"""
豆瓣电影 - 示例代码
"""

import requests

url = 'https://movie.douban.com/j/chart/top_list?type=11&interval_id=100%3A90&action=&start=0&limit=20'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'}

html = requests.get(url=url, headers=headers).json()
item = {}
for one_film in html:
    item['rank'] = one_film['rank']
    item['name'] = one_film['title']
    item['score'] = one_film['score']
    item['time'] = one_film['release_date']
    print(item)















