"""
豆瓣电影数据抓取 - 抓取所有类别下的所有电影
思路：
    请输入电影类别：动作
    把所有的动作类型的电影抓取下来
"""
import requests
import time
import random
from fake_useragent import UserAgent
import json
import re
import pymongo

class DoubanSpider:
    def __init__(self):
        self.url = 'https://movie.douban.com/j/chart/top_list?type={}&interval_id=100%3A90&action=&start={}&limit=20'
        # 创建3个对象
        self.conn = pymongo.MongoClient('localhost', 27017)
        self.db = self.conn['doubandb']
        self.myset = self.db['doubanset']

    def get_html(self, url):
        """功能函数1 - 请求函数"""
        headers = { 'User-Agent' : UserAgent().random }
        html = requests.get(url=url, headers=headers).text

        return html

    def parse_html(self, url):
        """提取电影信息的函数"""
        # json.loads() : 把json格式的字符串转为python数据类型
        html = json.loads(self.get_html(url=url))
        for one_film in html:
            item = {}
            item['rank'] = one_film['rank']
            item['name'] = one_film['title']
            item['time'] = one_film['release_date']
            item['score'] = one_film['score']
            print(item)
            # 存入mongodb数据库
            self.myset.insert_one(item)

    def run(self):
        """程序入口函数"""
        all_dict = self.get_all_dict()
        # 剧情 | 喜剧 | 爱情 | 动作 |
        menu = ''
        for one_type in all_dict:
            menu = menu + one_type + ' | '
        print(menu)
        choice = input('请选择:')
        choice_type = all_dict[choice]

        # 获取某个类别下的电影的总数量
        total = self.get_total(choice_type)
        for start in range(0, total, 20):
            url = self.url.format(choice_type, start)
            self.parse_html(url=url)
            time.sleep(random.randint(1,2))

    def get_all_dict(self):
        """获取所有类别以及type值的字典"""
        index_url = 'https://movie.douban.com/chart'
        html = self.get_html(url=index_url)
        regex = '<span><a.*?type_name=(.*?)&type=(.*?)&interval_id=100:90&action=">'
        pattern = re.compile(regex, re.S)
        # r_list: [('剧情','11'),('喜剧','5'),('爱情':'13'), ... ...]
        r_list = pattern.findall(html)
        # 处理成字典
        all_dict = {}
        for r in r_list:
            all_dict[r[0]] = r[1]

        return all_dict

    def get_total(self, choice_type):
        """获取某个类别下的电影的总数量"""
        page_url = 'https://movie.douban.com/j/chart/top_list_count?type={}&interval_id=100%3A90'.format(choice_type)
        html = json.loads(self.get_html(url=page_url))
        total = html['total']

        return total

if __name__ == '__main__':
    spider = DoubanSpider()
    spider.run()


















