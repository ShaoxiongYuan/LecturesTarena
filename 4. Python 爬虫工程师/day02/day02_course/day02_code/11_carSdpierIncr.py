"""
汽车之家二手车信息抓取
思路
    1、一级页面：汽车的链接
    2、二级页面：具体汽车信息

目标：用redis的集合实现增量爬虫,新更新的汽车信息在前面
"""
import requests
import re
import time
import random
from fake_useragent import UserAgent
import redis
from hashlib import md5
import sys
import pymongo

class CarSpider:
    def __init__(self):
        self.url = 'https://www.che168.com/beijing/a0_0msdgscncgpi1lto1csp{}exx0/'
        # 连接到redis
        self.r = redis.Redis(host='127.0.0.1', port=6379, db=0)
        # 3个对象
        self.conn = pymongo.MongoClient(host='localhost', port=27017)
        self.db = self.conn['cardb']
        self.myset = self.db['carset']

    def get_html(self, url):
        """功能函数1 - 获取html"""
        headers = { 'User-Agent':UserAgent().random }
        html = requests.get(url=url, headers=headers).text

        return html

    def re_func(self, regex, html):
        """功能函数2 - 正则解析函数"""
        pattern = re.compile(regex, re.S)
        r_list = pattern.findall(html)

        return r_list

    def md5_url(self, url):
        """功能函数3：对URL地址进行md5加密"""
        s = md5()
        s.update(url.encode())

        return s.hexdigest()

    def parse_html(self, one_url):
        """爬虫逻辑函数"""
        one_html = self.get_html(url=one_url)
        one_regex = '<li class="cards-li list-photo-li".*?<a href="(.*?)".*?</li>'
        href_list = self.re_func(regex=one_regex, html=one_html)
        for href in href_list:
            two_url = 'https://www.che168.com' + href
            # 生成指纹,去redis的集合中比对
            finger = self.md5_url(two_url)
            if self.r.sadd('car:spider', finger) == 1:
                try:
                    # 获取1辆汽车的具体信息
                    self.get_car_info(two_url)
                    # 控制爬取频率
                    time.sleep(random.randint(1,2))
                except Exception as e:
                    # 删除刚刚添加的指纹
                    self.r.srem('car:spider', finger)
            else:
                sys.exit('更新完成')

    def get_car_info(self, two_url):
        """获取1辆汽车的具体信息"""
        two_html = self.get_html(url=two_url)
        two_regex = '<div class="car-box">.*?<h3 class="car-brand-name">(.*?)</h3>.*?<h4>(.*?)</h4>.*?<h4>(.*?)</h4>.*?<h4>(.*?)</h4>.*?<h4>(.*?)</h4>.*?<span class="price" id="overlayPrice">￥(.*?)<b>'
        # car_list: [('福睿斯','3万公里','2016年3月','手动 / 1.5L', '廊坊', '5.60'),]
        car_list = self.re_func(regex=two_regex, html=two_html)
        item = {}
        item['name'] = car_list[0][0].strip()
        item['km'] = car_list[0][1].strip()
        item['time'] = car_list[0][2].strip()
        item['type'] = car_list[0][3].split('/')[0].strip()
        item['displace'] = car_list[0][3].split('/')[1].strip()
        item['address'] = car_list[0][4].strip()
        item['price'] = car_list[0][5].strip()
        print(item)
        # 数据存入到mongodb数据库
        self.myset.insert_one(item)

    def run(self):
        for i in range(1,5):
            url = self.url.format(i)
            self.parse_html(url)

if __name__ == '__main__':
    spider = CarSpider()
    spider.run()













