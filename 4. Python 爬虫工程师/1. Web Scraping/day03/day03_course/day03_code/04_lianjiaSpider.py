"""
抓取链家二手房中的100页的数据 - 3000个房源信息
"""
import requests
from lxml import etree
import time
import random
from fake_useragent import UserAgent
import pymongo

class LianjiaSpider:
    def __init__(self):
        self.url = 'https://bj.lianjia.com/ershoufang/pg{}/'
        # 计数
        self.i = 0
        # 创建3个对象
        self.conn = pymongo.MongoClient('localhost', 27017)
        self.db = self.conn['lianjiadb']
        self.myset = self.db['lianjiaset']

    def get_html(self, url):
        headers = { 'User-Agent' : UserAgent().random }
        for i in range(3):
            try:
                html = requests.get(url=url, headers=headers, timeout=3).text
                self.parse_html(html)
                break
            except Exception as e:
                print('Retry')

    def parse_html(self, html):
        """解析提取数据函数"""
        p = etree.HTML(html)
        # 基准xpath
        li_list = p.xpath('//ul[@class="sellListContent"]/li')
        for li in li_list:
            item = {}
            # 名称 + 区域
            name_list = li.xpath('.//div[@class="positionInfo"]/a[1]/text()')
            item['name'] = name_list[0].strip() if name_list else None
            add_list = li.xpath('.//div[@class="positionInfo"]/a[2]/text()')
            item['address'] = add_list[0].strip() if add_list else None
            # 户型+面积+方位+是否精装+楼层+年代+类型
            house_li = li.xpath('.//div[@class="houseInfo"]/text()')
            if house_li:
                # house_li: ['户型','面积','方位','是否精装','楼层','年代','类型']
                house_li = house_li[0].split('|')
                if len(house_li) == 7:
                    item['model'] = house_li[0].strip()
                    item['area'] = house_li[1].strip()
                    item['direct'] = house_li[2].strip()
                    item['perfect'] = house_li[3].strip()
                    item['floor'] = house_li[4].strip()
                    item['year'] = house_li[5].strip()
                    item['type'] = house_li[6].strip()
                else:
                    item['model']=item['area']=item['direct']=item['perfect']=item['floor']=item['year']=item['type'] = None
            else:
                item['model'] = item['area'] = item['direct'] = item['perfect'] = item['floor'] = item['year'] = item['type'] = None
            # 总价+单价
            total_list = li.xpath('.//div[@class="totalPrice"]/span/text()')
            item['total'] = total_list[0].strip() if total_list else None
            unit_list = li.xpath('.//div[@class="unitPrice"]/span/text()')
            item['unit'] = unit_list[0].strip() if unit_list else None
            print(item)
            # 存入到mongodb数据库
            self.myset.insert_one(item)

            self.i += 1

    def run(self):
        """程序入口函数"""
        for pg in range(1,101):
            url = self.url.format(pg)
            self.get_html(url=url)
            # 控制爬取频率
            time.sleep(random.randint(1,2))
        print('number:',self.i)

if __name__ == '__main__':
    spider = LianjiaSpider()
    spider.run()











