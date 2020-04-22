"""
汽车之家二手车信息抓取
思路
    1、一级页面：汽车的链接
    2、二级页面：具体汽车信息

建立User-Agent池：防止被网站检测到是爬虫
    使用fake_useragent模块
    安装：sudo pip3 install fake_useragent
    使用：
        from fake_useragent import UserAgent
        UserAgent().random
"""
import requests
import re
import time
import random
from fake_useragent import UserAgent

class CarSpider:
    def __init__(self):
        self.url = 'https://www.che168.com/beijing/a0_0msdgscncgpi1lto1csp{}exx0/'

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

    def parse_html(self, one_url):
        """爬虫逻辑函数"""
        one_html = self.get_html(url=one_url)
        one_regex = '<li class="cards-li list-photo-li".*?<a href="(.*?)".*?</li>'
        href_list = self.re_func(regex=one_regex, html=one_html)
        for href in href_list:
            two_url = 'https://www.che168.com' + href
            # 获取1辆汽车的具体信息
            self.get_car_info(two_url)
            # 控制爬取频率
            time.sleep(random.randint(1,2))

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

    def run(self):
        for i in range(1,5):
            url = self.url.format(i)
            self.parse_html(url)

if __name__ == '__main__':
    spider = CarSpider()
    spider.run()













