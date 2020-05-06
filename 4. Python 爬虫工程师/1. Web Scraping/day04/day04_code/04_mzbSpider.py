"""
民政部行政区划代码数据抓取
"""
import requests
from lxml import etree
import redis
import re
from hashlib import md5
import sys

class MzbSpider:
    def __init__(self):
        self.url = 'http://www.mca.gov.cn/article/sj/xzqh/2020/'
        self.headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'}
        self.r = redis.Redis(host='127.0.0.1', port=6379, db=0)

    def get_html(self, url):
        """功能函数1 - 请求"""
        html = requests.get(url=url, headers=self.headers).text
        return html

    def xpath_func(self, html, xpath_bds):
        """功能函数2 - xpath解析"""
        p = etree.HTML(html)
        r_list = p.xpath(xpath_bds)

        return r_list

    def md5_url(self, url):
        """对URL地址进行md5加密,生成指纹"""
        s = md5()
        s.update(url.encode())

        return s.hexdigest()

    def parse_html(self):
        one_html = self.get_html(url=self.url)
        one_xpath = '//table//tr[2]/td[2]/a/@href'
        href_list = self.xpath_func(html=one_html, xpath_bds=one_xpath)
        if href_list:
            # 获取地址,发请求,提取所有的代码和名称
            two_url = 'http://www.mca.gov.cn' + href_list[0]
            finger = self.md5_url(two_url)
            # 返回值1：说明redis集合中之前不存在此指纹,即之前没有抓取过
            if self.r.sadd('mzb:spiders', finger) == 1:
                self.parse_two_page(two_url)
            else:
                sys.exit('更新完成')
        else:
            print('提取最新月份链接失败')

    def parse_two_page(self, two_url):
        """提取具体数据的函数"""
        two_html = self.get_html(url=two_url)
        # 从two_html中提取真实返回数据的URL地址
        real_url = self.get_real_url(two_html)
        real_html = self.get_html(url=real_url)
        two_xpath = '//tr[@height="19"]'
        # tr_list: [<element tr at xxx>, <element tr at xxx>, <element tr at xxx>, ...]
        tr_list = self.xpath_func(html=real_html, xpath_bds=two_xpath)
        item = {}
        for tr in tr_list:
            item['name'] = tr.xpath('./td[3]/text()')[0].strip()
            item['code'] = tr.xpath('./td[2]/text() | ./td[2]/span/text()')[0].strip()
            print(item)

    def get_real_url(self, two_html):
        """获取真实返回数据的URL地址"""
        regex = 'window.location.href="(.*?)"'
        pattern = re.compile(regex, re.S)
        real_url = pattern.findall(two_html)[0]

        return real_url

    def run(self):
        self.parse_html()

if __name__ == '__main__':
    spider = MzbSpider()
    spider.run()















