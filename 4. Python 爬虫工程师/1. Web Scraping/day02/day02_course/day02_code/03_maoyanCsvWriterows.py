"""
猫眼电影top100抓取（电影名称、主演、上映时间）
使用writerows()方法实现 - 减少了磁盘IO次数,效率高
"""
import requests
import re
import time
import random
import csv

class MaoyanSpider:
    def __init__(self):
        self.url = 'https://maoyan.com/board/4?offset={}'
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
        # 打开文件、初始化写入对象、存放所有电影信息的大列表
        self.f = open('maoyan.csv', 'w', newline='')
        self.writer = csv.writer(self.f)
        self.all_list = []

    def get_html(self, url):
        html = requests.get(url=url, headers=self.headers).text
        # 直接调用解析函数
        self.parse_html(html)

    def parse_html(self, html):
        """解析提取数据"""
        regex = '<div class="movie-item-info">.*?title="(.*?)".*?<p class="star">(.*?)</p>.*?<p class="releasetime">(.*?)</p>'
        pattern = re.compile(regex, re.S)
        r_list = pattern.findall(html)
        # r_list: [('活着','牛犇','2000-01-01'),(),(),...,()]
        self.save_html(r_list)

    def save_html(self, r_list):
        """数据处理函数"""
        for r in r_list:
            t = ( r[0].strip(), r[1].strip(), r[2].strip() )
            self.all_list.append(t)
            print(t)

    def run(self):
        """程序入口函数"""
        for offset in range(0, 91, 10):
            url = self.url.format(offset)
            self.get_html(url=url)
            # 控制数据抓取频率:uniform()生成指定范围内的浮点数
            time.sleep(random.uniform(0,1))
        # 所有数据抓取完成后,一次性写入本地磁盘,并关闭文件
        self.writer.writerows(self.all_list)
        self.f.close()

if __name__ == '__main__':
    spider = MaoyanSpider()
    spider.run()











