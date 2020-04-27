"""
抓取百度图片网站中的首页30张图片
要求：
    1、输入关键字：赵丽颖
    2、创建对应目录： ./images/赵丽颖/
       保存文件名：赵丽颖_1.jpg  赵丽颖_2.jpg  赵丽颖_3.jpg ... 赵丽颖_30.jpg
"""
import requests
import re
from urllib import parse
import time
import random
from fake_useragent import UserAgent
import os

class BaiduImageSpider:
    def __init__(self):
        self.word = input('请输入关键字:')
        self.params = parse.quote(self.word)
        self.url = 'https://image.baidu.com/search/index?tn=baiduimage&word={}'.format(self.params)
        # 创建对应目录
        self.directory = './images/{}/'.format(self.word)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        # 定义计数变量,用于文件名命名
        self.i = 1

    def get_headers(self):
        """获取随机ua的功能函数"""
        headers = { 'User-Agent':UserAgent().random }

        return headers

    def get_images(self):
        """请求、解析、保存图片"""
        html = requests.get(url=self.url, headers=self.get_headers()).text
        pattern = re.compile('"thumbURL":"(.*?)"',re.S)
        # src_list: ['http://img1.jpg','http://img2.jpg','',...,'']
        src_list = pattern.findall(html)
        for src in src_list:
            # 保存1张图片到本地
            self.save_image(src)
            # 控制一下频率
            time.sleep(random.uniform(0,1))

    def save_image(self, src):
        """保存图片到本地文件"""
        html = requests.get(url=src, headers=self.get_headers()).content
        # filename: ./images/赵丽颖/赵丽颖_1.jpg
        filename = '{}{}_{}.jpg'.format(self.directory, self.word, self.i)
        with open(filename, 'wb') as f:
            f.write(html)
        self.i += 1
        print(filename,'下载成功')

    def run(self):
        self.get_images()

if __name__ == '__main__':
    spider = BaiduImageSpider()
    spider.run()






















