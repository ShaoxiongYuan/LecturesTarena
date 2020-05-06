# -*- coding: utf-8 -*-
import scrapy
import json
from ..items import SoItem

class SoSpider(scrapy.Spider):
    name = 'so'
    allowed_domains = ['image.so.com']
    # 重写start_requests()方法
    def start_requests(self):
        url = 'https://image.so.com/zjl?ch=beauty&sn={}&listtype=new&temp=1'
        for i in range(5):
            sn = i * 30
            page_url = url.format(sn)
            yield scrapy.Request(url=page_url, callback=self.parse)

    def parse(self, response):
        """提取数据：提取图片名称、链接"""
        html = json.loads(response.text)
        for one_img in html['list']:
            item = SoItem()
            item['title'] = one_img['title']
            item['imgurl'] = one_img['qhimg_url']

            # 把item对象交给管道文件处理
            yield item












