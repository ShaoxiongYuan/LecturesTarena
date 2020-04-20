# -*- coding: utf-8 -*-
import scrapy
from ..items import MaoyanItem


class MaoyanSpider(scrapy.Spider):
    name = 'maoyan'
    allowed_domains = ['maoyan.com']
    url = 'https://maoyan.com/board/4?offset={}'

    # 1、去掉start_urls
    # 2、重写start_requests()方法
    def start_requests(self):
        """生成10页地址,交给调度器入队列"""
        for i in range(0, 91, 10):
            page_url = self.url.format(i)
            yield scrapy.Request(url=page_url, callback=self.parse_page)

    def parse_page(self, response):
        # 创建item对象
        item = MaoyanItem()
        # 基准xpath:节点对象列表
        dd_list = response.xpath('//dl[@class="board-wrapper"]/dd')
        for dd in dd_list:
            item['name'] = dd.xpath('.//p[@class="name"]/a/text()').get().strip()
            item['star'] = dd.xpath('.//p[@class="star"]/text()').get().strip()
            item['time'] = dd.xpath('.//p[@class="releasetime"]/text()').get().strip()

            # 数据交给管道
            yield item
