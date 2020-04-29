# -*- coding: utf-8 -*-
import scrapy
from ..items import GuaziItem

class GuaziSpider(scrapy.Spider):
    name = 'guazi2'
    allowed_domains = ['www.guazi.com']
    # 1、删掉start_urls
    # 2、重写start_requests()方法
    def start_requests(self):
        """把所有要抓取的URL地址交给调度器入队列"""
        for o in range(1,6):
            url = 'https://www.guazi.com/dachang/buy/o{}/#bread'.format(o)
            # 交给调度器入队列
            yield scrapy.Request(url=url, callback=self.parse_one_page)

    def parse_one_page(self, response):
        """解析提取数据：名称、链接、价格"""
        li_list = response.xpath('//ul[@class="carlist clearfix js-top"]/li')
        for li in li_list:
            # 给 items.py 中的类GuaziItem实例化
            item = GuaziItem()
            item['name'] = li.xpath('./a[1]/@title').get()
            item['link'] = 'https://www.guazi.com' + li.xpath('./a[1]/@href').get()
            item['price'] = li.xpath('.//div[@class="t-price"]/p/text()').get()

            # 把详情页链接交给调度器入队列
            # meta参数：在不同的解析函数之间传递数据
            yield scrapy.Request(url=item['link'], meta={'item':item}, callback=self.parse_two_page)

    def parse_two_page(self, response):
        """二级页面解析函数：提取公里数、排量、类型"""
        # meta: 先到的调度器 - 又到了下载器 - 又作为了response的属性 传递给了指定callback解析函数
        item = response.meta['item']
        item['km'] = response.xpath('//ul[@class="assort clearfix"]/li[2]/span/text()').get()
        item['displace'] = response.xpath('//ul[@class="assort clearfix"]/li[3]/span/text()').get()
        item['typ'] = response.xpath('//ul[@class="assort clearfix"]/li[4]/span/text()').get()

        yield item





