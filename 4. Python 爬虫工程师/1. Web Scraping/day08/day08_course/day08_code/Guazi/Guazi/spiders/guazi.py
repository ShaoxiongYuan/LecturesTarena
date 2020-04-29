# -*- coding: utf-8 -*-
import scrapy
from ..items import GuaziItem

class GuaziSpider(scrapy.Spider):
    name = 'guazi'
    allowed_domains = ['www.guazi.com']
    start_urls = ['https://www.guazi.com/dachang/buy/o1/#bread']
    o = 1

    def parse(self, response):
        """解析提取数据：名称、链接、价格"""
        li_list = response.xpath('//ul[@class="carlist clearfix js-top"]/li')
        for li in li_list:
            # 给 items.py 中的类GuaziItem实例化
            item = GuaziItem()
            item['name'] = li.xpath('./a[1]/@title').get()
            item['link'] = li.xpath('./a[1]/@href').get()
            item['price'] = li.xpath('.//div[@class="t-price"]/p/text()').get()

            # 数据交给管道文件处理的方法 ：yield item
            yield item

        # 把下一页的地址交给调度器入队列
        if self.o < 5:
            self.o += 1
            next_url = 'https://www.guazi.com/dachang/buy/o{}/#bread'.format(self.o)
            yield scrapy.Request(url=next_url, callback=self.parse)


# 1、抓取数据交给管道：yield item
# 2、URL地址交给调度器：yield scrapy.Request(url=url, callback=self.xxx)






