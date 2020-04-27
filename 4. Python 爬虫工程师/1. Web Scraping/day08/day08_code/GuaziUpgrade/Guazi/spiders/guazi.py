# -*- coding: utf-8 -*-
import scrapy
from ..items import GuaziItem


class GuaziSpider(scrapy.Spider):
    # 爬虫名
    name = 'guazi'
    # 允许爬取的域名
    allowed_domains = ['www.guazi.com']
    # 初始的URL地址
    start_urls = ['https://www.guazi.com/langfang/buy/o1/#bread']
    # 生成URL地址的变量
    n = 1

    def parse(self, response):
        # 基准xpath: 匹配所有汽车的节点对象列表
        li_list = response.xpath('//ul[@class="carlist clearfix js-top"]/li')
        # 给items.py中的 GuaziItem类 实例化
        item = GuaziItem()
        for li in li_list:
            item['url'] = li.xpath('./a[1]/@href').extract()[0]
            item['name'] = li.xpath('.//h2[@class="t"]/text()').extract()[0]
            item['price'] = li.xpath('.//div[@class="t-price"]/p/text()').extract()[0]

            # 把抓取的数据,传递给了管道文件 pipelines.py
            yield item

        # 1页数据抓取完成,生成下一页的URL地址,交给调度器入队列
        if self.n < 5:
            self.n += 1
            url = 'https://www.guazi.com/langfang/buy/o{}/#bread'.format(self.n)
            # 把url交给调度器入队列
            yield scrapy.Request(url=url, callback=self.parse)

        # 所抓数据交给管道: yield item
        # 继续跟进的URL地址:yield scrapy.Request(url=url,callback=self.xxx)
