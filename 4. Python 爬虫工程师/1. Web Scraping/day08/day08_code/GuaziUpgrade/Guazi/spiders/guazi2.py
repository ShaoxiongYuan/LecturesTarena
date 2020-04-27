# -*- coding: utf-8 -*-
import scrapy
from ..items import GuaziItem


class GuaziSpider(scrapy.Spider):
    # 爬虫名
    name = 'guazi2'
    # 允许爬取的域名
    allowed_domains = ['www.guazi.com']

    # 1、去掉start_urls变量
    # 2、重写 start_requests() 方法
    def start_requests(self):
        """生成所有要抓取的URL地址,一次性交给调度器入队列"""
        for i in range(1, 6):
            url = 'https://www.guazi.com/langfang/buy/o{}/#bread'.format(i)
            # scrapy.Request(): 把请求交给调度器入队列
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # 基准xpath: 匹配所有汽车的节点对象列表
        li_list = response.xpath('//ul[@class="carlist clearfix js-top"]/li')
        # 存放所有汽车详情页的Request对象
        for li in li_list:
            # 每辆汽车的请求对象
            item = GuaziItem()
            item['url'] = 'https://www.guazi.com' + li.xpath('./a[1]/@href').extract()[0]
            item['name'] = li.xpath('.//h2[@class="t"]/text()').extract()[0]
            item['price'] = li.xpath('.//div[@class="t-price"]/p/text()').extract()[0]

            # Request()中meta参数: 在不同解析函数之间传递数据,item数据会随着response一起返回
            yield scrapy.Request(url=item['url'], meta={'meta_1': item}, callback=self.detail_parse)

    def detail_parse(self, response):
        """汽车详情页的解析函数"""
        # 获取上个解析函数传递过来的 meta 数据
        item = response.meta['meta_1']
        item['time'] = response.xpath('//ul[@class="assort clearfix"]/li[1]/span/text()').get()
        item['km'] = response.xpath('//ul[@class="assort clearfix"]/li[2]/span/text()').get()
        item['disp'] = response.xpath('//ul[@class="assort clearfix"]/li[3]/span/text()').get()
        item['trans'] = response.xpath('//ul[@class="assort clearfix"]/li[4]/span/text()').get()

        # 1条数据最终提取全部完成,交给管道文件处理
        yield item
