# -*- coding: utf-8 -*-
import scrapy


class CarSpider(scrapy.Spider):
    name = 'car'
    allowed_domains = ['www.guazi.com']
    start_urls = ['https://www.guazi.com/dachang/buy/']

    def parse(self, response):
        li_list = response.xpath('//ul[@class="carlist clearfix js-top"]/li')
        for li in li_list:
            item = {}
            # 1、xpath() 得到的一定是列表,列表中为 选择器对象
            #    结果：[<seletor xpath='xxx' data='数据1'>,<selector xpath='xxx' data='数据2'>]
            # 2、extract() ：把列表中所有的选择器对象序列化为unicode字符串
            #    结果：['数据1','数据2']
            # 3、extract_first() ：序列化提取第1个选择器的数据
            #    结果：'数据1'
            # 4、get() ：作用同 extract_first()
            #    结果：'数据1'
            item['name'] = li.xpath('.//h2[@class="t"]/text()').get()
            item['price'] = li.xpath('.//div[@class="t-price"]/p/text()').get()
            item['url'] = li.xpath('./a[1]/@href').get()

            print(item)
