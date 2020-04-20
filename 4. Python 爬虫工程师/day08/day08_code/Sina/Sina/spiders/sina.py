# -*- coding: utf-8 -*-
import scrapy
from ..items import SinaItem
import os


class SinaSpider(scrapy.Spider):
    name = 'sina'
    allowed_domains = ['sina.com.cn']
    start_urls = ['http://news.sina.com.cn/guide/']

    def parse(self, response):
        # 用来存放下一次交给调度器的所有Request - 所有小分类的请求
        items = []
        # 基准xpath: 提取所有大分类的节点对象列表
        div_list = response.xpath('//div[@id="tab01"]/div')
        for div in div_list:
            # 大分类名称+URL
            parent_name = div.xpath('./h3/a/text()').get()
            parent_url = div.xpath('./h3/a/@href').get()
            # 有的特殊大分类 get() 出来为None,里面并非新闻内容
            if parent_name and parent_url:
                # 小分类的 li 节点对象列表
                li_list = div.xpath('./ul/li')
                for li in li_list:
                    # 创建item对象: 继续交给调度器入队列的请求对象
                    item = SinaItem()
                    item['son_name'] = li.xpath('./a/text()').get()
                    item['son_url'] = li.xpath('./a/@href').get()
                    item['parent_name'] = parent_name
                    item['parent_url'] = parent_url
                    # son_directory: ./data/体育/NBA/
                    son_directory = './data/{}/{}/'.format(item['parent_name'], item['son_name'])
                    item['son_directory'] = son_directory
                    # 创建对应的目录结构
                    if not os.path.exists(son_directory):
                        os.makedirs(son_directory)

                    # 把每个小分类的item对象添加到列表中
                    items.append(item)

        # 1、大循环结束,items中存放了所有的小分类的 item 对象
        # 2、依次遍历交给调度器入队列,顺带把meta参数传递给下一个解析函数
        for item in items:
            yield scrapy.Request(url=item['son_url'], meta={'meta1': item}, callback=self.parse_son_url)

    def parse_son_url(self, response):
        """解析1个小分类的函数 - 提取新闻链接"""
        # 继续交给调度器入队列的items列表
        items = []
        meta1_item = response.meta['meta1']
        # 通过观察URL地址规律,新闻链接基本上都是以 大分类URL开头,且以.shtml结尾
        news_url_list = response.xpath('//a/@href').extract()
        for news_url in news_url_list:
            if news_url.startswith(meta1_item['parent_url']) and news_url.endswith('.shtml'):
                # 只要你想把URL地址交给调度器入队列了,说明你创建item对象的时刻到了
                item = SinaItem()
                item['news_url'] = news_url
                item['parent_name'] = meta1_item['parent_name']
                item['parent_url'] = meta1_item['parent_url']
                item['son_name'] = meta1_item['son_name']
                item['son_url'] = meta1_item['son_url']
                item['son_directory'] = meta1_item['son_directory']
                items.append(item)

        # 发送每条新闻的链接,到调度器入队列
        for item in items:
            yield scrapy.Request(url=item['news_url'], meta={'meta2': item}, callback=self.get_content)

    def get_content(self, response):
        """提取具体新闻内容的函数"""
        item = response.meta['meta2']
        item['news_head'] = response.xpath('//h1[@class="main-title"]/text()').get()
        item['news_content'] = ' '.join(response.xpath('//article[@class="article"]/p/text()').extract())

        yield item
