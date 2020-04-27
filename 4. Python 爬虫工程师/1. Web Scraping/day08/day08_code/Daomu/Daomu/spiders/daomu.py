# -*- coding: utf-8 -*-
import scrapy
from ..items import DaomuItem
import os


class DaomuSpider(scrapy.Spider):
    name = 'daomu'
    allowed_domains = ['www.daomubiji.com']
    start_urls = ['http://www.daomubiji.com/']

    def parse(self, response):
        """一级页面解析：提取大标题和链接"""
        a_list = response.xpath('//li[contains(@id,"menu-item-20")]/a')
        for a in a_list:
            # 思考: 此处是否需要继续交给调度器入队列？-需要！创建item对象
            item = DaomuItem()
            item['parent_title'] = a.xpath('./text()').get()
            item['parent_url'] = a.xpath('./@href').get()
            directory = './novel/{}/'.format(item['parent_title'])
            item['directory'] = directory
            # 创建对应目录
            if not os.path.exists(directory):
                os.makedirs(directory)
            # 继续交给调度器入队列
            yield scrapy.Request(
                url=item['parent_url'], meta={'meta_1': item}, callback=self.detail_page)

    def detail_page(self, response):
        """二级页面解析：提取小标题名字、链接"""
        meta1_item = response.meta['meta_1']
        article_list = response.xpath('//article')
        for article in article_list:
            # 又有继续交给调度器入队列的请求了
            item = DaomuItem()
            item['son_title'] = article.xpath('./a/text()').get()
            item['son_url'] = article.xpath('./a/@href').get()
            item['parent_title'] = meta1_item['parent_title']
            item['parent_url'] = meta1_item['parent_url']
            item['directory'] = meta1_item['directory']
            # 把每一个章节的item对象交给调度器入队列
            yield scrapy.Request(
                url=item['son_url'], meta={'meta_2': item}, callback=self.get_content)

    def get_content(self, response):
        """三级页面：提取小说具体内容"""
        # 最后一级页面,没有继续交给调度器入队列的请求了,所以不需要创建item对象了
        item = response.meta['meta_2']
        # content_list: ['段落1','段落2','段落3','段落4']
        content_list = response.xpath('//article[@class="article-content"]/p/text()').extract()
        item['content'] = '\n'.join(content_list)

        # 1条数据彻底搞完,交给管道文件处理
        yield item
