# -*- coding: utf-8 -*-
import scrapy
from ..items import PptItem


class PptSpider(scrapy.Spider):
    name = 'ppt'
    allowed_domains = ['www.1ppt.com']
    start_urls = ['http://www.1ppt.com/xiazai/']

    def parse(self, response):
        """解析一级页面,获取所有大分类的名称和链接"""
        li_list = response.xpath('//div[@class="col_nav clearfix"]/ul/li')
        # 第一个元素为 "栏目分类" ,所以把它略过
        for li in li_list[1:]:
            item = PptItem()
            item['ppt_type'] = li.xpath('./a/text()').get()
            item['ppt_url'] = 'http://www.1ppt.com' + li.xpath('./a/@href').get()
            # 继续交给调度器入队列
            yield scrapy.Request(url=item['ppt_url'], meta={'meta1': item}, callback=self.get_total_page)

    def get_total_page(self, response):
        """获取每个分类的总页数,并依次交给调度器入队列"""
        meta1_item = response.meta['meta1']
        # 总页数
        last_page = response.xpath('//ul[@class="pages"]/li')[-1]
        try:
            total_page = int(last_page.xpath('./a/@href').get().split('.')[0].split('_')[-1])
            name = last_page.xpath('./a/@href').get().split('.')[0].split('_')[1]
            for i in range(1, total_page + 1):
                item = PptItem()
                ppt_page_url = meta1_item['ppt_url'] + 'ppt_{}_{}.html'.format(name, i)
                item['ppt_type'] = meta1_item['ppt_type']
                # 把当前类别下的所有页依次入队列
                yield scrapy.Request(url=ppt_page_url, meta={'meta2': item}, callback=self.get_ppt_info_url)
        except Exception as e:
            pass

    def get_ppt_info_url(self, response):
        """获取PPT详情页的链接"""
        meta2_item = response.meta['meta2']
        href_list = response.xpath('//ul[@class="tplist"]/li/a/@href').extract()
        for href in href_list:
            item = PptItem()
            item['ppt_info_url'] = 'http://www.1ppt.com' + href
            item['ppt_type'] = meta2_item['ppt_type']

            yield scrapy.Request(url=item['ppt_info_url'], meta={'meta3': item}, callback=self.get_ppt_down_url)

    def get_ppt_down_url(self, response):
        """获取具体的下载链接"""
        item = response.meta['meta3']
        item['ppt_down_url'] = response.xpath('//ul[@class="downurllist"]/li/a/@href').get()
        item['ppt_name'] = response.xpath('//div[@class="ppt_info clearfix"]/h1/text()').get()

        yield item
