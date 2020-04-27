# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class SinaItem(scrapy.Item):
    # define the fields for your item here like:
    # 一级页面: 大类名称、大类URL、小类名称、小类URL
    parent_name = scrapy.Field()
    parent_url = scrapy.Field()
    son_name = scrapy.Field()
    son_url = scrapy.Field()
    # 二级页面: 新闻链接
    news_url = scrapy.Field()
    # 三级页面: 新闻标题、新闻内容
    news_head = scrapy.Field()
    news_content = scrapy.Field()
    # 路径: ./data/体育/NBA/
    son_directory = scrapy.Field()
