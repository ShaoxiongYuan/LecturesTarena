# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class DaomuItem(scrapy.Item):
    # define the fields for your item here like:
    # 1、一级页面: 大标题+链接
    parent_title = scrapy.Field()
    parent_url = scrapy.Field()
    # 2、二级页面：小标题+链接
    son_title = scrapy.Field()
    son_url = scrapy.Field()
    # 3、三级页面：小说内容
    content = scrapy.Field()
    # 4、目录
    directory = scrapy.Field()
