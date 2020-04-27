# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class PptItem(scrapy.Item):
    # define the fields for your item here like:
    # PPT类别、类别链接
    ppt_type = scrapy.Field()
    ppt_url = scrapy.Field()

    # PPT分类链接
    ppt_info_url = scrapy.Field()
    # PPT详情页链接
    ppt_info_url = scrapy.Field()
    # PPT下载链接
    ppt_down_url = scrapy.Field()

    # PPT名字
    ppt_name = scrapy.Field()
