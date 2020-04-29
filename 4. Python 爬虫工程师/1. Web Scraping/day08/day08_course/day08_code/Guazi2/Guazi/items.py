# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class GuaziItem(scrapy.Item):
    # define the fields for your item here like:
    # 名称、链接、价格
    name = scrapy.Field()
    link = scrapy.Field()
    price = scrapy.Field()

    # 行驶里程、排量、变速箱
    km = scrapy.Field()
    displace = scrapy.Field()
    typ = scrapy.Field()












