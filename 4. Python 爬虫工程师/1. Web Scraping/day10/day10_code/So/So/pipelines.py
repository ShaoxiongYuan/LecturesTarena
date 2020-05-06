# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.pipelines.images import ImagesPipeline
import scrapy
import os

class SoPipeline(ImagesPipeline):
    # 重写第一个方法：下载图片
    def get_media_requests(self, item, info):
        yield scrapy.Request(url=item['imgurl'], meta={'item':item})

    # 重写第二个方法：重命名
    def file_path(self, request, response=None, info=None):
        item = request.meta['item']
        filename = '{}{}'.format(item['title'], os.path.splitext(item['imgurl'])[-1])
        # 处理文件名中的特殊字符
        all_chars = '\/:*?"<>'
        for char in filename:
            if char in all_chars:
                filename = filename.replace(char, '')

        return filename









