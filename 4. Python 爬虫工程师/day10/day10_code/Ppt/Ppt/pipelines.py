# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import scrapy
from scrapy.pipelines.files import FilesPipeline


class PptPipeline(FilesPipeline):
    def get_media_requests(self, item, info):
        yield scrapy.Request(url=item['ppt_down_url'], meta={'item': item})

    def file_path(self, request, response=None, info=None):
        item = request.meta['item']
        filename = '{}/{}.zip'.format(item['ppt_type'], item['ppt_name'])

        return filename
