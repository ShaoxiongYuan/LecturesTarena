# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


class DaomuPipeline(object):
    def process_item(self, item, spider):
        # 最终目标:  ./novel/盗墓笔记1:七星鲁王宫/七星鲁王_第一章_血尸.txt
        # directory: ./novel/盗墓笔记1:七星鲁王宫/
        filename = item['directory'] + item['son_title'].replace(' ', '_') + '.txt'
        with open(filename, 'w') as f:
            f.write(item['content'])

        return item
