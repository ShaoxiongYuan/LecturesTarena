# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


class TencentPipeline(object):
    def process_item(self, item, spider):
        return item


import pymysql
from .settings import *


class TencentMysqlPipeline(object):
    def open_spider(self, spider):
        """爬虫项目启动时,连接数据库1次"""
        self.db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PWD, MYSQL_DB, charset=CHARSET)
        self.cursor = self.db.cursor()

    def process_item(self, item, spider):
        ins = 'insert into tencenttab values(%s,%s,%s,%s,%s,%s)'
        job_li = [
            item['job_name'],
            item['job_type'],
            item['job_duty'],
            item['job_require'],
            item['job_address'],
            item['job_time']
        ]
        self.cursor.execute(ins, job_li)
        self.db.commit()

        return item

    def close_spider(self, spider):
        """爬虫项目结束时,断开数据库1次"""
        self.cursor.close()
        self.db.close()
