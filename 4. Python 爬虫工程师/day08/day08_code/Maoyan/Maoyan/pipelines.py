# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


# 管道1 - 打印输出
class MaoyanPipeline(object):
    def process_item(self, item, spider):
        print(item['name'], item['time'], item['star'])
        return item


# 管道2 - 存入MySQL
import pymysql
from .settings import *


class MaoyanMysqlPipeline(object):
    def open_spider(self, spider):
        self.db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PWD, MYSQL_DB, charset=CHARSET)
        self.cursor = self.db.cursor()

    def process_item(self, item, spider):
        ins = 'insert into maoyantab values(%s,%s,%s)'
        film_li = [item['name'], item['star'], item['time']]
        self.cursor.execute(ins, film_li)
        self.db.commit()

        return item

    def close_spider(self, spider):
        self.cursor.close()
        self.db.close()


# 管道3 - 存入MongoDB
import pymongo


class MaoyanMongoPipeline(object):
    def open_spider(self, spider):
        self.conn = pymongo.MongoClient(MONGO_HOST, MONGO_PORT)
        self.db = self.conn[MONGO_DB]
        self.myset = self.db[MONGO_SET]

    def process_item(self, item, spider):
        film_dict = {
            'name': item['name'],
            'star': item['star'],
            'time': item['time'],
        }
        self.myset.insert_one(film_dict)
