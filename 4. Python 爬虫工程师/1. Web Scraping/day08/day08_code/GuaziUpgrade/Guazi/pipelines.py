# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

# 管道1 - 从终端打印输出汽车信息
class GuaziPipeline(object):
    # item: 从爬虫文件中 yield item 过来的这个item对象
    def process_item(self, item, spider):
        # 简单打印输出测试
        print(item['name'], item['price'], item['url'])
        return item


# 管道2 - 存入MySQL数据库
import pymysql
from .settings import *


class GuaziMysqlPipeline(object):
    def open_spider(self, spider):
        """爬虫项目启动时,只执行1次,一般用于数据库的连接"""
        self.db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PWD, MYSQL_DB, charset=CHARSET)
        self.cursor = self.db.cursor()

    def process_item(self, item, spider):
        ins = 'insert into guazitab values(%s,%s,%s)'
        car_li = [item['name'], item['price'], item['url']]
        self.cursor.execute(ins, car_li)
        self.db.commit()

        return item

    def close_spider(self, spider):
        """爬虫程序结束时,只执行1次,一般用于数据库的断开"""
        self.cursor.close()
        self.db.close()


# 管道3 - 存入MongoDB数据库
import pymongo


class GuaziMongoPipeline(object):
    def open_spider(self, spider):
        """爬虫项目启动时只执行1次,用于连接MongoDB数据库"""
        self.conn = pymongo.MongoClient(MONGO_HOST, MONGO_PORT)
        self.db = self.conn[MONGO_DB]
        self.myset = self.db[MONGO_SET]

    def process_item(self, item, spider):
        car_dict = {
            'name': item['name'],
            'price': item['price'],
            'url': item['url'],
        }
        self.myset.insert_one(car_dict)
        return item
