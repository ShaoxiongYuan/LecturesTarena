"""
mongodb无须提前建库建集合,会自动创建
库名：maoyandb
集合名：maoyanset
在maoyanset集合中插入1条文档：{'name':'肖申克的救赎', 'star':'ABC', 'time':'1970-01-01'}
"""
import pymongo

# 1、创建连接对象：连接到mongodb数据库
conn = pymongo.MongoClient('localhost', 27017)
# 2、创建库对象
db = conn['maoyandb']
# 3、创建集合对象
myset = db['maoyanset']
# 4、插入文档
myset.insert_one({'name':'肖申克的救赎','star':'ABC','time':'1970-01-01'})

myset.insert_many([{},{},{}])



























