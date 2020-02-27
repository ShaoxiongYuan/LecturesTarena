"""
将dict.txt 单词本中的单词插入数据库中words表
"""
import pymysql
import re

# 链接数据库
db = pymysql.connect(host = 'localhost',
                     port = 3306,
                     user = 'root',
                     password = '123456',
                     database = 'dict',
                     charset = 'utf8')

# 创建游标  （操作数据 执行sql语句，获取结果）
cur = db.cursor()

# 插入单词
args_list = []
f = open('dict.txt')

for line in f:
    # 获取单词和解释
    t = re.findall(r"(\w+)\s+(.*)",line)[0]
    args_list.append(t)

sql = "insert into words (word,mean) values (%s,%s);"
try:
    cur.executemany(sql,args_list)
    db.commit()
except:
    db.rollback()



# 关闭游标和数据库
cur.close()
db.close()