"""
数据库查询操作
"""

import pymysql

# 链接数据库
db = pymysql.connect(host = 'localhost',
                     port = 3306,
                     user = 'root',
                     password = '123456',
                     database = 'student',
                     charset = 'utf8')

# 创建游标  （操作数据 执行sql语句，获取结果）
cur = db.cursor()

# 查询数据库
sql = "select name,age,sex,score from cls where score>85;"
cur.execute(sql)  # 执行语句

# 获取结果方法1
# for i in cur:
#     print(i)

# 获取结果方法2
# print(cur.fetchone())  # 获取第一个查询结果
# print(cur.fetchall()) # 获取所有查询结果
print(cur.fetchmany(2)) #  获取2条结果


# 关闭游标和数据库
cur.close()
db.close()
