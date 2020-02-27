"""
executemany用法
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

# 插入多条记录 ，数据操作
l = [
    ('Dave',15,'m',81),
    ('Ala',13,'w',79),
    ('Eva',14,'w',91),
    ('Baron',13,'m',61)
]

sql = "insert into cls (name,age,sex,score) values (%s,%s,%s,%s);"
try:
    # for i in l:
    #     cur.execute(sql,i)

    cur.executemany(sql,l) # 执行多次sql语句

    db.commit()
except Exception as e:
    print(e)
    db.rollback()



# 关闭游标和数据库
cur.close()
db.close()