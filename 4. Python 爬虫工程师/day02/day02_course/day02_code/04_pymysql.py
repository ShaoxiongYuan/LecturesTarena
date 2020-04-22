"""
单条插入表记录：execute()
多条插入表记录：executemany()
"""
import pymysql

# 1、创建数据库连接对象
db = pymysql.connect('localhost', 'root', '123456', 'maoyandb', charset='utf8')
# 2、游标对象
cur = db.cursor()
# 3、执行SQL命令
ins = 'insert into maoyantab values(%s,%s,%s)'
cur.execute(ins, ['喜剧之王','周星驰','1993-01-01'])

li = [('大话西游之月光宝盒','周星驰','1993-01-01'),('大圣娶亲','周星驰','1994-01-01')]
cur.executemany(ins, li)

# 4、提交到数据库执行
db.commit()
# 5、关闭游标
cur.close()
# 6、断开数据库连接
db.close()










