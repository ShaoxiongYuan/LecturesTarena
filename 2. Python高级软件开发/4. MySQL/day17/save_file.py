"""
文件存取演示

准备工作
 alter table cls add image blob;

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

# 存入文件
# with open('zly.jpg','rb') as f:
#     data = f.read() # 字节串
# try:
#     sql = "update cls set image=%s where name='Lily';"
#     cur.execute(sql,[data])
#     db.commit()
# except:
#     db.rollback()

# 提取文件
sql = "select image from cls where name='Lily';"
cur.execute(sql)
data = cur.fetchone()  # (xxxxxxxxx,)
with open('mm.jpg','wb') as f:
    f.write(data[0])


# 关闭游标和数据库
cur.close()
db.close()
