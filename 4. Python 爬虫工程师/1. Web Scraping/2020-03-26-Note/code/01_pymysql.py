import pymysql

db = pymysql.connect('192.168.100.128', 'root', '123456', 'AID1911db', charset='utf8')
cursor = db.cursor()
ins = 'insert into stu_tab values(%s,%s)'
cursor.execute(ins, ['凯哥', '20200401'])
db.commit()
cursor.close()
db.close
