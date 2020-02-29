"""
服务端
数据处理部分
"""

import pymysql

class Database:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 3306
        self.user = 'root'
        self.password = '123456'
        self.charset='utf8'
        self.database = 'dict'
        self.connect_database()

    def connect_database(self):
        self.db = pymysql.connect(host = self.host,
                                  port = self.port,
                                  user = self.user,
                                  password =self.password,
                                  database = self.database,
                                  charset = self.charset)

    def create_cursor(self):
        self.cur = self.db.cursor()

    # 帮 server 处理注册 成功 True 失败返回 False
    def register(self,name,passwd):
        # 判断这个姓名的用户是否存在
        sql = "select name from user where name=%s;"
        self.cur.execute(sql,[name])
        r = self.cur.fetchone() # 如果查询到了说明该用户已经存在
        if r:
            return False  # 不可注册
        else:
            # 插入数据库
            try:
                sql = "insert into user (name,passwd) values (%s,%s);"
                self.cur.execute(sql,[name,passwd])
                self.db.commit()
                return True
            except:
                self.db.rollback()
                return False

    # 处理登录
    def login(self,name,passwd):
        sql = "select name from user where name=%s and passwd=%s;"
        self.cur.execute(sql,[name,passwd])
        r = self.cur.fetchone()
        if r:
            return True
        else:
            return False

    # 查询单词
    def query(self,word):
        sql = "select mean from words where word=%s;"
        self.cur.execute(sql,[word])
        r = self.cur.fetchone() # 也有可能查不到

        if r:
            return r[0] # 将单词解释返回
        else:
            return "没有找到该单词"

    # 插入历史记录
    def insert_history(self,name,word):
        sql = "insert into hist (name,word) values (%s,%s);"
        try:
            self.cur.execute(sql,[name,word])
            self.db.commit()
        except:
            self.db.rollback()