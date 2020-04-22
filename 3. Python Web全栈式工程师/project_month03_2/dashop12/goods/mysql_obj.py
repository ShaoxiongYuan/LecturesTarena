import pymysql
from .common import write_log


class Mysql(object):
    # mysql 端口号,注意：必须是int类型
    def __init__(self,host,user,passwd,db_name,port=3306):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db_name = db_name
        self.port = port

    def select(self,sql):
        """
        执行sql命令
        :param sql: 命令
        :return: 元祖
        """
        try:
            conn = pymysql.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                port=self.port,
                database=self.db_name,
                charset='utf8',
                cursorclass=pymysql.cursors.DictCursor
            )
            cur = conn.cursor()  # 创建游标
            cur.execute(sql)  # 执行sql命令
            res = cur.fetchall()  # 获取执行的返回结果
            cur.close()
            conn.close()  # 关闭mysql 连接
            return res
        except Exception as e:
            print(e)
            return False

    def update(self, sql):
        """
        更新操作，比如insert, delete,update
        :param sql: sql命令
        :return: bool
        """
        try:
            conn = pymysql.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                port=self.port,
                database=self.db_name,
            )
            cur = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 创建游标
            sta = cur.execute(sql)  # 执行sql命令，返回影响的行数
            if isinstance(sta,int):  # 判断返回结果, 是数字就是正常的
                pass
            else:
                write_log('错误，远程执行sql: %s 失败'%sql)
                return False

            conn.commit()  # 主动提交，否则执行sql不生效
            cur.close()
            conn.close()  # 关闭mysql 连接
            return sta
        except Exception as e:
            print(e)
            return False