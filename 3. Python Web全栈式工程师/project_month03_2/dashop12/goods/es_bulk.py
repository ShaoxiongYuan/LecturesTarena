import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from . import es_conf
from .mysql_obj import Mysql
from .common import write_log, valid_ip, check_tcp


class ElasticObj:
    def __init__(self, timeout=3600):
        """
        :param timeout: 超时时间
        """
        self.index_name = es_conf.INDEX_NAME  # 索引名称
        self.index_type = es_conf.INDEX_TYPE  # 索引类型
        self.es_ip = es_conf.ES_IP  # es ip

        # 无用户名密码状态
        self.es = Elasticsearch([self.es_ip], port=9200, timeout=timeout)

    # 创建es索引
    def create_index(self):
        """
        创建索引
        :return: bool
        """
        # 创建映射
        # 使用python API 创建索引跟使用Restful基本一致
        _index_mappings = {
            # 设置字段
            "mappings": {
                self.index_type: {
                    "properties": {
                        "type": {"type": "keyword"},
                        "id": {"type": "keyword"},
                        "name": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart"
                        },
                        "caption":{
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart"
                        }
                    }
                }
            }
        }
        # 判断索引不存在时
        # es.indices：查看当前es服务器下index
        if self.es.indices.exists(index=self.index_name) is not True:
            # 创建索引
            try:
                res = self.es.indices.create(index=self.index_name, body=_index_mappings)
            except Exception as e:
                print(e)

            if not res:
                write_log("错误，创建索引{}失败".format(self.index_name))
                return False

            write_log("正常，创建索引{}成功".format(self.index_name))
            return True
        else:
            write_log("正常，索引{}已存在".format(self.index_name))
            return True

    # 数据批量插入
    def bulk_insert(self, table, data_list):
        """
        批量写入数据
        :param table: 表名
        :param data_list: 数据列表
        :return: bool
        """
        # 批量插入
        start_time = time.time()  # 开始时间
        actions = []  # 临时数据列表
        i = 0  # 计数值

        try:
            # 循环数据列表
            for data in data_list:
                action = {
                    "_index": self.index_name,
                    "_type": self.index_type,
                    "_id": data["id"],
                    "_source": {
                        'type': "sku",
                        'id': data['id'],
                        'name': data['name'],
                        'caption': data['caption'],
                    }
                }
                i += 1
                actions.append(action)  # 添加到列表
                if len(actions) == es_conf.MAXIMUM:  # 列表数量达到100时
                    helpers.bulk(self.es, actions)  # 批量插入数据
                    del actions[0:len(action)]  # 删除列表元素

            if i > 0:  # 不足100时,插入剩余数据
                helpers.bulk(self.es, actions)

            end_time = time.time()  # 结束时间
            t = round((end_time - start_time), 2)  # 计算耗时
            write_log("正常，{} 表写入ES {}条数据，用时{}s".format(table, i, t))
            return True
        except Exception as e:
            print(e)
            return False

    def has_table(self, db_name, target_table):
        """
        mysql表是否存在
        :return: bool
        """
        mysql_obj = Mysql(es_conf.HOST, es_conf.USER, es_conf.PASSWD, es_conf.DB_NAME, es_conf.PORT)
        sql = "select count(1) from {}.{}".format(db_name, target_table)
        res = mysql_obj.select(sql)

        if res is False:
            write_log("错误，mysql中 {}.{} 不存在".format(db_name, target_table))
            return False
        else:
            return True

    def has_es_mysql_conf(self):
        """
        判断配置文件中的mysql和es端口是否正常
        :return:
        """
        if not valid_ip(es_conf.HOST):
            write_log("错误，MySQL IP配置异常")
            return False

        if not valid_ip(es_conf.ES_IP):
            write_log("错误，ES IP配置异常")
            return False

        if not check_tcp(es_conf.HOST, es_conf.PORT):
            write_log("错误，MySQL {} 端口异常".format(es_conf.PORT))
            return False

        if not check_tcp(es_conf.ES_IP, 9200):
            write_log("错误，ES 9200 端口异常")
            return False

        return True

    def read_mysql_es(self):
        """
        读取goods_sku表的数据，并写入es
        :return: bool
        """
        # 判断配置文件中的mysql和es 端口是否正常
        if not self.has_es_mysql_conf():
            return False

        # 创建索引
        if self.create_index() is False:
            return False

        max = es_conf.MAXIMUM  # 一次性查询多少条

        mysql_obj = Mysql(es_conf.HOST, es_conf.USER, es_conf.PASSWD, es_conf.DB_NAME, es_conf.PORT)
        res = self.has_table(es_conf.DB_NAME, "goods_sku")
        # mysql table not exist
        if not res:
            return False
        id = 0  # 每一次查询后的最大id
        while True:
            # 查询数据 1-100。101-200。 201-300.。。
            sql = "select id, name, caption from goods_sku where id > %s order by id limit %s" % (id, max)
            print("======run sql======= ", sql, " ======run sql======= \n")
            data_list = mysql_obj.select(sql)
            print("=======sql result======== ", data_list, " =======sql result======== \n")
            if not data_list:  # 当结果为空时,结束循环
                write_log("警告，执行sql: %s 记录为空，无需写入es" % (sql))
                break  # 跳出循环

            last_row = data_list[-1]  # 最后一行记录
            id = last_row['id']  # 修改最大id

            res = self.bulk_insert('goods_sku', data_list)
            if not res:
                write_log("错误，goods_sku 写入ES 失败")
                return False

        write_log("正常，goods_sku 表全部写入ES成功")
        return True

    def main(self):
        self.read_mysql_es()


if __name__ == "__main__":
    ElasticObj().main()  # 执行主程序