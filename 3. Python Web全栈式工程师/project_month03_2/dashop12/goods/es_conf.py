import os
# 保存配置文件路径，log文件的保存路径与之相同
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# mysql
HOST = "127.0.0.1"
USER = "root"
PASSWD = "123456"
DB_NAME = "dadashop"
PORT = 3306

# elasticsearch
INDEX_NAME = "dadashop"
INDEX_TYPE = "_doc"
ES_IP = "127.0.0.1"

# 控制每次批处理数据量
MAXIMUM = 100