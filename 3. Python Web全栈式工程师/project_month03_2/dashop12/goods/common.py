from . import es_conf
import time
import os
import sys
import ipaddress
import socket


# 1.日志写入功能
def write_log(content):
    """
    写入日志文件
    :param content: 写入内容
    :return:
    """
    path = os.path.join(es_conf.BASE_DIR, "es_output.log")  # 日志文件
    with open(path, mode='a+', encoding='utf-8') as f:
        content = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + content
        print(content)
        f.write(content + "\n")


# 2.验证ip有效性
def valid_ip(ip):
    """
    验证ip是否有效：256.256.256.256无效ip
    :return: bool
    """
    try:
        # 判断 python 版本
        if sys.version_info[0] == 2:
            ipaddress.ip_address(ip.strip().decode("utf-8"))
        elif sys.version_info[0] == 3:
            ipaddress.ip_address(ip)
        return True
    except Exception as e:
        print(e)
        return False


# 3.验证port有效性
def check_tcp(ip, port, timeout=1):
    """
    检测tcp端口
    :param ip: ip地址
    :param port: 端口号
    :param timeout: 超时时间
    :return: bool
    """
    flag = False
    try:
        socket.setdefaulttimeout(timeout)  # 整个socket层设置超时时间
        cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address = (str(ip), int(port))

        # 开始连接,功能与connect(address)相同，但是成功返回0，失败返回errno的值
        status = cs.connect_ex((address))
        cs.settimeout(timeout)

        if not status:
            flag = True

        return flag
    except Exception as e:
        print(e)
        return flag