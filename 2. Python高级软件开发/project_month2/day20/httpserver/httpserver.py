"""
主程序代码

功能：
获取http请求
解析http请求
将请求发送给WebFrame
从WebFrame接收反馈数据
将数据组织为Response格式发送给客户端
"""
from socket import *
import sys
from threading import Thread
from config import *  # 导入用户配置
import re,json

# 服务器地址
ADDR = (HOST, PORT)

# 处理和webframe交互
def connect_frame(env):
    # 创建tcp客户端
    s = socket()
    s.connect((frame_ip,frame_port))

    try:
        # 将请求字典转换为json格式发送
        data = json.dumps(env)
        s.send(data.encode())
        # 等待数据
        data = s.recv(1024 * 1024 * 10).decode()
        return json.loads(data) # 转换为字典返回
    except:
        return



# 主体功能 (处理与浏览器交互，负责http协议部分操作)
class HTTPServer:
    def __init__(self):
        self.address = ADDR
        self.create_socket()
        self.bind()

    # 创建套接字
    def create_socket(self):
        # tcp套接字 与浏览器通信
        self.sockfd = socket()
        self.sockfd.setsockopt(SOL_SOCKET, SO_REUSEADDR, DEBUG)

    def bind(self):
        self.sockfd.bind(self.address)
        self.host = self.address[0]
        self.port = self.address[1]

    # 搭建并发服务
    def serve_forever(self):
        self.sockfd.listen(3)
        print("Listen the port %d" % self.port)
        while True:
            connfd, addr = self.sockfd.accept()
            print("Connect from ", addr)
            t = Thread(target=self.handle, args=(connfd,))
            t.setDaemon(True)
            t.start()

    # 具体处理浏览器请求
    def handle(self,connfd):
        request = connfd.recv(4096).decode()  # 接收http请求
        pattern = r"(?P<method>[A-Z]+)\s+(?P<info>/\S*)"
        try:
            env = re.match(pattern,request).groupdict()
        except:
            connfd.close()
            return
        else:
            data = connect_frame(env)  # 交给 frame 交互函数处理
            if data:
                # data ==> {'status':'200','data':'xxxxx'}
                self.response(connfd,data)

    # 组织给浏览器的响应格式
    def response(self,connfd,data):
        if data['status']  == '200':
            res = "HTTP/1.1 200 OK\r\n"
            res += "Content-Type:text/html\r\n"
            res += "\r\n"
            res += data['data']
        elif data['status']  == '404':
            res = "HTTP/1.1 404 Not Found\r\n"
            res += "Content-Type:text/html\r\n"
            res += "\r\n"
            res += data['data']
        elif data['status']  == '500':
            pass
        connfd.send(res.encode()) # 发送给浏览器

httpd = HTTPServer()
httpd.serve_forever()  # 启动服务
