"""
ftp 客户端
"""
from socket import *
import sys,time

# 全局变量
ADDR = ('127.0.0.1',8800)

# 查看文件列表，上传，下载，退出
class FTPClient:
    def __init__(self,sockfd):
        self.sockfd = sockfd

    def quit(self):
        self.sockfd.send(b'E')
        self.sockfd.close()
        sys.exit("谢谢使用")

    # 获取文件列表
    def list(self):
        self.sockfd.send(b'L') # 发送请求
        # 等待回复
        data = self.sockfd.recv(128).decode()
        if data == 'OK':
            # 接收文件列表
            data = self.sockfd.recv(4096)
            print(data.decode())
        else:
            print(data)  # 打印原因

    # 下载文件
    def get(self,filename):
        data = "G "+filename
        self.sockfd.send(data.encode()) # 发送请求
        # 等待回复
        data = self.sockfd.recv(128).decode()
        if data == 'OK':
            # 接收文件
            f = open(filename,'wb')
            # 循环接收文件写入本地
            while True:
                data = self.sockfd.recv(1024)
                # 文件接收完毕的标志
                if data == b'##':
                    break
                f.write(data)
            f.close()

        else:
            print(data)

    # 上传文件
    def put(self, filename):
        try:
            f = open(filename,'rb')
        except:
            print("文件不存在")
            return
        data = "P " + filename
        self.sockfd.send(data.encode())  # 发送请求
        # 等待回复
        data = self.sockfd.recv(128).decode()
        if data == 'OK':
            while True:
                data = f.read(1024)
                if not data:
                    time.sleep(0.1)
                    self.sockfd.send(b'##')  # 结束标志
                    break
                self.sockfd.send(data)
            f.close()
        else:
            print(data)

# 启动函数
def main():
    # 链接服务端
    s = socket()
    s.connect(ADDR)

    ftp = FTPClient(s) # 实例化对象，用于调用类中的方法，具体与服务端交互

    # 循环发送请求
    while True:
        print("=============命令选项==============")
        print("***          list            ***")
        print("***         get file         ***")
        print("***         put file         ***")
        print("***          quit            ***")
        print("===================================")
        cmd = input("输入命令：")

        # 根据不同的输入进行不同的处理
        if cmd == 'quit':
            ftp.quit()
        elif cmd == 'list':
            ftp.list()
        elif cmd[:3] == 'get':
            filename = cmd.split(' ')[-1]
            ftp.get(filename)
        elif cmd[:3] == 'put':
            filename = cmd.split(' ')[-1]
            ftp.put(filename)
        else:
            print("请输入正确命令")

if __name__ == '__main__':
    main()
