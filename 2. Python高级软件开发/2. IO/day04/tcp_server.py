"""
tcp_server.py
tcp 服务端流程学习 ： 重点代码
"""

import socket

# 创建tcp套接字对象
sockfd = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# 绑定地址 ip: "localhost" "127.0.0.1"  "0.0.0.0"  "192.168.31.196"
sockfd.bind(("127.0.0.1",8888))

# 设置监听套接字
sockfd.listen(3)

# 处理客户端链接
while True:
    print("Waiting for connect..")
    connfd,addr = sockfd.accept()
    print("Connect from",addr)

    # 收发消息 (网络传输数据用字节串)
    while True:
        data = connfd.recv(1024)
        if not data:
            break
        # 收到b"##"则退出
        # if data == b"##":
        #     break
        print("Recv:",data.decode())
        n = connfd.send(b'Thanks')
        print("Send %d bytes"%n)
    connfd.close()

# 关闭套接字
sockfd.close()


