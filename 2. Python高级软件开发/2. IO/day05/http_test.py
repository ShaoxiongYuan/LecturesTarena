from  socket import *

s = socket()
s.bind(('127.0.0.1',8000))
s.listen(3)

c,addr = s.accept()

data = c.recv(1024)
print(data)

f = open('baidu.html')

data="""HTTP1.1 200 OK
Content-Type:text/html


""" + f.read()

c.send(data.encode())

c.close()
s.close()