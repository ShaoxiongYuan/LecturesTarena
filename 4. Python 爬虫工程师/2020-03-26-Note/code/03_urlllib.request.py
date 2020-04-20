from urllib import request

url = 'http://httpbin.org/get'
headers = {
    'User-Agent': 'xxxxxxxxxxxxxxxxxxx'
}

# 1、创建请求对象 - 包装请求头Request()
req = request.Request(url=url, headers=headers)
# 2、获取响应对象 - urlopen()
res = request.urlopen(req)
# 3、获取响应内容 - read().decode()
html = res.read().decode()
print(html)
