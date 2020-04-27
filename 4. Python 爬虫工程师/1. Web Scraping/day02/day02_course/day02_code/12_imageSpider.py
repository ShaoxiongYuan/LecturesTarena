"""
抓取图片到本地文件
"""
import requests
import os

url = 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1587471106058&di=ac551ff4da217437d789d11d7c772596&imgtype=0&src=http%3A%2F%2Fwww.people.com.cn%2Fmediafile%2Fpic%2F20170922%2F34%2F16265603900060641634.jpg'
headers = {'User-Agent':'Mozilla/5.0'}
html = requests.get(url=url, headers=headers).content
# 图片保存路径：/home/tarena/images/赵丽颖/
directory = '/home/tarena/images/赵丽颖/'
if not os.path.exists(directory):
    os.makedirs(directory)

# filename: /home/tarena/images/赵丽颖/xxxx.jpg
filename = directory + url[-10:]
with open(filename, 'wb') as f:
    f.write(html)












