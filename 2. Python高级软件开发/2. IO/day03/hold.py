"""
空洞文件
"""

f = open("皮卡丘",'wb')

f.write(b'begin')
f.seek(1024,2)
f.write(b'end')

f.close()
