import execjs

with open('hello.js', 'r', encoding='utf-8') as f:
    js_code = f.read()

loader = execjs.compile(js_code)
print(loader.call('test', '周芷若'))