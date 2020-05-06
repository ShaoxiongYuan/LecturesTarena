import execjs

with open('translate.js', 'r', encoding='utf-8') as f:
    js_code = f.read()

loader = execjs.compile(js_code)
print(loader.call('e', 'hello', '320305.131321201'))