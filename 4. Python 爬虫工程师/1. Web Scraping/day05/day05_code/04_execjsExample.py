"""
execjs模块使用
"""
import execjs

js_code = """
    function test(name){
        return "Hello," + name;
    }
"""
loader = execjs.compile(js_code)
print(loader.call('test', '赵敏'))

















