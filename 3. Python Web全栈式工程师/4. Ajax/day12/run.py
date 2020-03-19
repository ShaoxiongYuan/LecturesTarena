import json

from flask import Flask
from flask import render_template#响应模板的函数
from flask import request#请求对象 包含前端所有和请求有关的数据


app = Flask(__name__)

#http://127.0.0.1:5000/
@app.route('/')
def index_view():
    return render_template('homework.html')


@app.route('/server')
def server():
    kw = request.args.get('kw')#python
    search_list = ['的历史','简介','官网','发展前景']
    #data = 'python的历史|python简介|python官网|python发展前景|'
    data = ''
    for i in search_list:
        data += kw+i+'|'
    return data


# @app.route('/post')
# def post_view():
#     return render_template('ajax_post.html')
# @app.route('/post_server',methods=['post'])
# def post_server():
#     #接收post请求中携带的参数
#     print(request.form)
#     uname = request.form.get('uname')
#     return "欢迎%s" % uname

@app.route('/post',methods=['get','post'])
def post_view():
    #如果用户直接访问地址127.0.0.1:5000/post (get)将页面显示给用户
    #如果用户在页面中点击发送post请求 处理请求中的数据并返回结果
    if request.method == 'GET':
        return render_template('ajax_post.html')
    elif request.method == 'POST':
        uname = request.form.get('uname')
        return '欢迎%s' % uname


@app.route('/getData')
def get_data():
    return render_template('getData.html')

@app.route('/getData_server')
def getData_server():
    #flask允许的响应类型有 字符串、字典、元组和其他封装好的响应类型
    #字典 flask1.1.1版本新增的允许传递的类型
    #元组 是指flask返回值时的参数可以用元组的方式添加 但书记不能是自定义元组
    # lis = ["laowang","Maira","QTX"]
    #将列表转换成符合JSON格式的字符串
    #import json
    #json.dumps    列表-->JSON串
    # data = json.dumps(lis)

    dic = {"uname":"laowang","age":18}
    # separators参数表示转成JSON串以后的分隔符，默认键值对之间用', '分隔，键和值之间用': '分隔
    #如果不需要无意义的空格时，可以指定一个元组(',',':')替换默认值
    data = json.dumps(dic,separators=(',',':'),sort_keys=True)
    return data




@app.route('/exer2')
def exer2_view():
    return render_template('exer2.html')


@app.route('/exer2_server')
def exer2_server():
    uname = request.args.get('uname')
    lis = ['QTX','laowang','Maria']
    if uname in lis:
        data = json.dumps({"code":1000,"msg":"用户名已存在"})
    else:
        data = json.dumps({"code":1001,"msg":"OK"})
    return data


@app.route('/register',methods=['post'])
def register_view():
    print(request.json)
    uname = request.json.get('uname')
    pwd = request.json.get('pwd')
    print(uname,pwd)
    return '注册成功'


if __name__ == "__main__":
    app.run(debug=True)